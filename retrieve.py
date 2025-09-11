# retrieve.py

import logging
from qdrant_client import QdrantClient, models # <-- Ensure 'models' is imported
from collections import defaultdict
import config
import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import CrossEncoder

# (The QwenEmbedder class is unchanged, ensure it's here)
class QwenEmbedder:
    _instance = None
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(QwenEmbedder, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance
    def __init__(self, model_name, device):
        if self.initialized: return
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
        self.device = device; self.model.eval(); self.initialized = True
        logging.info(f"Qwen model '{model_name}' loaded on {device}.")
    def encode(self, texts):
        if isinstance(texts, str): texts = [texts]
        with torch.no_grad():
            inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=config.MAX_CHUNK_TOKENS).to(self.device)
            last_hidden_state = self.model(**inputs).last_hidden_state
            attention_mask = inputs['attention_mask'].unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_embeddings = torch.sum(last_hidden_state * attention_mask, 1)
            sum_mask = torch.clamp(attention_mask.sum(1), min=1e-9)
            embeddings = sum_embeddings / sum_mask
            return embeddings.cpu().numpy()

# --- MODIFIED: This is the new hierarchical retrieval function ---
def retrieve(query, top_k=config.TOP_K):
    """
    Performs a full two-stage hierarchical retrieval pipeline with reranking.
    """
    # --- SETUP ---
    logging.info(f"Starting HIERARCHICAL retrieval for query: '{query}'")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    qdrant_client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)
    embedder = QwenEmbedder(config.EMBEDDING_MODEL, device=device)
    cross_encoder = CrossEncoder(config.CROSS_ENCODER_MODEL, device=device)

    # === STAGE 1: SEARCH SUMMARIES TO FIND RELEVANT DOCUMENTS ===
    logging.info("Stage 1: Searching on document summaries...")

    # Prepend the BGE instruction to the query for optimal retrieval
    bge_instructed_query = f"Represent this sentence for searching relevant passages: {query}"
    query_vector = embedder.encode(bge_instructed_query)
    
    summary_results = qdrant_client.search(
        collection_name=config.SUMMARY_COLLECTION_NAME, # <-- Search summary collection
        query_vector=query_vector[0].tolist(),
        limit=config.RECALL_K,  # Retrieve the top 5 most relevant documents
        with_payload=True
    )
    
    # Extract the unique document IDs from the results
    relevant_doc_ids = list(set([point.payload['source_file'] for point in summary_results]))
    if not relevant_doc_ids:
        logging.warning("No relevant documents found in summary search. Cannot proceed.")
        return []
    
    logging.info(f"Found {len(relevant_doc_ids)} potentially relevant documents: {relevant_doc_ids}")

    # === STAGE 2: SEARCH CHUNKS WITHIN THE RELEVANT DOCUMENTS ===
    logging.info("Stage 2: Searching for specific chunks within those documents...")
    
    # Use a filter to search only within the chunks of the relevant documents
    chunk_results = qdrant_client.search(
        collection_name=config.CHUNK_COLLECTION_NAME, # <-- Search chunk collection
        query_vector=query_vector[0].tolist(),
        query_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="source_file", # The key in the payload to filter on
                    match=models.MatchAny(any=relevant_doc_ids) # Match any of the doc IDs
                )
            ]
        ),
        limit=config.RECALL_K, # Fetch a larger number of chunks for the reranker
        with_payload=True
    )

    if not chunk_results:
        logging.warning("No specific chunks found within the relevant documents.")
        return []

    # === STAGE 3: CROSS-ENCODER RERANKING ===
    chunks_for_reranking = [point.payload for point in chunk_results]
    logging.info(f"Stage 3: Reranking top {len(chunks_for_reranking)} chunks...")

    rerank_pairs = [[query, chunk['text']] for chunk in chunks_for_reranking]
    rerank_scores = cross_encoder.predict(rerank_pairs)

    reranked_chunks = []
    for i, chunk in enumerate(chunks_for_reranking):
        reranked_chunks.append({
            'chunk_data': chunk, 
            'score': rerank_scores[i]
        })
        
    reranked_chunks.sort(key=lambda x: x['score'], reverse=True)

    # === STAGE 4: AGGREGATE AND RANK SOURCE FILES ===
    logging.info("Stage 4: Aggregating chunk scores to rank source files...")
    file_scores = defaultdict(lambda: -1000.0) # Use a very low default score

    for item in reranked_chunks:
        # The payload from Qdrant contains all metadata
        source_file = item['chunk_data'].get('source_file')
        if source_file:
            score = item['score']
            # Use the highest score from any chunk belonging to that file
            if score > file_scores[source_file]:
                file_scores[source_file] = score

    sorted_files = sorted(file_scores.items(), key=lambda item: item[1], reverse=True)
    final_ranked_files = [file for file, score in sorted_files[:top_k]]
    
    logging.info(f"Top {len(final_ranked_files)} relevant files: {final_ranked_files}")
    return final_ranked_files