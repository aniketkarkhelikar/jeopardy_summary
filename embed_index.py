# embed_index.py

import json
import logging
from tqdm import tqdm
from qdrant_client import QdrantClient, models
import ray
import config
from utils import split_list
import torch
from transformers import AutoTokenizer, AutoModel
import uuid # <-- 1. Import the uuid library

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# (The QwenEmbedder class is unchanged)
class QwenEmbedder:
    def __init__(self, model_name, device):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
        self.device = device
        self.model.eval()
        logging.info(f"Qwen model '{model_name}' loaded on {device}.")

    def encode(self, texts):
        with torch.no_grad():
            inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=config.MAX_CHUNK_TOKENS).to(self.device)
            last_hidden_state = self.model(**inputs).last_hidden_state
            attention_mask = inputs['attention_mask'].unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_embeddings = torch.sum(last_hidden_state * attention_mask, 1)
            sum_mask = torch.clamp(attention_mask.sum(1), min=1e-9)
            embeddings = sum_embeddings / sum_mask
            return embeddings.cpu().numpy()

# (setup_qdrant_collections is unchanged)
def setup_qdrant_collections(client):
    """Creates both the summary and chunk collections in Qdrant."""
    try:
        logging.info("Determining vector size from embedding model...")
        temp_model = AutoModel.from_pretrained(config.EMBEDDING_MODEL, trust_remote_code=True)
        vector_size = temp_model.config.hidden_size
        del temp_model
        
        logging.info(f"Creating collection '{config.SUMMARY_COLLECTION_NAME}'...")
        client.recreate_collection(
            collection_name=config.SUMMARY_COLLECTION_NAME,
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE)
        )
        
        logging.info(f"Creating collection '{config.CHUNK_COLLECTION_NAME}'...")
        client.recreate_collection(
            collection_name=config.CHUNK_COLLECTION_NAME,
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE)
        )
        logging.info("✅ Both collections created successfully.")
    except Exception as e:
        logging.error(f"Failed to create Qdrant collections: {e}")
        raise

@ray.remote(num_gpus=1 if torch.cuda.is_available() else 0)
def process_and_embed_batch(records, collection_name, qdrant_host):
    """A generic Ray worker to embed and upload records to a specified collection."""
    client = QdrantClient(host=qdrant_host, port=config.QDRANT_PORT)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = QwenEmbedder(config.EMBEDDING_MODEL, device=device)
    
    batch_to_process = []
    for record in tqdm(records, desc=f"Worker processing for {collection_name}"):
        batch_to_process.append(record)
        if len(batch_to_process) >= config.BATCH_SIZE:
            texts_to_embed = [item['text'] for item in batch_to_process]
            vectors = model.encode(texts_to_embed)
            
            points = []
            for i, item in enumerate(batch_to_process):
                # --- THIS IS THE FIX ---
                # Qdrant needs a UUID or int. Use the existing chunk_id (which is a UUID)
                # or generate a new, deterministic UUID from the filename (doc_id) for summaries.
                point_id = item.get('chunk_id')
                if not point_id:
                    doc_id_str = item.get('doc_id')
                    # Create a consistent UUID based on the filename
                    point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, doc_id_str))
                # --- END FIX ---
                points.append(models.PointStruct(
                    id=point_id,
                    vector=vectors[i].tolist(),
                    payload={**item.get('metadata', {}), "text": item.get('text', '')}
                ))

            client.upsert(collection_name=collection_name, points=points, wait=True)
            batch_to_process = []
    
    # Process the final leftover batch
    if batch_to_process:
        texts_to_embed = [item['text'] for item in batch_to_process]
        vectors = model.encode(texts_to_embed)
        
        points = []
        for i, item in enumerate(batch_to_process):
            # --- APPLY THE SAME FIX HERE ---
            point_id = item.get('chunk_id')
            if not point_id:
                doc_id_str = item.get('doc_id')
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, doc_id_str))
            # --- END FIX ---
            points.append(models.PointStruct(
                id=point_id,
                vector=vectors[i].tolist(),
                payload={**item.get('metadata', {}), "text": item.get('text', '')}
            ))
            
        client.upsert(collection_name=collection_name, points=points, wait=True)

# (embed_and_index_hierarchical is unchanged)
def embed_and_index_hierarchical():
    client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)
    setup_qdrant_collections(client)
    
    # --- STAGE 1: Process and Index Summaries ---
    logging.info(f"--- Stage 1: Processing Summaries from {config.SUMMARIES_JSONL} ---")
    with open(config.SUMMARIES_JSONL, 'r', encoding='utf-8') as f:
        all_summaries = [json.loads(line) for line in f]
    
    if all_summaries:
        summary_parts = split_list(all_summaries, config.NUM_WORKERS)
        summary_futures = [process_and_embed_batch.remote(part, config.SUMMARY_COLLECTION_NAME, config.QDRANT_HOST) for part in summary_parts if part]
        for future in tqdm(ray.get(summary_futures), desc="Indexing summaries"):
            pass
        logging.info(f"✅ Indexed {len(all_summaries)} summaries.")
    else:
        logging.warning("No summaries found to index.")

    # --- STAGE 2: Process and Index Chunks ---
    logging.info(f"--- Stage 2: Processing Chunks from {config.PARENT_CHUNKS_JSONL} ---")
    with open(config.PARENT_CHUNKS_JSONL, 'r', encoding='utf-8') as f:
        all_chunks = [json.loads(line) for line in f]
    
    if all_chunks:
        chunk_parts = split_list(all_chunks, config.NUM_WORKERS)
        chunk_futures = [process_and_embed_batch.remote(part, config.CHUNK_COLLECTION_NAME, config.QDRANT_HOST) for part in chunk_parts if part]
        for future in tqdm(ray.get(chunk_futures), desc="Indexing chunks"):
            pass
        logging.info(f"✅ Indexed {len(all_chunks)} chunks.")
    else:
        logging.warning("No chunks found to index.")
        
    logging.info("🎉 Hierarchical indexing complete!")

if __name__ == "__main__":
    if not ray.is_initialized():
        ray.init()
    embed_and_index_hierarchical()