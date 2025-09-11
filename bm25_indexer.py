# bm25_indexer.py (Corrected Path Version)

import json
import pickle
import logging
from tqdm import tqdm
from rank_bm25 import BM25Okapi
import config

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_bm25_index():
    """
    Reads text chunks from the JSONL file, validates them using the correct nested path, 
    creates a BM25 index, and saves the index and the corpus to disk.
    """
    logging.info(f"Loading chunks from {config.CHUNKS_JSONL}...")
    
    valid_chunks_metadata = []
    
    try:
        with open(config.CHUNKS_JSONL, 'r', encoding='utf-8') as f_in:
            for i, line in enumerate(tqdm(f_in, desc="Loading and validating chunks")):
                chunk_data = json.loads(line)
                
                # --- THIS IS THE FIX ---
                # Validate that the essential keys exist using the correct nested path.
                if 'text' not in chunk_data or 'metadata' not in chunk_data or 'source_file' not in chunk_data.get('metadata', {}):
                    logging.warning(f"Skipping malformed chunk at line {i+1}: Missing 'text' or nested 'source_file'.")
                    continue
                # --- END FIX ---
                
                valid_chunks_metadata.append(chunk_data)

    except FileNotFoundError:
        logging.error(f"Chunk file not found at {config.CHUNKS_JSONL}. Please run the ingest pipeline first.")
        return

    if not valid_chunks_metadata:
        logging.error("No valid chunks found to index. Please check your chunks.jsonl file.")
        return
        
    corpus = [chunk['text'] for chunk in valid_chunks_metadata]
        
    logging.info(f"Loaded {len(corpus)} valid chunks. Tokenizing corpus for BM25...")
    tokenized_corpus = [doc.lower().split(" ") for doc in tqdm(corpus, desc="Tokenizing")]
    
    logging.info("Creating the BM25 index...")
    bm25 = BM25Okapi(tokenized_corpus)
    
    bm25_index_path = config.DATA_DIR + "/bm25_index.pkl"
    corpus_metadata_path = config.DATA_DIR + "/corpus_metadata.pkl"

    with open(bm25_index_path, "wb") as f:
        pickle.dump(bm25, f)
    logging.info(f"BM25 index saved to {bm25_index_path}")
    
    with open(corpus_metadata_path, "wb") as f:
        pickle.dump(valid_chunks_metadata, f)
    logging.info(f"Corpus metadata saved to {corpus_metadata_path}")


if __name__ == "__main__":
    create_bm25_index()