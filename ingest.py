import os
import json
import re
import logging
import glob
import ray
from tqdm import tqdm
import uuid
from ollama import Client
# --- Project Imports ---
import config
from utils import split_list
from chunk import chunk_text_intelligently 

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("ingestion_hierarchical.log", "w"), logging.StreamHandler()]
)

# --- MODIFIED: Function now accepts the host IP ---
def generate_summary(text: str, ollama_host: str) -> str:
    """
    Generates a high-quality, structured summary using a specific on-device LLM host.
    """
    try:
        # --- THIS IS THE FIX ---
        # The client now connects to the specific host assigned to its worker.
        client = Client(host=ollama_host)

        prompt = f"""
You are a meticulous research analyst. Your task is to create a structured, information-dense summary of the following document. Do not add any conversational text or introductions.

Your output must contain these four sections, and only these four sections:
1. **Main Thesis**: A single, concise sentence that explains the absolute core argument or purpose of the text.
2. **Key Entities**: A comma-separated list of the most important people, places, organizations, or specific terms mentioned.
3. **Core Arguments**: A numbered list of the 3-5 primary claims or arguments made in the text. Each point should be a complete sentence.
4. **Critical Facts & Data**: A bulleted list of any specific, verifiable facts, statistics, or data points mentioned in the text.

Document:
\"\"\"
{text}
\"\"\"

Structured Summary:
"""
        response = client.chat(
            model=config.SUMMARIZATION_MODEL,
            messages=[{'role': 'user', 'content': prompt}],
        )
        summary = response['message']['content']
        return summary.strip()
    except Exception as e:
        logging.error(f"Failed to generate summary with host {ollama_host}: {e}")
        logging.warning("Falling back to simple text truncation for summary.")
        summary = text[:500].strip()
        if len(text) > 500:
            summary += "..."
        return summary
    # --- END FIX ---

def clean_snippet_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'\s*\.{3,}\s*', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- MODIFIED: Function now accepts the host IP to pass it down ---
def parse_and_process_document(file_path, ollama_host: str):
    """
    Reads a JSON file, creates a summary for the entire document using a specific
    Ollama host, and then chunks the document.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if not data.get('question') or not data.get('search_results'):
            return None

        doc_id = os.path.basename(file_path)
        snippets = [clean_snippet_text(result.get('snippet') or '') for result in data.get('search_results', [])]
        combined_text = "\n\n".join(filter(None, snippets)).strip()

        if not combined_text:
            return None

        # --- NEW: Pass the assigned host to the summary function ---
        doc_summary = generate_summary(combined_text, ollama_host=ollama_host)
        summary_record = {
            "doc_id": doc_id,
            "text": doc_summary,
            "metadata": {
                "source_file": doc_id,
                "original_question": data.get('question', '')
            }
        }
        
        text_chunks = chunk_text_intelligently(combined_text)
        
        chunk_records = []
        for i, chunk_text in enumerate(text_chunks):
            chunk_metadata = {
                "source_file": doc_id,
                "category": data.get('category', '').strip(),
                "chunk_index": i,
            }
            
            chunk_records.append({
                "chunk_id": str(uuid.uuid4()),
                "doc_id": doc_id,
                "text": chunk_text,
                "metadata": chunk_metadata
            })
            
        return {"summary": summary_record, "chunks": chunk_records}

    except Exception as e:
        logging.error(f"Error processing file {os.path.basename(file_path)}: {e}", exc_info=False)
        return None

# --- MODIFIED: Ray worker now determines its assigned host ---
@ray.remote
def process_file_batch(files, worker_id):
    """
    A Ray worker that processes a batch of files. It determines its assigned
    Ollama host and passes it to the processing function.
    """
    # --- THIS IS THE FIX ---
    # Assign an Ollama host to this worker based on its ID
    num_hosts = len(config.OLLAMA_HOSTS)
    host_for_this_worker = config.OLLAMA_HOSTS[worker_id % num_hosts]
    logging.info(f"Worker {worker_id} assigned to Ollama host: {host_for_this_worker}")
    # --- END FIX ---

    all_summaries = []
    all_chunks = []
    for file_path in files:
        if file_path.endswith('.json'):
            # Pass the assigned host to the document processor
            processed_data = parse_and_process_document(file_path, ollama_host=host_for_this_worker)
            if processed_data:
                all_summaries.append(processed_data['summary'])
                all_chunks.extend(processed_data['chunks'])
    return {"summaries": all_summaries, "chunks": all_chunks}

def ingest_and_chunk_data():
    input_dir = config.RAW_MD_DIR
    if not os.path.isdir(input_dir):
        logging.error(f"Input directory not found: {input_dir}")
        return

    all_files = glob.glob(os.path.join(input_dir, '*.json'))
    if not all_files:
        logging.warning(f"No JSON files found in {input_dir}")
        return
        
    logging.info(f"Found {len(all_files)} documents to process for hierarchical ingestion.")
    
    parts = split_list(all_files, config.NUM_WORKERS)
    
    # --- THIS IS THE FIX ---
    # Pass a unique worker_id to each Ray task
    futures = [process_file_batch.remote(part, i) for i, part in enumerate(parts) if part]
    # --- END FIX ---
    
    final_summaries = []
    final_chunks = []
    for batch in tqdm(ray.get(futures), desc="Processing file batches"):
        final_summaries.extend(batch['summaries'])
        final_chunks.extend(batch['chunks'])

    if final_summaries:
        os.makedirs(os.path.dirname(config.SUMMARIES_JSONL), exist_ok=True)
        with open(config.SUMMARIES_JSONL, 'w', encoding='utf-8') as f_out:
            for record in final_summaries:
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
        logging.info(f"✅ Saved {len(final_summaries)} summaries to: {config.SUMMARIES_JSONL}")
    else:
        logging.warning("No summaries were generated.")

    if final_chunks:
        os.makedirs(os.path.dirname(config.PARENT_CHUNKS_JSONL), exist_ok=True)
        with open(config.PARENT_CHUNKS_JSONL, 'w', encoding='utf-8') as f_out:
            for record in final_chunks:
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
        logging.info(f"✅ Saved {len(final_chunks)} chunks to: {config.PARENT_CHUNKS_JSONL}")
    else:
        logging.warning("No chunks were generated.")

if __name__ == "__main__":
    if not ray.is_initialized():
        ray.init()
    ingest_and_chunk_data()