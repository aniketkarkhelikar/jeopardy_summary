#!/bin/bash

# This script creates a complete directory structure and all necessary files
# for the RAG (Retrieval-Augmented Generation) pipeline.

# Create project directory
PROJECT_DIR="rag_pipeline"
mkdir -p "$PROJECT_DIR"
echo "✅ Created project directory: $PROJECT_DIR"

# --- Create config.py ---
cat << 'EOF' > "${PROJECT_DIR}/config.py"
import os

# Paths
# IMPORTANT: Update RAW_MD_DIR to the path of your markdown files.
RAW_MD_DIR = "path/to/your/markdown/files/"
DATA_DIR = os.path.join(os.getcwd(), "data") # Use current working dir for data
NORMALIZED_JSONL = os.path.join(DATA_DIR, "md_doc.jsonl")
CHUNKS_JSONL = os.path.join(DATA_DIR, "chunks.jsonl")

# Schema and models
SCHEMA_LABEL = "md_doc"
TOKENIZER_MODEL = "sentence-transformers/all-mpnet-base-v2"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
MAX_CHUNK_TOKENS = 512

# Processing
BATCH_SIZE = 64
NUM_WORKERS = 4 # Adjust based on your CPU cores

# Qdrant
COLLECTION_NAME = "rag_competition_v1"
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
TOP_K = 10

# Ray
RAY_ADDRESS = None  # Set to 'ray://<head-ip>:10001' for cluster; None for local
EOF
echo "   -> Created config.py"

# --- Create utils.py ---
cat << 'EOF' > "${PROJECT_DIR}/utils.py"
def split_list(lst, n):
    """Splits a list into n roughly equal parts."""
    if n <= 0:
        raise ValueError("Number of parts must be positive.")
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]
EOF
echo "   -> Created utils.py"

# --- Create ingest.py ---
cat << 'EOF' > "${PROJECT_DIR}/ingest.py"
import os
import json
import re
import logging
import glob
import ray
from tqdm import tqdm
import config
from utils import split_list

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("ingestion_md.log", "w"), logging.StreamHandler()]
)

def clean_md(content):
    """Enhanced cleaning for MD content."""
    content = re.sub(r'!\[.*?\]\(.*?\)', '', content)
    content = re.sub(r'#### \*?Fig\. \d+\.?\d*\*?\n?.*?(century|Edition|battlefield)?', '', content, flags=re.IGNORECASE | re.DOTALL)
    content = re.sub(r'A terracotta sculpture \*depicting a scene from\* the Mahabharata \(West Bengal\), c. seventeenth century', '', content)
    content = re.sub(r'\*Fig\. \d+\.?\d*\*', '', content)
    content = re.sub(r'\n\s*\n+', '\n\n', content).strip()
    content = '\n'.join(line.strip() for line in content.split('\n'))
    return content

def parse_md(file_path):
    """Parser for Markdown files. Reads, cleans, and returns as a single record."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
            cleaned_content = clean_md(content)
            return [{"text": cleaned_content}]
    except Exception as e:
        logging.error(f"Error parsing MD file {os.path.basename(file_path)}: {e}")
        return []

@ray.remote
def process_files(files):
    """Process a batch of files in parallel."""
    all_records = []
    for file_path in tqdm(files, desc="Processing files in remote task"):
        records = parse_md(file_path)
        for record in records:
            record['orig_file'] = os.path.basename(file_path)
            record['schema'] = config.SCHEMA_LABEL
        if records:
            all_records.extend(records)
    return all_records

def ingest_md():
    if not os.path.isdir(config.RAW_MD_DIR):
        logging.error(f"Input directory not found: {config.RAW_MD_DIR}")
        logging.error("Please update RAW_MD_DIR in config.py")
        return

    md_files = glob.glob(os.path.join(config.RAW_MD_DIR, '*.md'))
    
    if not md_files:
        logging.warning(f"No Markdown (.md) files found in {config.RAW_MD_DIR}")
        return
        
    logging.info(f"Found {len(md_files)} markdown files to process.")
    parts = split_list(md_files, config.NUM_WORKERS)
    futures = [process_files.remote(part) for part in parts if part]
    final_records = [record for batch in ray.get(futures) for record in batch]

    if not final_records:
        logging.warning("No records were generated from any of the files.")
        return

    os.makedirs(config.DATA_DIR, exist_ok=True)
    with open(config.NORMALIZED_JSONL, 'w', encoding='utf-8') as f_out:
        for record in final_records:
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

    logging.info(f"Ingestion complete. {len(final_records)} records saved to: {config.NORMALIZED_JSONL}")
EOF
echo "   -> Created ingest.py"

# --- Create chunk.py ---
cat << 'EOF' > "${PROJECT_DIR}/chunk.py"
import os
import json
import logging
import uuid
import re
from tqdm import tqdm
from transformers import AutoTokenizer
import ray
import config
from utils import split_list

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
tokenizer = AutoTokenizer.from_pretrained(config.TOKENIZER_MODEL)

def chunk_md_text(text):
    paragraphs = re.split(r'\n\s*\n', text)
    chunks = [p.strip() for p in paragraphs if p.strip()]
    final_chunks = []
    for chunk in chunks:
        if len(tokenizer.encode(chunk)) > config.MAX_CHUNK_TOKENS:
            sentences = re.split(r'(?<=\.)\s+', chunk)
            sub_chunk = ''
            for sent in sentences:
                if len(tokenizer.encode(sub_chunk + ' ' + sent if sub_chunk else sent)) > config.MAX_CHUNK_TOKENS and sub_chunk:
                    final_chunks.append(sub_chunk.strip())
                    sub_chunk = sent
                else:
                    sub_chunk = sub_chunk + ' ' + sent if sub_chunk else sent
            if sub_chunk:
                final_chunks.append(sub_chunk.strip())
        else:
            final_chunks.append(chunk)
    return final_chunks

def chunk_single_document(record):
    try:
        md_chunks = chunk_md_text(record['text'])
        output_chunks = []
        for chunk_text in md_chunks:
            token_ids = tokenizer.encode(chunk_text, truncation=False, add_special_tokens=False)
            output_chunks.append({
                "chunk_id": str(uuid.uuid4()), "text": chunk_text, "orig_file": record['orig_file'],
                "schema": record['schema'], "chunk_len_tokens": len(token_ids)
            })
        return output_chunks
    except (KeyError, TypeError) as e:
        logging.warning(f"Skipping record due to bad format: {record.get('doc_id', 'Unknown ID')}. Error: {e}")
        return []

@ray.remote
def process_records(records):
    all_chunks = []
    for record in tqdm(records, desc="Chunking records in remote task"):
        chunks = chunk_single_document(record)
        if chunks:
            all_chunks.extend(chunks)
    return all_chunks

def chunk_md():
    logging.info("Starting MD chunking.")
    if not os.path.exists(config.NORMALIZED_JSONL):
        logging.error(f"Normalized file not found: {config.NORMALIZED_JSONL}")
        return
    
    records_to_process = []
    with open(config.NORMALIZED_JSONL, 'r', encoding='utf-8') as f_in:
        for i, line in enumerate(tqdm(f_in, desc="Loading records")):
            try:
                record = json.loads(line)
                record['doc_id'] = f"{record['orig_file']}-line{i}"
                records_to_process.append(record)
            except (json.JSONDecodeError, KeyError):
                continue

    parts = split_list(records_to_process, config.NUM_WORKERS)
    futures = [process_records.remote(part) for part in parts if part]
    final_chunks = [chunk for batch in ray.get(futures) for chunk in batch]

    os.makedirs(os.path.dirname(config.CHUNKS_JSONL), exist_ok=True)
    with open(config.CHUNKS_JSONL, 'w', encoding='utf-8') as f_out:
        for chunk in final_chunks:
            f_out.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    logging.info(f"✅ MD chunking complete! All chunks saved to: {config.CHUNKS_JSONL}")
EOF
echo "   -> Created chunk.py"

# --- Create embed_index.py ---
cat << 'EOF' > "${PROJECT_DIR}/embed_index.py"
import os
import json
import logging
from tqdm import tqdm
from qdrant_client import QdrantClient, models
import ray
import config
from utils import split_list

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def setup_qdrant_collection(host, port):
    client = QdrantClient(host=host, port=port)
    try:
        client.get_collection(collection_name=config.COLLECTION_NAME)
        logging.info(f"Collection '{config.COLLECTION_NAME}' already exists.")
    except Exception:
        logging.info(f"Creating collection '{config.COLLECTION_NAME}'...")
        from sentence_transformers import SentenceTransformer
        temp_model = SentenceTransformer(config.EMBEDDING_MODEL)
        vector_size = temp_model.get_sentence_embedding_dimension()
        del temp_model
        
        client.recreate_collection(
            collection_name=config.COLLECTION_NAME,
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE)
        )
        logging.info("Collection created successfully.")
    return client

def process_batch(batch, model, client):
    texts_to_embed = [chunk['text'] for chunk in batch]
    vectors = model.encode(texts_to_embed)
    points_to_upload = []
    for i, chunk in enumerate(batch):
        payload = {k: chunk.get(k, '') for k in ["text", "orig_file", "schema", "chunk_len_tokens"]}
        point_id = chunk.get('chunk_id')
        if point_id:
            points_to_upload.append(models.PointStruct(id=point_id, vector=vectors[i].tolist(), payload=payload))
    
    if points_to_upload:
        client.upsert(collection_name=config.COLLECTION_NAME, points=points_to_upload, wait=True)

@ray.remote(num_gpus=1 if "CUDA_VISIBLE_DEVICES" in os.environ else 0)
def process_chunks(chunks, qdrant_host):
    from sentence_transformers import SentenceTransformer
    from qdrant_client import QdrantClient
    
    client = QdrantClient(host=qdrant_host, port=config.QDRANT_PORT)
    model = SentenceTransformer(config.EMBEDDING_MODEL)
    
    batch_of_chunks = []
    for chunk in tqdm(chunks, desc="Processing chunks in remote task"):
        batch_of_chunks.append(chunk)
        if len(batch_of_chunks) >= config.BATCH_SIZE:
            process_batch(batch_of_chunks, model, client)
            batch_of_chunks = []
    if batch_of_chunks:
        process_batch(batch_of_chunks, model, client)
    logging.info(f"Remote task completed for {len(chunks)} chunks.")

def embed_and_index():
    setup_qdrant_collection(host=config.QDRANT_HOST, port=config.QDRANT_PORT)
    
    all_chunks = []
    with open(config.CHUNKS_JSONL, 'r', encoding='utf-8') as f_in:
        for line in tqdm(f_in, desc="Loading chunks"):
            try:
                all_chunks.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    
    logging.info(f"Loaded {len(all_chunks)} chunks from {config.CHUNKS_JSONL}.")
    if not all_chunks:
        logging.warning("No chunks to embed. Exiting.")
        return

    parts = split_list(all_chunks, config.NUM_WORKERS)
    futures = [process_chunks.remote(part, config.QDRANT_HOST) for part in parts if part]
    ray.get(futures)
    logging.info("All parallel tasks completed. ✅")
EOF
echo "   -> Created embed_index.py"

# --- Create retrieve.py ---
cat << 'EOF' > "${PROJECT_DIR}/retrieve.py"
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
import config

def retrieve(query, top_k=config.TOP_K, host=config.QDRANT_HOST, port=config.QDRANT_PORT):
    client = QdrantClient(host=host, port=port)
    model = SentenceTransformer(config.EMBEDDING_MODEL)
    query_vector = model.encode(query).tolist()

    results = client.search(
        collection_name=config.COLLECTION_NAME,
        query_vector=query_vector,
        limit=top_k,
        with_payload=True,
        score_threshold=0.5
    )

    retrieved = []
    for result in results:
        retrieved.append({
            "chunk_id": result.id,
            "text": result.payload.get('text', ''),
            "score": result.score,
            "metadata": {
                "orig_file": result.payload.get('orig_file', ''),
                "schema": result.payload.get('schema', ''),
                "chunk_len_tokens": result.payload.get('chunk_len_tokens', 0)
            }
        })
    return retrieved
EOF
echo "   -> Created retrieve.py"

# --- Create main.py ---
cat << 'EOF' > "${PROJECT_DIR}/main.py"
import argparse
import ray
import logging
import sys
from ingest import ingest_md
from chunk import chunk_md
from embed_index import embed_and_index
from retrieve import retrieve
import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    parser = argparse.ArgumentParser(description="Control the RAG pipeline.", formatter_class=argparse.RawTextHelpFormatter)
    subparsers = parser.add_subparsers(dest='command', required=True)

    subparsers.add_parser('ingest', help='Scan markdown files and normalize them.')
    subparsers.add_parser('chunk', help='Split normalized documents into smaller chunks.')
    subparsers.add_parser('embed', help='Generate vectors and upload to Qdrant.')
    retrieve_parser = subparsers.add_parser('retrieve', help='Search for relevant chunks based on a query.')
    retrieve_parser.add_argument('--query', required=True, help='The search query')
    retrieve_parser.add_argument('--top-k', type=int, default=config.TOP_K, help=f'Number of results to return (default: {config.TOP_K})')
    subparsers.add_parser('all', help='Run full pipeline: ingest -> chunk -> embed')

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
        
    args = parser.parse_args()

    if args.command != 'retrieve':
        try:
            ray.init(address=config.RAY_ADDRESS, ignore_reinit_error=True)
            logging.info("Ray cluster initialized.")
        except Exception as e:
            logging.error(f"Could not initialize Ray: {e}. Please ensure Ray is running.", exc_info=True)
            sys.exit(1)

    try:
        if args.command == 'ingest':
            ingest_md()
        elif args.command == 'chunk':
            chunk_md()
        elif args.command == 'embed':
            embed_and_index()
        elif args.command == 'retrieve':
            results = retrieve(args.query, top_k=args.top_k)
            print("\n--- Retrieved Chunks ---")
            if not results:
                print("No results found.")
            for i, res in enumerate(results):
                print(f"\n{i+1}. Score: {res['score']:.4f} | Source: {res['metadata']['orig_file']}")
                print(f"   Text: {res['text'][:300].strip()}...")
            print("\n----------------------")
        elif args.command == 'all':
            logging.info("--- Starting full pipeline: INGEST ---")
            ingest_md()
            logging.info("\n--- Continuing full pipeline: CHUNK ---")
            chunk_md()
            logging.info("\n--- Finishing full pipeline: EMBED ---")
            embed_and_index()
            logging.info("\n✅ Full pipeline completed successfully!")
    except Exception as e:
        logging.error(f"An error occurred during pipeline execution: {e}", exc_info=True)
    finally:
        if args.command != 'retrieve' and ray.is_initialized():
            ray.shutdown()
            logging.info("Ray cluster shut down.")

if __name__ == "__main__":
    main()
EOF
echo "   -> Created main.py"

# --- Create requirements.txt ---
cat << 'EOF' > "${PROJECT_DIR}/requirements.txt"
# Core libraries for the RAG pipeline
ray[default]>=2.9.0
tqdm
transformers>=4.0.0
sentence-transformers>=2.2.2
qdrant-client>=1.7.0
torch>=2.0.0
EOF
echo "   -> Created requirements.txt"

# --- Create run_pipeline.sh ---
cat << 'EOF' > "${PROJECT_DIR}/run_pipeline.sh"
#!/bin/bash

# run_pipeline.sh
# A shell script to control the RAG pipeline using main.py.
# ==========================================================
# Usage: ./run_pipeline.sh [command] [options]
#
# Commands:
#   ingest          - Scan and normalize markdown files.
#   chunk           - Split documents into smaller chunks.
#   embed           - Generate embeddings and upload to Qdrant.
#   all             - Run the full ingest -> chunk -> embed pipeline.
#   retrieve        - Search for chunks with a query.
#
# Examples:
#   ./run_pipeline.sh all
#   ./run_pipeline.sh retrieve --query "history of machine learning"
#   ./run_pipeline.sh retrieve --query "your query" --top-k 5
# ==========================================================

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "python3 could not be found. Please install Python 3."
    exit 1
fi

# Function to run the command
run_command() {
    local cmd=$1
    shift
    echo "Executing command: $cmd..."
    python3 main.py "$cmd" "$@"
}

# Parse the command
if [ $# -eq 0 ]; then
    echo "No command provided."
    python3 main.py --help
    exit 1
fi

COMMAND=$1
shift

case "$COMMAND" in
    ingest|chunk|embed|all|retrieve)
        run_command "$COMMAND" "$@"
        ;;
    *)
        echo "Invalid command: $COMMAND"
        python3 main.py --help
        exit 1
        ;;
esac

echo "Command '$COMMAND' finished."
EOF

# Make the script executable
chmod +x "${PROJECT_DIR}/run_pipeline.sh"
echo "   -> Created run_pipeline.sh and made it executable."

# --- Final Instructions ---
echo
echo "🚀 RAG Pipeline Codebase Created Successfully! 🚀"
echo
echo "--- Next Steps ---"
echo "1. Navigate to the project directory:"
echo "   cd ${PROJECT_DIR}"
echo
echo "2. (Recommended) Create and activate a virtual environment:"
echo "   python3 -m venv venv && source venv/bin/activate"
echo
echo "3. Install the required Python packages:"
echo "   pip install -r requirements.txt"
echo
echo "4. IMPORTANT: Edit the configuration file:"
echo "   - Open config.py with a text editor."
echo "   - Change 'RAW_MD_DIR' to the actual path of your markdown files."
echo
echo "5. Make sure you have a Qdrant vector database instance running."
echo "   The easiest way is with Docker:"
echo "   docker run -p 6333:6333 qdrant/qdrant"
echo
echo "6. Run the pipeline using the shell script:"
echo "   - To run the full data processing pipeline:"
echo "     ./run_pipeline.sh all"
echo "   - To retrieve information:"
echo "     ./run_pipeline.sh retrieve --query \"What is the history of India?\""
echo
