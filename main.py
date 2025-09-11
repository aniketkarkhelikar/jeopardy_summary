# main.py

import argparse
import ray
import logging
import sys
import json
from ingest import ingest_and_chunk_data
# --- THIS IS THE FIX ---
from embed_index import embed_and_index_hierarchical # Use the new function name
# --- END FIX ---
from retrieve import retrieve
from bm25_indexer import create_bm25_index
import config

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    # --- Argument Parser Setup ---
    parser = argparse.ArgumentParser(
        description="Control the RAG pipeline for retrieval and indexing.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    subparsers = parser.add_subparsers(dest='command', required=True)

    # Ingest command
    subparsers.add_parser('ingest', help='Run ingestion & chunking of raw data.')

    # Embed command
    subparsers.add_parser('embed', help='Run embedding generation and indexing into Qdrant.')
    
    # BM25 Indexing Command
    subparsers.add_parser('index-bm25', help='Create and save a BM25 index from the chunks.')

    # Retrieve command
    retrieve_parser = subparsers.add_parser('retrieve', help='Run hybrid retrieval with reranking.')
    retrieve_parser.add_argument('--query', required=True, help='The search query to run.')
    retrieve_parser.add_argument(
        '--top-k',
        type=int,
        default=config.TOP_K,
        help=f'The final number of source files to return (default: {config.TOP_K}).'
    )
    retrieve_parser.add_argument(
        '--json-output',
        action='store_true',
        help='Use this flag to output results as a single line of JSON for scripting.'
    )

    # Full pipeline command
    subparsers.add_parser('all', help='Run the full pipeline: ingest -> embed -> index-bm25.')

    # --- Command-line Argument Parsing ---
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
        
    args = parser.parse_args()

    # --- Ray Initialization ---
    is_retrieval = args.command == 'retrieve'
    if not is_retrieval:
        try:
            if config.RAY_ADDRESS:
                logging.info(f"Connecting to Ray cluster at {config.RAY_ADDRESS}...")
                ray.init(address=config.RAY_ADDRESS, ignore_reinit_error=True, runtime_env={"working_dir": "."})
            else:
                logging.info("Initializing Ray locally...")
                ray.init(ignore_reinit_error=True)
            logging.info("Ray initialized successfully.")
        except Exception as e:
            logging.error(f"Could not initialize Ray: {e}", exc_info=True)
            sys.exit(1)


    # --- Main Execution Logic ---
    try:
        if args.command == 'ingest':
            ingest_and_chunk_data()
            
        elif args.command == 'embed':
            # --- THIS IS THE FIX ---
            embed_and_index_hierarchical() # Call the correct function
            # --- END FIX ---

        elif args.command == 'index-bm25':
            create_bm25_index()

        elif is_retrieval:
            relevant_files = retrieve(args.query, top_k=args.top_k)
            
            if args.json_output:
                output_data = { "query": args.query, "relevant_source_files": relevant_files }
                print(json.dumps(output_data))
            else:
                print("\n--- Top Relevant Source Files ---")
                if not relevant_files:
                    print("No relevant source files were found for your query.")
                for i, filename in enumerate(relevant_files):
                    print(f"{i+1}. {filename}")
                print("---------------------------------\n")

        elif args.command == 'all':
            logging.info("--- Starting full pipeline run ---")
            
            logging.info("\n[STEP 1/3] Ingesting and chunking data...")
            ingest_and_chunk_data()
            
            logging.info("\n[STEP 2/3] Generating embeddings and indexing in Qdrant...")
            # --- THIS IS THE FIX ---
            embed_and_index_hierarchical() # Call the correct function
            # --- END FIX ---

            logging.info("\n[STEP 3/3] Creating and saving BM25 index...")
            create_bm25_index()
            
            logging.info("\n✅ Full pipeline (ingest, embed, index-bm25) completed successfully!")

    except Exception as e:
        logging.error(f"An error occurred during execution: {e}", exc_info=True)
        
    finally:
        if not is_retrieval and ray.is_initialized():
            ray.shutdown()
            logging.info("Ray has been shut down.")

if __name__ == "__main__":
    main()