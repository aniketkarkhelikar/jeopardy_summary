# config.py

import os

# --- 1. File Paths ---
DATA_DIR = os.path.join(os.getcwd(), "data")
RAW_MD_DIR = "/mnt/nfs_share/jeopardy_test" # Shared data source for the cluster

# Paths for the new hierarchical pipeline
SUMMARIES_JSONL = os.path.join(DATA_DIR, "summaries.jsonl")
PARENT_CHUNKS_JSONL = os.path.join(DATA_DIR, "parent_chunks.jsonl")


# --- 2. Model and LLM Configuration ---
# Embedding and chunking models
TOKENIZER_MODEL = "BAAI/bge-large-en-v1.5"
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# --- THIS IS THE CHANGE ---
# On-device LLM for summarization
# List all Ollama hosts you want to use for summarization.
# The workload will be distributed among them.
OLLAMA_HOSTS = [
    "http://172.25.120.214:11434",
    "http://172.25.120.122:11434" # <-- 👈 ADD THE IP OF YOUR SECOND PC HERE
]
SUMMARIZATION_MODEL = "gpt-oss:20b"
# --- END CHANGE ---


# --- 3. Processing and Chunking Parameters ---
MAX_CHUNK_TOKENS = 384 # Increased for better contextual meaning
BATCH_SIZE = 64


# --- 4. Ray Cluster Configuration ---
NUM_WORKERS = 3 # Adjust based on your cluster size
RAY_ADDRESS = "ray://172.25.120.212:10001" # Address of your Ray head node


# --- 5. Qdrant Vector Database Configuration ---
QDRANT_HOST = "172.25.120.212" # IP of your Qdrant instance
QDRANT_PORT = 6333

# Collection names for the hierarchical strategy
SUMMARY_COLLECTION_NAME = "jeopardy_summaries_v2" # Version up for the new model
CHUNK_COLLECTION_NAME = "jeopardy_chunks_v2"


# --- 6. Retrieval Parameters ---
TOP_K = 10          # Final number of documents to return
RECALL_K = 50       # Number of chunks to fetch for the reranker
'''
2.1. Install NFS Client Packages

The clients need a different package to be able to connect to the server.
Bash

sudo apt-get update
sudo apt-get install -y nfs-common

2.2. Create the Mount Point

Create a local directory on the client machine where the shared folder will be mounted. It's good practice to use the same path.
Bash

sudo mkdir -p /mnt/nfs_share

2.3. Mount the Shared Directory

Now, mount the server's shared folder onto the local mount point you just created.
Bash

sudo mount 172.25.120.212:/mnt/nfs_share /mnt/nfs_share

Note: Replace 172.25.120.212 with your server's actual IP if it's different'''