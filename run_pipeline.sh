#!/bin/bash

# ==========================================================
# run_pipeline.sh
# A shell script to control the RAG pipeline using main.py.
#
# NEW, SIMPLIFIED WORKFLOW:
# The 'ingest' command now handles both parsing and chunking.
# The separate 'chunk' command has been removed.
# ==========================================================
# Usage: ./run_pipeline.sh [command] [options]
#
# Commands:
#   ingest          - Scan, parse, enrich, and chunk data.
#   embed           - Generate embeddings and upload to Qdrant.
#   all             - Run the full ingest -> embed pipeline.
#   retrieve        - Search for chunks with a query.
#
# Examples:
#   ./run_pipeline.sh all
#   ./run_pipeline.sh retrieve --query "history of machine learning"
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

# Updated case statement to remove the 'chunk' command
case "$COMMAND" in
    ingest|embed|all|retrieve)
        run_command "$COMMAND" "$@"
        ;;
    *)
        echo "Invalid command: $COMMAND"
        python3 main.py --help
        exit 1
        ;;
esac

echo "Command '$COMMAND' finished."
