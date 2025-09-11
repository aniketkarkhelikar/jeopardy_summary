#!/bin/bash

# ==============================================================================
# FOCUSED Retrieval Isolation Debugging Script
# ------------------------------------------------------------------------------
# This script runs a small, targeted list of known problematic queries against
# the dense (Qdrant) and sparse (BM25) retrievers separately.
#
# Usage: ./run_debug_tests.sh
# ==============================================================================

# An array containing only the queries that failed or ranked poorly
declare -a QUERIES_TO_INSPECT=(
    "What was Irving Stone's occupation before he became a famous writer?"
    "What was Jon Bon Jovi's 1997 solo album called?"
    "Which rock star from a major band released a solo project in 1997?"
    "Tell me about a poet inspired by the French Revolution."
)

echo "🚀 Starting FOCUSED Isolated Retrieval Analysis..."

# Loop through each query in the array
for i in "${!QUERIES_TO_INSPECT[@]}"; do
    query_text="${QUERIES_TO_INSPECT[$i]}"

    echo ""
    echo "=============================================================================="
    printf "🔬 ANALYZING QUERY: \"%s\"\n" "$query_text"
    echo "=============================================================================="
    echo ""

    # Run the dense (Qdrant) retrieval test
    python3 debug_dense.py --query "$query_text"
    
    echo "" # Add a spacer

    # Run the sparse (BM25) retrieval test
    python3 debug_sparse.py --query "$query_text"
done

echo ""
echo "✅ Focused analysis complete."
