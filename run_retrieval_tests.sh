#!/bin/bash

# ==============================================================================
# Competition Evaluation Script
# ------------------------------------------------------------------------------
# This script reads queries from a CSV file, runs them through the retrieval
# pipeline, and generates a final JSON file with the results, formatted
# as required for the competition.
#
# Usage: ./run_retrieval_tests.sh
# ==============================================================================

# --- Configuration ---
INPUT_CSV="test_questions.csv"
OUTPUT_JSON="results.json"

# --- Pre-flight Check ---
if ! [ -f "$INPUT_CSV" ]; then
    echo "ERROR: Test data file not found at '$INPUT_CSV'"
    exit 1
fi

# --- Script Execution ---

echo "Starting competition evaluation run..."
echo "Input questions: $INPUT_CSV"
echo "Output will be saved to: $OUTPUT_JSON"

# 1. Initialize the JSON output file with an opening bracket.
echo "[" > "$OUTPUT_JSON"

# 2. Read the CSV file line by line, skipping the header.
#    'tail -n +2' skips the first line.
#    'IFS=,' sets the delimiter for 'read'.
is_first_line=true
tail -n +2 "$INPUT_CSV" | while IFS=, read -r id query; do
    # Trim potential whitespace and quotes from the query
    clean_query=$(echo "$query" | xargs)

    echo "-----------------------------------------------------"
    echo "Running query #$id: \"$clean_query\""

    # 3. For the first entry, don't prepend a comma.
    if [ "$is_first_line" = true ]; then
        is_first_line=false
    else
        # Add a comma before every subsequent entry for valid JSON array format.
        echo "," >> "$OUTPUT_JSON"
    fi

    # 4. Run the retrieval command with the --json-output flag and append to the file.
    # 'stdbuf -o0' ensures output is not buffered and written immediately.
    stdbuf -o0 python3 main.py retrieve --query "$clean_query" --json-output >> "$OUTPUT_JSON"
done

# 5. Close the JSON array with a final bracket.
echo "]" >> "$OUTPUT_JSON"

echo "-----------------------------------------------------"
echo "✅ Evaluation complete."
echo "Final results are available in $OUTPUT_JSON"

