import logging
import re
from transformers import AutoTokenizer
import config

# Set up logging for this module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize the tokenizer once when the module is loaded for efficiency
try:
    tokenizer = AutoTokenizer.from_pretrained(config.TOKENIZER_MODEL)
except Exception as e:
    logging.error(f"Could not load tokenizer model '{config.TOKENIZER_MODEL}'. Error: {e}")
    # Fallback to a basic split if tokenizer fails
    tokenizer = None

def chunk_text_intelligently(text):
    """
    Chunks text by paragraphs, and further splits long paragraphs by sentences
    to respect the MAX_CHUNK_TOKENS limit. This function is now a utility
    called by the ingest script.
    """
    # Split by double newlines, which typically separate paragraphs
    paragraphs = re.split(r'\n\s*\n', text)
    # Filter out any empty strings that may result from the split
    chunks = [p.strip() for p in paragraphs if p.strip()]
    
    final_chunks = []
    for chunk in chunks:
        # If no tokenizer is available, just return the paragraphs
        if not tokenizer:
            final_chunks.append(chunk)
            continue

        # Check if the whole paragraph is within the token limit
        if len(tokenizer.encode(chunk)) <= config.MAX_CHUNK_TOKENS:
            final_chunks.append(chunk)
            continue

        # If the paragraph is too long, split it by sentences
        # This regex looks for sentence-ending punctuation followed by a space
        sentences = re.split(r'(?<=[.!?])\s+', chunk)
        
        current_sub_chunk = ""
        for sent in sentences:
            # Check if adding the next sentence would exceed the token limit
            if len(tokenizer.encode(current_sub_chunk + ' ' + sent if current_sub_chunk else sent)) > config.MAX_CHUNK_TOKENS:
                # If the current sub_chunk is not empty, save it
                if current_sub_chunk:
                    final_chunks.append(current_sub_chunk.strip())
                # Start a new sub_chunk with the current sentence
                current_sub_chunk = sent
            else:
                # Add the sentence to the current sub_chunk
                current_sub_chunk = (current_sub_chunk + ' ' + sent) if current_sub_chunk else sent
        
        # Add the last remaining sub_chunk if it exists
        if current_sub_chunk:
            final_chunks.append(current_sub_chunk.strip())
            
    return final_chunks