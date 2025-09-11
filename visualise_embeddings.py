import logging
import pandas as pd
import numpy as np
from qdrant_client import QdrantClient
import umap
import plotly.express as px
import config

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_all_vectors(client, collection_name):
    """Fetches all points (vectors and payloads) from a Qdrant collection."""
    logging.info(f"Fetching all vectors from collection '{collection_name}'...")
    all_points = []
    next_offset = None
    while True:
        points, next_offset = client.scroll(
            collection_name=collection_name,
            limit=256,
            with_payload=True,
            with_vectors=True,
            offset=next_offset
        )
        all_points.extend(points)
        if not next_offset:
            break
    
    logging.info(f"Fetched {len(all_points)} total points.")
    return all_points

def visualize():
    """Main function to fetch, reduce, and plot embeddings."""
    try:
        client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)
        # Verify collection exists
        client.get_collection(collection_name=config.COLLECTION_NAME)
    except Exception as e:
        logging.error(f"Could not connect to Qdrant or find collection. Error: {e}")
        return

    points = fetch_all_vectors(client, config.COLLECTION_NAME)
    
    if not points:
        logging.warning("No points found in the collection to visualize.")
        return

    # Extract vectors and prepare data for DataFrame
    vectors = np.array([point.vector for point in points])
    payloads = [point.payload for point in points]
    
    # Create a DataFrame for easier data handling
    df = pd.DataFrame(payloads)
    # Truncate text for cleaner hover labels
    df['hover_text'] = df['text'].str.slice(0, 150) + '...'

    # Perform dimensionality reduction
    logging.info("Starting UMAP for dimensionality reduction...")
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    embedding_2d = reducer.fit_transform(vectors)
    
    df['x'] = embedding_2d[:, 0]
    df['y'] = embedding_2d[:, 1]
    
    # Create interactive plot
    logging.info("Generating interactive plot...")
    fig = px.scatter(
        df,
        x='x',
        y='y',
        color='source_file',  # Color points by the source file
        hover_data=['hover_text', 'schema', 'chunk_index'],
        title="2D Visualization of Text Chunk Embeddings",
        labels={'color': 'Source File'}
    )

    fig.update_layout(
        xaxis_title="UMAP Dimension 1",
        yaxis_title="UMAP Dimension 2",
        legend_title="Source File"
    )

    # Save the plot as an HTML file
    output_file = "embedding_visualization.html"
    fig.write_html(output_file)
    logging.info(f"✅ Visualization saved to '{output_file}'. Open this file in your browser.")

if __name__ == "__main__":
    visualize()