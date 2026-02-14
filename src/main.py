"""
Main Entry Point
Orchestrates the embedding generation pipeline using modular components.
"""

import config
from data_loader import DataLoader
from embedding_manager import EmbeddingManager
from visualizer import Visualizer


def main() -> None:
    """Main execution pipeline for embedding generation and visualization."""

    print("=" * 60)
    print("Sentence Embeddings Generation Pipeline")
    print("=" * 60)

    # Step 1: Load data
    print("\n[1/4] Loading data from CSV...")
    loader = DataLoader(config.CLASSMATES_CSV)
    classmates_map = loader.load_data()
    paragraphs = loader.get_paragraphs()
    names = loader.get_names()
    print(f"✓ Loaded {len(names)} classmate records")

    # Step 2: Generate embeddings
    print(f"\n[2/4] Generating embeddings using {config.DEFAULT_MODEL}...")
    embedding_manager = EmbeddingManager(config.DEFAULT_MODEL)
    embeddings = embedding_manager.generate_embeddings(paragraphs, names)
    print(f"✓ Generated {len(embeddings)} embeddings")

    # Step 3: Save embeddings
    print(f"\n[3/4] Saving embeddings to {config.EMBEDDINGS_JSON}...")
    embedding_manager.save_embeddings(config.EMBEDDINGS_JSON)
    print("✓ Saved embeddings to JSON")

    # Step 4: Create visualization
    print("\n[4/4] Creating UMAP visualization...")
    visualizer = Visualizer(random_state=config.UMAP_RANDOM_STATE)
    visualizer.create_visualization(
        embeddings=embeddings,
        output_path=config.VISUALIZATION_PNG,
        figsize=config.PLOT_FIGSIZE,
        dpi=config.PLOT_DPI,
        fontsize=config.PLOT_FONTSIZE,
    )
    print(f"✓ Saved visualization to {config.VISUALIZATION_PNG}")

    print("\n" + "=" * 60)
    print("✓ Pipeline completed successfully!")
    print("=" * 60)
    print("\nOutputs:")
    print(f"  - Embeddings: {config.EMBEDDINGS_JSON}")
    print(f"  - Visualization: {config.VISUALIZATION_PNG}")
    print()


if __name__ == "__main__":
    main()
