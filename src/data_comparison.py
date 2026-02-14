"""
Data Comparison Script
Performs sensitivity analysis by comparing embeddings before and after data modifications.
"""

import numpy as np

import config
from data_loader import DataLoader
from embedding_manager import EmbeddingManager
from utils import calculate_cosine_similarity


def main() -> None:
    """Main execution pipeline for data sensitivity analysis."""

    print("=" * 80)
    print("Data Sensitivity Analysis")
    print("=" * 80)

    # Modified names to analyze
    modified_names = ["Greg Kirczenow", "Mohammad Pakdoust", "Bhavik Kantilal Bhagat"]

    # Step 1: Load the current (modified) data
    print("\n[1/4] Loading modified classmates data...")
    loader = DataLoader(config.CLASSMATES_CSV)
    # classmates_map = loader.load_data()
    paragraphs = loader.get_paragraphs()
    names = loader.get_names()
    print(f"✓ Loaded {len(names)} classmate records")

    # Step 2: Generate embeddings with modified data
    print(
        f"\n[2/4] Generating embeddings for modified data using {config.DEFAULT_MODEL}..."
    )
    embedding_manager = EmbeddingManager(config.DEFAULT_MODEL)
    new_embeddings = embedding_manager.generate_embeddings(paragraphs, names)

    # Save new embeddings
    embeddings_modified_path = config.OUTPUT_DIR / "embeddings_modified.json"
    embedding_manager.save_embeddings(embeddings_modified_path)
    print(
        f"✓ Generated and saved {len(new_embeddings)} embeddings to embeddings_modified.json"
    )

    # Step 3: Load original (baseline) embeddings
    print("\n[3/4] Loading original baseline embeddings...")
    embeddings_baseline_path = config.OUTPUT_DIR / "embeddings.json"

    try:
        # instantiate embedding manager with default model
        original_manager = EmbeddingManager(config.DEFAULT_MODEL)

        # load embeddings from embeddings.json file
        original_embeddings = original_manager.load_embeddings(embeddings_baseline_path)
        print(f"✓ Loaded {len(original_embeddings)} baseline embeddings")

    except FileNotFoundError:
        print(f"❌ Error: {embeddings_baseline_path} not found!")
        print("Please ensure you have a baseline embeddings file.")
        print(
            "You can create one by copying embeddings.json to embeddings_baseline.json"
        )
        return

    # Step 4: Compare embeddings for modified entries
    print("\n[4/4] Calculating cosine similarity for all entries...")
    print("\n" + "=" * 80)
    print("MODIFIED SENTENCES: Cosine Similarity Scores")
    print("=" * 80)
    print(f"{'Name':<30} | {'Change Type':<15} | {'Similarity Score'}")
    print("-" * 80)

    # Define change types based on README
    change_types = {
        "Greg Kirczenow": "Minor",
        "Mohammad Pakdoust": "Major",
        "Bhavik Kantilal Bhagat": "Minor",
    }

    results = []

    for name in modified_names:
        if name in original_embeddings and name in new_embeddings:
            original_vec = original_embeddings[name]
            new_vec = new_embeddings[name]

            similarity = calculate_cosine_similarity(original_vec, new_vec)
            change_type = change_types.get(name, "Unknown")

            print(f"{name:<30} | {change_type:<15} | {similarity:.6f}")

            results.append(
                {"name": name, "change_type": change_type, "similarity": similarity}
            )
        else:
            print(f"{name:<30} | {'N/A':<15} | Not found in embeddings")

    print("=" * 80)

    # Step 5: Compare embeddings for UNCHANGED entries (control group)
    print("\n" + "=" * 80)
    print("UNCHANGED SENTENCES (Control Group): Cosine Similarity Scores")
    print("=" * 80)
    print(f"{'Name':<30} | {'Change Type':<15} | {'Similarity Score'}")
    print("-" * 80)

    # Get all names that were NOT modified
    all_names = set(original_embeddings.keys())
    modified_names_set = set(modified_names)
    unchanged_names = sorted(all_names - modified_names_set)

    unchanged_results = []

    # Sample a few unchanged entries to display (show first 5)
    sample_unchanged = unchanged_names[:5]

    for name in sample_unchanged:
        if name in original_embeddings and name in new_embeddings:
            original_vec = original_embeddings[name]
            new_vec = new_embeddings[name]

            similarity = calculate_cosine_similarity(original_vec, new_vec)

            print(f"{name:<30} | {'Unchanged':<15} | {similarity:.6f}")

            unchanged_results.append({"name": name, "similarity": similarity})

    print("=" * 80)

    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if results:
        minor_changes = [r for r in results if r["change_type"] == "Minor"]
        major_changes = [r for r in results if r["change_type"] == "Major"]

        if minor_changes:
            avg_minor = np.mean([r["similarity"] for r in minor_changes])
            print(f"Average similarity for Minor changes:    {avg_minor:.6f}")

        if major_changes:
            avg_major = np.mean([r["similarity"] for r in major_changes])
            print(f"Average similarity for Major changes:    {avg_major:.6f}")

        if unchanged_results:
            avg_unchanged = np.mean([r["similarity"] for r in unchanged_results])
            print(f"Average similarity for Unchanged entries: {avg_unchanged:.6f}")

        print("\nInterpretation:")
        print("  - Unchanged sentences should have similarity ≈ 1.0 (perfect match)")
        print("  - Minor changes (synonyms) should have high scores (0.70-0.90)")
        print("  - Major changes (antonyms) should have lower scores (0.40-0.70)")

    print("\n" + "=" * 80)
    print("✓ Analysis completed successfully!")
    print("=" * 80)
    print("\nOutput files:")
    print(f"  - Original: {embeddings_baseline_path}")
    print(f"  - Modified: {embeddings_modified_path}")
    print()


if __name__ == "__main__":
    main()
