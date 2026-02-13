"""
Model Comparison Module
Compares embeddings from different models using cosine similarity and Spearman correlation.
"""
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity

import config


# ============================================================================
# Data Loading Functions
# ============================================================================

def load_embeddings(filepath: Path) -> dict[str, list[float]]:
    """
    Load embeddings from a JSON file.
    
    Args:
        filepath: Path to the JSON file containing embeddings
        
    Returns:
        Dictionary mapping names to embedding vectors
        
    Raises:
        FileNotFoundError: If the file doesn't exist
    """
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================================
# Similarity Calculation Functions
# ============================================================================

def calculate_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        vec1: First embedding vector
        vec2: Second embedding vector
        
    Returns:
        Cosine similarity score (0 to 1)
    """
    v1 = vec1.reshape(1, -1)
    v2 = vec2.reshape(1, -1)
    return cosine_similarity(v1, v2)[0][0]


def get_similarity_scores(
    target_name: str, 
    embeddings: dict[str, list[float]]
) -> dict[str, float]:
    """
    Calculate cosine similarity between target person and all others.
    
    Args:
        target_name: Name of the target person
        embeddings: Dictionary of name -> embedding mappings
        
    Returns:
        Dictionary mapping names to similarity scores (excluding target)
        
    Raises:
        ValueError: If target_name is not found in embeddings
    """
    if target_name not in embeddings:
        raise ValueError(f"Name '{target_name}' not found in embeddings!")
    
    target_vec = np.array(embeddings[target_name])
    scores = {}
    
    for name, vec in embeddings.items():
        if name == target_name:
            continue
        
        other_vec = np.array(vec)
        similarity = calculate_cosine_similarity(target_vec, other_vec)
        scores[name] = similarity
    
    return scores


# ============================================================================
# Ranking and Comparison Functions
# ============================================================================

def get_top_matches(
    scores: dict[str, float], 
    top_n: int = 3
) -> list[tuple[str, float]]:
    """
    Get the top N matches based on similarity scores.
    
    Args:
        scores: Dictionary of name -> score mappings
        top_n: Number of top matches to return
        
    Returns:
        List of (name, score) tuples sorted by score (descending)
    """
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]


def calculate_rank_correlation(
    scores_a: dict[str, float],
    scores_b: dict[str, float]
) -> tuple[float, float]:
    """
    Calculate Spearman's rank correlation between two sets of scores.
    
    Args:
        scores_a: First set of similarity scores
        scores_b: Second set of similarity scores
        
    Returns:
        Tuple of (correlation coefficient, p-value)
        
    Note:
        Assumes both dictionaries have the same keys
    """
    # Get aligned lists of scores
    names = sorted(scores_a.keys())
    list_a = [scores_a[name] for name in names]
    list_b = [scores_b[name] for name in names]
    
    return spearmanr(list_a, list_b)


# ============================================================================
# Display Functions
# ============================================================================

def print_header(target_name: str) -> None:
    """Print the analysis header."""
    print("=" * 80)
    print(f"Model Comparison Analysis for: {target_name}")
    print("=" * 80)
    print()


def print_correlation_results(correlation: float, p_value: float) -> None:
    """
    Print Spearman correlation results with interpretation.
    
    Args:
        correlation: Spearman correlation coefficient
        p_value: Statistical significance p-value
    """
    print(f"Spearman's Rank Correlation: {correlation:.4f}")
    print(f"P-value: {p_value:.6f}")
    print()
    
    # Interpretation
    if correlation > 0.9:
        interpretation = "Very strong positive correlation - rankings are nearly identical"
    elif correlation > 0.7:
        interpretation = "Strong positive correlation - rankings are similar"
    elif correlation > 0.5:
        interpretation = "Moderate positive correlation - some agreement in rankings"
    elif correlation > 0.3:
        interpretation = "Weak positive correlation - limited agreement"
    else:
        interpretation = "Very weak or no correlation - rankings differ significantly"
    
    print(f"Interpretation: {interpretation}")
    print()


def print_top_matches_comparison(
    top_a: list[tuple[str, float]],
    top_b: list[tuple[str, float]],
    model_a_name: str = "Model A (MiniLM)",
    model_b_name: str = "Model B (mpnet)"
) -> None:
    """
    Print side-by-side comparison of top matches from two models.
    
    Args:
        top_a: Top matches from model A
        top_b: Top matches from model B
        model_a_name: Display name for model A
        model_b_name: Display name for model B
    """
    print(f"{'Top Matches Comparison'}")
    print("-" * 80)
    print(f"{model_a_name:<40} | {model_b_name}")
    print("-" * 80)
    
    max_len = max(len(top_a), len(top_b))
    for i in range(max_len):
        if i < len(top_a):
            name_a, score_a = top_a[i]
            left = f"{i+1}. {name_a[:30]:<30} ({score_a:.4f})"
        else:
            left = " " * 40
        
        if i < len(top_b):
            name_b, score_b = top_b[i]
            right = f"{i+1}. {name_b[:30]:<30} ({score_b:.4f})"
        else:
            right = ""
        
        print(f"{left:<40} | {right}")
    
    print()


# ============================================================================
# Main Execution Function
# ============================================================================

def compare_models(
    target_name: str,
    model_a_path: Path,
    model_b_path: Path,
    top_n: int = 3
) -> None:
    """
    Compare embeddings from two different models for a target person.
    
    Args:
        target_name: Name of the person to analyze
        model_a_path: Path to first model's embeddings
        model_b_path: Path to second model's embeddings
        top_n: Number of top matches to display
    """
    print_header(target_name)
    
    # Load embeddings
    print(f"Loading embeddings...")
    try:
        embeddings_a = load_embeddings(model_a_path)
        embeddings_b = load_embeddings(model_b_path)
        print(f"✓ Loaded {len(embeddings_a)} embeddings from Model A")
        print(f"✓ Loaded {len(embeddings_b)} embeddings from Model B")
        print()
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print(f"Make sure both embedding files exist:")
        print(f"  - {model_a_path}")
        print(f"  - {model_b_path}")
        return
    
    # Calculate similarity scores
    print(f"Calculating similarity scores...")
    try:
        scores_a = get_similarity_scores(target_name, embeddings_a)
        scores_b = get_similarity_scores(target_name, embeddings_b)
        print(f"✓ Calculated similarities for {len(scores_a)} classmates")
        print()
    except ValueError as e:
        print(f"❌ Error: {e}")
        return
    
    # Calculate rank correlation
    correlation, p_value = calculate_rank_correlation(scores_a, scores_b)
    print_correlation_results(correlation, p_value)
    
    # Get and display top matches
    top_a = get_top_matches(scores_a, top_n)
    top_b = get_top_matches(scores_b, top_n)
    print_top_matches_comparison(top_a, top_b)
    
    print("=" * 80)


def main():
    """Main entry point for model comparison."""
    # Configuration
    target_name = "Nikola Kriznar"  # Change this to your name
    
    # Run comparison
    compare_models(
        target_name=target_name,
        model_a_path=config.EMBEDDINGS_MODEL_A,
        model_b_path=config.EMBEDDINGS_MODEL_B,
        top_n=5  # Show top 5 matches
    )


if __name__ == "__main__":
    main()