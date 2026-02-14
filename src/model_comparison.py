"""
Model Comparison Module
Compares embeddings from different models using cosine similarity and Spearman correlation.
"""
from pathlib import Path
import numpy as np

from data_loader import DataLoader
from embedding_manager import EmbeddingManager
from utils import (
    calculate_cosine_similarity, 
    get_top_matches, 
    calculate_rank_correlation
)
import config


# ============================================================================
# Similarity Calculation Functions
# ============================================================================

def get_similarity_scores(
    target_name: str, 
    embeddings: dict[str, np.ndarray]
) -> dict[str, float]:
    """
    Calculate cosine similarity between target person and all others.
    
    Params:
        target_name: Name of the target person
        embeddings: Dictionary of name -> embedding mappings
        
    Returns:
        Dictionary mapping names to similarity scores (excluding target)
        
    Raises:
        ValueError: If target_name is not found in embeddings
    """
    if target_name not in embeddings:
        raise ValueError(f"Name '{target_name}' not found in embeddings!")
    
    target_vec = embeddings[target_name]
    scores = {}
    
    for name, vec in embeddings.items():
        if name == target_name:
            continue
        
        similarity = calculate_cosine_similarity(target_vec, vec)
        scores[name] = similarity
    
    return scores


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
    
    Params:
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
    model_a_name: str,
    model_b_name: str
) -> None:
    """
    Print side-by-side comparison of top matches from two models.
    
    Params:
        top_a: Top matches from model A
        top_b: Top matches from model B
        model_a_name: Display name for model A
        model_b_name: Display name for model B
    """
    print(f"{'Top Matches Comparison'}")
    print("-" * 80)
    
    # Shorten names for table display (handle both huggingface and local paths)
    m_a = model_a_name.split('/')[-1]
    m_b = model_b_name.split('/')[-1]
    
    print(f"{m_a:<40} | {m_b}")
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
    model_a_name: str,
    model_b_name: str,
    top_n: int = 3
) -> None:
    """
    Compare embeddings from two different models for a target person.
    
    Params:
        target_name: Name of the person to analyze
        model_a_name: Name of the first model
        model_b_name: Name of the second model
        top_n: Number of top matches to display
    """
    print_header(target_name)
    
    # Load data
    print(f"Loading data from CSV...")
    loader = DataLoader(config.CLASSMATES_CSV)
    loader.load_data()
    paragraphs = loader.get_paragraphs()
    names = loader.get_names()
    
    # Generate embeddings for Model A
    print(f"Generating embeddings for {model_a_name}...")
    manager_a = EmbeddingManager(model_a_name)
    embeddings_a = manager_a.generate_embeddings(paragraphs, names)
    
    # Generate embeddings for Model B
    print(f"Generating embeddings for {model_b_name}...")
    manager_b = EmbeddingManager(model_b_name)
    embeddings_b = manager_b.generate_embeddings(paragraphs, names)
    print()
    
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
    print_top_matches_comparison(top_a, top_b, model_a_name, model_b_name)
    
    print("=" * 80)


def main() -> None:
    """Main entry point for model comparison."""
    # Configuration
    target_name = "Nikola Kriznar"  # Change this to your name
    
    # Run comparison using names from config
    compare_models(
        target_name=target_name,
        model_a_name=config.DEFAULT_MODEL,
        model_b_name=config.ALTERNATIVE_MODEL,
        top_n=5  # Show top 5 matches
    )


if __name__ == "__main__":
    main()