"""
Utility Functions
Shared helper functions used across multiple modules.
"""
import json
from pathlib import Path
import numpy as np
from sklearn.metrics.pairwise import (
    cosine_similarity, 
    euclidean_distances
)
from scipy.stats import spearmanr


def calculate_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Params:
        vec1: First embedding vector
        vec2: Second embedding vector
        
    Returns:
        Cosine similarity score (0 to 1)
    """
    # Ensure vectors are numpy arrays and reshaped for sklearn
    v1 = np.array(vec1).reshape(1, -1)
    v2 = np.array(vec2).reshape(1, -1)
    return float(cosine_similarity(v1, v2)[0][0])


def load_embeddings_json(filepath: Path) -> dict[str, np.ndarray]:
    """
    Load embeddings from a JSON file and convert to numpy arrays.
    
    Params:
        filepath: Path to the JSON file
        
    Returns:
        Dictionary mapping names to numpy embedding vectors
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    return {name: np.array(vec) for name, vec in data.items()}


def save_embeddings_json(embeddings: dict[str, np.ndarray], filepath: Path) -> None:
    """
    Save embeddings dictionary to a JSON file.
    
    Params:
        embeddings: Dictionary mapping names to embedding vectors
        filepath: Path where the JSON file should be saved
    """
    # Convert numpy arrays to lists for JSON serialization
    serializable_data = {
        name: vec.tolist() if isinstance(vec, np.ndarray) else vec
        for name, vec in embeddings.items()
    }
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(serializable_data, f, indent=2)


def get_top_matches(scores: dict[str, float], top_n: int = 3) -> list[tuple[str, float]]:
    """
    Get the top N matches based on similarity scores.
    
    Params:
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
    
    Params:
        scores_a: First set of similarity scores
        scores_b: Second set of similarity scores
        
    Returns:
        Tuple of (correlation coefficient, p-value)
    """
    # Aligned lists of scores based on sorted keys
    common_names = sorted(set(scores_a.keys()) & set(scores_b.keys()))
    list_a = [scores_a[name] for name in common_names]
    list_b = [scores_b[name] for name in common_names]
    
    memo = spearmanr(list_a, list_b)
    return float(memo.correlation), float(memo.pvalue)


def calculate_global_preservation_score(
    high_dim_data: np.ndarray, 
    low_dim_data: np.ndarray
) -> float:
    """
    Calculate how well the global structure is preserved after dimensionality reduction.
    Uses mean Spearman correlation between high-dim similarities and low-dim distances.
    
    Params:
        high_dim_data: Original high-dimensional embeddings
        low_dim_data: Reduced 2D/3D embeddings
        
    Returns:
        Average Spearman correlation score (higher is better)
    """
    # High-dimensional cosine similarity
    high_dim_sim = cosine_similarity(high_dim_data)
    
    # Low-dimensional Euclidean distance
    low_dim_dist = euclidean_distances(low_dim_data)
    
    correlations = []
    n = high_dim_sim.shape[0]
    
    for i in range(n):
        # Exclude the point itself to avoid 1.0/0.0 artifacts
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        
        sims = high_dim_sim[i, mask]
        dists = low_dim_dist[i, mask]
        
        # Spearman correlation between high-D similarity and low-D negative distance
        # (Lower distance in 2D should correspond to higher similarity in 384D)
        corr, _ = spearmanr(sims, -dists)
        if not np.isnan(corr):
            correlations.append(corr)
            
    return float(np.mean(correlations)) if correlations else 0.0
