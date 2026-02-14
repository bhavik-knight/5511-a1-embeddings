"""
Utility Functions
Shared helper functions used across multiple modules.
"""
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def calculate_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        vec1: First embedding vector
        vec2: Second embedding vector
        
    Returns:
        Cosine similarity score (0 to 1)
        
    Example:
        >>> import numpy as np
        >>> vec1 = np.array([1, 2, 3])
        >>> vec2 = np.array([4, 5, 6])
        >>> similarity = calculate_cosine_similarity(vec1, vec2)
        >>> print(f"Similarity: {similarity:.4f}")
    """
    v1 = vec1.reshape(1, -1)
    v2 = vec2.reshape(1, -1)
    return cosine_similarity(v1, v2)[0][0]
