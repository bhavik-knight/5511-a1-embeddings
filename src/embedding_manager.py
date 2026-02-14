"""
Embedding Manager Module
Handles generation and saving of sentence embeddings.
"""

from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from utils import load_embeddings_json, save_embeddings_json


class EmbeddingManager:
    """Manages embedding generation and persistence."""

    def __init__(self, model_name: str):
        """
        Initialize the EmbeddingManager.

        Params:
            model_name: Name of the sentence transformer model to use
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.embeddings: dict[str, np.ndarray] = {}

    def generate_embeddings(
        self, paragraphs: list[str], names: list[str]
    ) -> dict[str, np.ndarray]:
        """
        Generate embeddings for given paragraphs.

        Params:
            paragraphs: List of text paragraphs to embed
            names: List of names corresponding to each paragraph

        Returns:
            Dictionary mapping names to their embeddings
        """
        # Generate embeddings for all paragraphs
        embeddings_array = self.model.encode(paragraphs)

        # Create name -> embedding mapping
        self.embeddings = {
            name: embedding for name, embedding in zip(names, embeddings_array)
        }

        return self.embeddings

    def save_embeddings(self, output_path: Path) -> None:
        """
        Save embeddings to JSON file.

        Params:
            output_path: Path where JSON file should be saved
        """
        if not self.embeddings:
            raise ValueError("No embeddings to save. Generate embeddings first.")

        save_embeddings_json(self.embeddings, output_path)

    def get_embeddings(self) -> dict[str, np.ndarray]:
        """
        Get the generated embeddings.

        Returns:
            Dictionary mapping names to embeddings
        """
        return self.embeddings

    def load_embeddings(self, input_path: Path) -> dict[str, np.ndarray]:
        """
        Load embeddings from JSON file.

        Params:
            input_path: Path to JSON file containing embeddings

        Returns:
            Dictionary mapping names to embeddings
        """
        self.embeddings = load_embeddings_json(input_path)
        return self.embeddings
