"""
Visualizer Module
Handles UMAP dimensionality reduction and visualization.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import umap
from sklearn.preprocessing import StandardScaler


class Visualizer:
    """Handles UMAP reduction and visualization of embeddings."""
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the Visualizer.
        
        Args:
            random_state: Random state for reproducibility
        """
        self.random_state = random_state
        self.reducer = umap.UMAP(random_state=random_state)
        self.scaler = StandardScaler()
        self.reduced_data = None
    
    def reduce_dimensions(self, embeddings: dict[str, np.ndarray]) -> np.ndarray:
        """
        Reduce embeddings to 2D using UMAP.
        
        Args:
            embeddings: Dictionary mapping names to embeddings
            
        Returns:
            2D numpy array of reduced embeddings
        """
        # Convert embeddings dict to array
        embeddings_array = np.array(list(embeddings.values()))
        
        # Scale and reduce
        scaled_data = self.scaler.fit_transform(embeddings_array)
        self.reduced_data = self.reducer.fit_transform(scaled_data)
        
        return self.reduced_data
    
    def create_visualization(
        self, 
        embeddings: dict[str, np.ndarray],
        output_path: Path,
        figsize: tuple = (12, 8),
        dpi: int = 800,
        fontsize: int = 8
    ) -> None:
        """
        Create and save a 2D visualization of embeddings.
        
        Args:
            embeddings: Dictionary mapping names to embeddings
            output_path: Path where visualization should be saved
            figsize: Figure size (width, height)
            dpi: Resolution in dots per inch
            fontsize: Font size for labels
        """
        # Reduce dimensions if not already done
        if self.reduced_data is None:
            self.reduce_dimensions(embeddings)
        
        # Extract coordinates and labels
        x = self.reduced_data[:, 0]
        y = self.reduced_data[:, 1]
        labels = list(embeddings.keys())
        
        # Create plot
        plt.figure(figsize=figsize)
        plt.scatter(x, y)
        
        # Add labels for each point
        for i, name in enumerate(labels):
            plt.annotate(name, (x[i], y[i]), fontsize=fontsize)
        
        # Remove axes for cleaner look
        plt.axis("off")
        
        # Save and close
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close()
    
    def get_reduced_data(self) -> np.ndarray:
        """
        Get the reduced 2D data.
        
        Returns:
            2D numpy array of reduced embeddings
        """
        if self.reduced_data is None:
            raise ValueError("No reduced data available. Run reduce_dimensions first.")
        return self.reduced_data
