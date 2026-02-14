"""
Configuration file for the embeddings project.
Centralizes all paths, model settings, and API keys.
"""
import os
from pathlib import Path

# Project root directory (parent of src/)
PROJECT_ROOT = Path(__file__).parent.parent

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Output paths
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Input file paths
CLASSMATES_CSV = DATA_DIR / "classmates.csv"

# Output file paths
EMBEDDINGS_JSON = OUTPUT_DIR / "embeddings.json"
VISUALIZATION_PNG = OUTPUT_DIR / "visualization.png"

# Model settings
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
ALTERNATIVE_MODEL = "sentence-transformers/all-mpnet-base-v2"

# Visualization settings
UMAP_RANDOM_STATE = 42
PLOT_FIGSIZE = (12, 8)
PLOT_DPI = 800
PLOT_FONTSIZE = 8

# API Keys (if needed in the future)
# Load from environment variables for security
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", None)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)
