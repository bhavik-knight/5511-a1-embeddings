# Project Structure

This project follows the standard Python ML project structure:

```
assignment1-embeddings/
├── data/                    # Input data files
│   ├── classmates.csv      # Student information
│   └── .gitkeep
├── output/                  # Generated outputs
│   ├── embeddings.json     # Generated embeddings
│   ├── visualization.png   # UMAP visualization
│   ├── embeddings_example_plot.png
│   └── .gitkeep
├── src/                     # Source code
│   ├── __init__.py         # Makes src a Python package
│   ├── config.py           # Centralized configuration (paths, models, API keys)
│   ├── main.py             # Main embedding generation script
│   └── model_comparison.py # Model comparison script
├── .gitignore
├── LICENSE
├── README.md
└── pyproject.toml
```

## Running the Code

All Python scripts are now in the `src/` directory. To run them:

```bash
# From the project root
cd src
python main.py

# Or for model comparison
cd src
python model_comparison.py
```

## Path Changes

The code has been updated to use centralized configuration via `src/config.py`:
- **Input data**: Configured via `config.CLASSMATES_CSV`
- **Output files**: Configured via `config.EMBEDDINGS_JSON`, `config.VISUALIZATION_PNG`
- **Model settings**: Configured via `config.DEFAULT_MODEL`, `config.ALTERNATIVE_MODEL`

## Configuration

All paths, model settings, and API keys are centralized in `src/config.py`. This file:
- Uses `pathlib.Path` for cross-platform compatibility
- Automatically creates necessary directories
- Loads API keys from environment variables for security
- Provides constants for all configurable parameters

### Setting API Keys (Optional)

If you need to use API keys (e.g., for HuggingFace or OpenAI), set them as environment variables:

```bash
# Linux/Mac
export HUGGINGFACE_API_KEY="your_key_here"
export OPENAI_API_KEY="your_key_here"

# Or create a .env file (recommended)
echo "HUGGINGFACE_API_KEY=your_key_here" >> .env
echo "OPENAI_API_KEY=your_key_here" >> .env
```

The config file will automatically load these from the environment.

This structure follows ML best practices by separating:
- **data/**: Raw and processed data
- **output/**: Generated artifacts (models, plots, results)
- **src/**: Source code and scripts
