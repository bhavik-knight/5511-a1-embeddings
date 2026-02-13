# Modular Code Structure

The project has been refactored into a modular architecture with separate classes for each task:

## File Structure

```
src/
├── __init__.py              # Package initialization
├── config.py                # Configuration and paths
├── data_loader.py           # DataLoader class
├── embedding_manager.py     # EmbeddingManager class
├── visualizer.py            # Visualizer class
├── main.py                  # Main entry point
└── model_comparison.py      # Model comparison script
```

## Module Descriptions

### 1. **data_loader.py** - `DataLoader` class
**Purpose**: Load and parse CSV data files

**Methods**:
- `__init__(csv_path, encoding)` - Initialize with CSV file path
- `load_data()` - Load data from CSV and return dictionary mapping paragraphs to names
- `get_paragraphs()` - Get list of all paragraphs
- `get_names()` - Get list of all names

**Usage**:
```python
loader = DataLoader(config.CLASSMATES_CSV)
classmates_map = loader.load_data()
paragraphs = loader.get_paragraphs()
names = loader.get_names()
```

---

### 2. **embedding_manager.py** - `EmbeddingManager` class
**Purpose**: Generate and save sentence embeddings

**Methods**:
- `__init__(model_name)` - Initialize with sentence transformer model
- `generate_embeddings(paragraphs, names)` - Generate embeddings for paragraphs
- `save_embeddings(output_path)` - Save embeddings to JSON file
- `get_embeddings()` - Get the generated embeddings dictionary
- `load_embeddings(input_path)` - Load embeddings from JSON file

**Usage**:
```python
manager = EmbeddingManager(config.DEFAULT_MODEL)
embeddings = manager.generate_embeddings(paragraphs, names)
manager.save_embeddings(config.EMBEDDINGS_JSON)
```

---

### 3. **visualizer.py** - `Visualizer` class
**Purpose**: UMAP dimensionality reduction and visualization

**Methods**:
- `__init__(random_state)` - Initialize with random state for reproducibility
- `reduce_dimensions(embeddings)` - Reduce embeddings to 2D using UMAP
- `create_visualization(embeddings, output_path, ...)` - Create and save 2D scatter plot
- `get_reduced_data()` - Get the reduced 2D data

**Usage**:
```python
visualizer = Visualizer(random_state=42)
visualizer.create_visualization(
    embeddings=embeddings,
    output_path=config.VISUALIZATION_PNG,
    figsize=(12, 8),
    dpi=800,
    fontsize=8
)
```

---

### 4. **main.py** - Main Entry Point
**Purpose**: Orchestrate the entire pipeline

**Pipeline Steps**:
1. Load data using `DataLoader`
2. Generate embeddings using `EmbeddingManager`
3. Save embeddings to JSON
4. Create UMAP visualization using `Visualizer`

**Usage**:
```bash
cd src
python3 main.py
```

---

## Benefits of Modular Design

1. **Separation of Concerns**: Each class handles one specific task
2. **Reusability**: Classes can be imported and used in other scripts
3. **Testability**: Each module can be tested independently
4. **Maintainability**: Easier to update and debug individual components
5. **Extensibility**: Easy to add new features or swap implementations

## Migration from Original main.py

The original monolithic `main.py` has been successfully refactored into:
- **Data loading logic** → `data_loader.py`
- **Embedding generation & saving** → `embedding_manager.py`
- **UMAP & visualization** → `visualizer.py`
- **Pipeline orchestration** → `main.py` (new modular version)

The old file has been removed and replaced with the new modular structure.
