# Model Comparison Refactoring

## Overview

The `model_comparison.py` file has been refactored into **well-organized helper functions** rather than a class-based approach. This decision was made because:

1. **Linear workflow**: The script follows a straightforward sequence of operations
2. **No state management**: No need to maintain state between operations
3. **Functional composition**: Each function performs a single, clear task
4. **Reusability**: Functions can be imported and used independently

## Function Organization

The code is organized into **5 functional groups**:

### 1. 📥 Data Loading Functions
- `load_embeddings(filepath)` - Load embeddings from JSON file

### 2. 📊 Similarity Calculation Functions
- `calculate_cosine_similarity(vec1, vec2)` - Calculate cosine similarity between two vectors
- `get_similarity_scores(target_name, embeddings)` - Calculate similarities for all classmates

### 3. 🏆 Ranking and Comparison Functions
- `get_top_matches(scores, top_n)` - Get top N matches by score
- `calculate_rank_correlation(scores_a, scores_b)` - Calculate Spearman's rank correlation

### 4. 🖥️ Display Functions
- `print_header(target_name)` - Print analysis header
- `print_correlation_results(correlation, p_value)` - Print correlation with interpretation
- `print_top_matches_comparison(top_a, top_b, ...)` - Print side-by-side comparison

### 5. 🚀 Main Execution Function
- `compare_models(target_name, model_a_path, model_b_path, top_n)` - Orchestrate the entire comparison
- `main()` - Entry point with configuration

## Key Improvements

### ✨ Better Organization
- Functions are grouped by purpose with clear section headers
- Each function has a single, well-defined responsibility

### 📝 Comprehensive Documentation
- Every function has detailed docstrings
- Type hints for all parameters and return values
- Clear descriptions of arguments and return values

### 🛡️ Enhanced Error Handling
- Better error messages with helpful suggestions
- Graceful handling of missing files or names
- Clear feedback on what went wrong

### 📊 Improved Output
- Better formatted output with visual separators
- Correlation interpretation (not just numbers)
- More informative progress messages
- Configurable number of top matches to display

### 🔧 Increased Flexibility
- `compare_models()` function can be imported and used programmatically
- Easy to change target name or number of matches
- Modular functions can be reused in other scripts

## Usage Examples

### Basic Usage (as a script)
```bash
cd src
python3 model_comparison.py
```

### Programmatic Usage (import as module)
```python
from model_comparison import compare_models, get_similarity_scores, load_embeddings
import config

# Compare models for a specific person
compare_models(
    target_name="Your Name",
    model_a_path=config.EMBEDDINGS_MODEL_A,
    model_b_path=config.EMBEDDINGS_MODEL_B,
    top_n=10  # Show top 10 matches
)

# Or use individual functions
embeddings = load_embeddings(config.EMBEDDINGS_JSON)
scores = get_similarity_scores("Your Name", embeddings)
top_matches = get_top_matches(scores, top_n=5)
```

### Custom Analysis
```python
from model_comparison import (
    load_embeddings,
    get_similarity_scores,
    calculate_rank_correlation
)

# Load your embeddings
emb_a = load_embeddings("path/to/model_a.json")
emb_b = load_embeddings("path/to/model_b.json")

# Compare multiple people
for person in ["Alice", "Bob", "Charlie"]:
    scores_a = get_similarity_scores(person, emb_a)
    scores_b = get_similarity_scores(person, emb_b)
    corr, p_val = calculate_rank_correlation(scores_a, scores_b)
    print(f"{person}: correlation = {corr:.4f}")
```

## Function vs Class Decision

**Why helper functions instead of a class?**

✅ **Advantages of helper functions for this use case:**
- Simpler and more straightforward
- No unnecessary state management
- Easy to test individual functions
- Natural for a linear workflow
- Functions can be imported independently

❌ **When a class would be better:**
- If we needed to maintain state (e.g., cached embeddings)
- If we had multiple related operations on the same data
- If we needed inheritance or polymorphism
- If we wanted to encapsulate configuration

For this model comparison script, **helper functions are the right choice** because the workflow is linear and stateless.
