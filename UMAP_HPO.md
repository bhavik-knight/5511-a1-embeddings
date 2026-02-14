# UMAP Hyperparameter Optimization (HPO) Report
**Author: Sridhar Vadla**

## Objective
The 384-dimensional space of sentence embeddings is too complex to visualize directly. We use **UMAP** (Uniform Manifold Approximation and Projection) to reduce this to 2D. However, default UMAP parameters often "squash" local relationships. 

We used **Bayesian Optimization (Optuna)** to find the parameters that best preserve the 384D global structure in a 2D plot.

## Optimization Strategy
- **Library**: `optuna` with `TPESampler`.
- **Trials**: 100.
- **Metric**: Maximize the **Spearman Global Preservation Score**.
    - This score measures the correlation between high-dimensional cosine similarities and low-dimensional (2D) Euclidean distances.

## Best Results (Trial 96)
| Parameter | Best Value |
| :--- | :--- |
| **Metric** | **manhattan** |
| **n_neighbors** | **8** |
| **min_dist** | **0.051** |
| **Best Correlation** | **0.5641** |

### Convergence History
![Optimization History](output/optimization_history.png)

## Detailed Parameters & Impact
1. **n_neighbors (8)**: A low value indicates that the model prioritizes **local structure** (preserving small clusters of very similar classmates) over the global "cloud" shape.
2. **min_dist (0.051)**: This low value allows points to be packed closely together, which helps highlight dense clusters of similar interests.
3. **Metric (manhattan)**: Interestingly, Manhattan distance outperformed Euclidean and Cosine for the 2D projection, suggesting it better captures the grid-like relationship between distinct hobbies in reduced space.

## Stability Analysis
We tested the "Best Parameters" across three different random seeds (42, 123, 999).
- **Observation**: The overall layout remained consistent. Key clusters (e.g., the "Sports Group" and "Tech/Reading Group") appeared together in all seeds.
- **Visual Evidence**: See `output/visualization_seed_*.png`.

## Conclusion
The optimized parameters significantly improved the "readability" of the map. With a preservation score of **0.56**, we can be confident that two people appearing close together in our 2D plot are actually similar in the original high-dimensional embedding space.
