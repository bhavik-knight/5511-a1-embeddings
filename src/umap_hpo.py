"""
UMAP Hyperparameter Optimization Script
Performs Bayesian optimization to find UMAP settings that preserve high-dimensional embedding relationships.
"""
import json
import numpy as np
import optuna
import umap
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import optuna.visualization as vis

def load_embeddings(path: Path):
    """Load embeddings from JSON file."""
    with open(path, 'r') as f:
        data = json.load(f)
    names = list(data.keys())
    embeddings = np.array(list(data.values()))
    return names, embeddings

def calculate_spearman_correlation(high_dim_sim, low_dim_dist):
    """
    Calculate the average Spearman rank correlation across all points.
    Correlates high-D similarity with low-D negative distance.
    """
    correlations = []
    n = high_dim_sim.shape[0]
    for i in range(n):
        # Exclude the point itself
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        
        sims = high_dim_sim[i, mask]
        dists = low_dim_dist[i, mask]
        
        # Rank correlation between sims and -dists (higher similarity should be lower distance)
        corr, _ = spearmanr(sims, -dists)
        if not np.isnan(corr):
            correlations.append(corr)
            
    return np.mean(correlations) if correlations else 0.0

def objective(trial, embeddings):
    """Optuna objective function."""
    n_neighbors = trial.suggest_int('n_neighbors', 2, 50)
    min_dist = trial.suggest_float('min_dist', 0.0, 0.5)
    metric = trial.suggest_categorical('metric', ['cosine', 'euclidean', 'manhattan'])
    
    # Mirror existing visualizer pipeline: Standardize then UMAP
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(embeddings)
    
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        n_components=2,
        random_state=42,
        n_jobs=1 # For reproducibility and predictability in HPO
    )
    
    try:
        reduced_data = reducer.fit_transform(scaled_data)
        
        # High-dimensional cosine similarity
        high_dim_sim = cosine_similarity(embeddings)
        
        # Low_dimensional Euclidean distance
        low_dim_dist = euclidean_distances(reduced_data)
        
        score = calculate_spearman_correlation(high_dim_sim, low_dim_dist)
    except Exception as e:
        print(f"Trial failed with error: {e}")
        return -1.0 # Return poor score on failure
        
    return score

def run_hpo():
    """Main HPO loop."""
    print("Loading data...")
    # Using absolute paths as requested
    base_dir = Path(__file__).parent.parent
    embeddings_path = base_dir / "output" / "embeddings.json"
    names, embeddings = load_embeddings(embeddings_path)
    
    print(f"Loaded {len(names)} embeddings. Starting optimization...")
    
    study = optuna.create_study(
        direction='maximize', 
        sampler=optuna.samplers.TPESampler(seed=42),
        study_name="umap_tuning"
    )
    study.optimize(lambda trial: objective(trial, embeddings), n_trials=100)
    
    print("\nOptimization complete!")
    print(f"Best parameters: {study.best_params}")
    print(f"Best Spearman correlation: {study.best_value:.4f}")
    
    # 1. Export results to CSV
    df = study.trials_dataframe()
    results_path = base_dir / "output" / "umap_hpo_results.csv"
    df.to_csv(results_path, index=False)
    print(f"Trials exported to {results_path}")
    
    # 2. Generate Optimization History Plot
    fig = vis.plot_optimization_history(study)
    history_plot_path = base_dir / "output" / "optimization_history.png"
    fig.write_image(str(history_plot_path))
    print(f"Optimization history plot saved to {history_plot_path}")
    
    # 3. Verification & Stability: Run best model with different seeds
    best_params = study.best_params
    seeds = [42, 123, 999]
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(embeddings)
    
    for seed in seeds:
        print(f"Generating visualization for seed {seed}...")
        reducer = umap.UMAP(
            **best_params,
            n_components=2,
            random_state=seed,
            n_jobs=1
        )
        reduced = reducer.fit_transform(scaled_data)
        
        plt.figure(figsize=(12, 8))
        plt.scatter(reduced[:, 0], reduced[:, 1], alpha=0.6, edgecolors='w', s=100)
        
        for i, name in enumerate(names):
            plt.annotate(
                name, 
                (reduced[i, 0], reduced[i, 1]), 
                fontsize=8, 
                xytext=(5, 5), 
                textcoords='offset points'
            )
            
        plt.title(f"UMAP Visualization (Seed {seed})\nn_neighbors={best_params['n_neighbors']}, min_dist={best_params['min_dist']:.3f}, metric={best_params['metric']}")
        plt.axis("off")
        
        vis_path = base_dir / "output" / f"visualization_seed_{seed}.png"
        plt.savefig(vis_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved visualization to {vis_path}")

if __name__ == "__main__":
    run_hpo()
