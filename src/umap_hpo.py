"""
UMAP Hyperparameter Optimization Script
Performs Bayesian optimization to find UMAP settings that preserve high-dimensional embedding relationships.
"""

import matplotlib.pyplot as plt
import numpy as np
import optuna
import optuna.visualization as vis
import umap
from sklearn.preprocessing import StandardScaler

import config
from data_loader import DataLoader
from embedding_manager import EmbeddingManager
from utils import calculate_global_preservation_score, load_embeddings_json


def objective(trial: optuna.trial.Trial, embeddings: np.ndarray) -> float:
    """
    Optuna objective function for UMAP hyperparameter optimization.

    Params:
        trial: Optuna trial object
        embeddings: High-dimensional embeddings to reduce

    Returns:
        Preservation score (Spearman correlation)
    """
    n_neighbors = trial.suggest_int("n_neighbors", 2, 50)
    min_dist = trial.suggest_float("min_dist", 0.0, 0.5)
    metric = trial.suggest_categorical("metric", ["cosine", "euclidean", "manhattan"])

    # Mirror existing visualizer pipeline: Standardize then UMAP
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(embeddings)

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        n_components=2,
        random_state=42,
        n_jobs=1,  # For reproducibility and predictability in HPO
    )

    try:
        reduced_data = reducer.fit_transform(scaled_data)

        # Calculate preservation score using centralized utility
        score = calculate_global_preservation_score(embeddings, reduced_data)

    except Exception as e:
        print(f"Trial failed with error: {e}")
        return -1.0  # Return poor score on failure

    return score


def run_hpo() -> None:
    """
    Main HPO loop for UMAP tuning.
    Loads embeddings, runs optimization, and saves results.
    """
    print("=" * 80)
    print("UMAP Hyperparameter Optimization")
    print("=" * 80)

    # Using absolute paths from config
    embeddings_path = config.EMBEDDINGS_JSON

    # Try to load existing embeddings; if not, generate them
    if not embeddings_path.exists():
        print(f"Embeddings file not found at {embeddings_path}. Generating now...")
        loader = DataLoader(config.CLASSMATES_CSV)
        loader.load_data()
        paragraphs = loader.get_paragraphs()
        names = loader.get_names()

        manager = EmbeddingManager(config.DEFAULT_MODEL)
        embeddings_dict = manager.generate_embeddings(paragraphs, names)
        manager.save_embeddings(embeddings_path)
        print(f"✓ Generated and saved embeddings for {config.DEFAULT_MODEL}")
    else:
        print(f"Loading embeddings from {embeddings_path}...")
        embeddings_dict = load_embeddings_json(embeddings_path)

    names = list(embeddings_dict.keys())
    embeddings = np.array(list(embeddings_dict.values()))

    print(f"Loaded {len(names)} embeddings. Starting optimization (100 trials)...")
    print()

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        study_name="umap_tuning",
    )
    study.optimize(lambda trial: objective(trial, embeddings), n_trials=100)

    print("\n" + "=" * 80)
    print("Optimization Complete!")
    print(f"Best parameters: {study.best_params}")
    print(f"Best Spearman correlation: {study.best_value:.4f}")
    print("=" * 80)

    # 1. Export results to CSV
    df = study.trials_dataframe()
    results_path = config.OUTPUT_DIR / "umap_hpo_results.csv"
    df.to_csv(results_path, index=False)
    print(f"✓ Trials exported to {results_path}")

    # 2. Generate Optimization History Plot (using plotly if available, but output to png)
    fig = vis.plot_optimization_history(study)
    history_plot_path = config.OUTPUT_DIR / "optimization_history.png"
    fig.write_image(str(history_plot_path))
    print(f"✓ Optimization history plot saved to {history_plot_path}")

    # 3. Verification & Stability: Run best model with different seeds
    print("\nVerifying stability across different seeds...")
    best_params = study.best_params
    seeds = [42, 123, 999]
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(embeddings)

    for seed in seeds:
        print(f"  - Generating visualization for seed {seed}...")
        reducer = umap.UMAP(**best_params, n_components=2, random_state=seed, n_jobs=1)
        reduced = reducer.fit_transform(scaled_data)

        plt.figure(figsize=config.PLOT_FIGSIZE)
        plt.scatter(reduced[:, 0], reduced[:, 1], alpha=0.6, edgecolors="w", s=100)

        for i, name in enumerate(names):
            plt.annotate(
                name,
                (reduced[i, 0], reduced[i, 1]),
                fontsize=config.PLOT_FONTSIZE,
                xytext=(5, 5),
                textcoords="offset points",
            )

        plt.title(
            f"UMAP Visualization (Seed {seed})\nn_neighbors={best_params['n_neighbors']}, min_dist={best_params['min_dist']:.3f}, metric={best_params['metric']}"
        )
        plt.axis("off")

        vis_path = config.OUTPUT_DIR / f"visualization_seed_{seed}.png"
        plt.savefig(vis_path, dpi=config.PLOT_DPI, bbox_inches="tight")
        plt.close()

    print(f"\n✓ Stability visualizations saved to {config.OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    run_hpo()
