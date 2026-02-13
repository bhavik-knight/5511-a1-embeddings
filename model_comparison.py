import json
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity

def load_embeddings(filename):
    with open(filename, "r") as f:
        return json.load(f)

def get_similarity_scores(target_name, embeddings):
    """
    Calculates cosine similarity between target_name and everyone else.
    Returns a dictionary of {name: score}.
    """
    if target_name not in embeddings:
        raise ValueError(f"Name '{target_name}' not found in embeddings!")

    target_vec = np.array(embeddings[target_name]).reshape(1, -1)
    scores = {}

    for name, vec in embeddings.items():
        if name == target_name:
            continue
        
        other_vec = np.array(vec).reshape(1, -1)
        sim = cosine_similarity(target_vec, other_vec)[0][0]
        scores[name] = sim

    return scores

def main():
    # 1. Setup
    me = "Nikola Kriznar"
    file_a = "embeddings_model_a.json" # The original MiniLM model
    file_b = "embeddings_model_b.json" # The new mpnet model

    print(f"--- Sensitivity Analysis for: {me} ---\n")

    # 2. Load Data
    try:
        data_a = load_embeddings(file_a)
        data_b = load_embeddings(file_b)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure you have renamed your embedding files to 'embeddings_model_a.json' and 'embeddings_model_b.json'")
        return

    # 3. Calculate Similarities for both models
    # We filter out ourselves to avoid comparing self-to-self
    classmates = [k for k in data_a.keys() if k != me]
    
    scores_a_map = get_similarity_scores(me, data_a)
    scores_b_map = get_similarity_scores(me, data_b)

    # Extract lists of scores aligned by classmate name for correlation math
    list_a = [scores_a_map[name] for name in classmates]
    list_b = [scores_b_map[name] for name in classmates]

    # 4. Compute Spearman's Rank Correlation
    # This checks if the ORDER of classmates is preserved
    correlation, p_value = spearmanr(list_a, list_b)

    print(f"Spearman's Rank Correlation: {correlation:.4f}")
    print(f"(1.0 = Exact same ranking, 0.0 = Random, -1.0 = Opposite ranking)\n")

    # 5. Qualitative Check: Who are my top 3 matches in each?
    sorted_a = sorted(scores_a_map.items(), key=lambda x: x[1], reverse=True)[:3]
    sorted_b = sorted(scores_b_map.items(), key=lambda x: x[1], reverse=True)[:3]

    print(f"{'Model A Top 3 (MiniLM)':<30} | {'Model B Top 3 (mpnet)'}")
    print("-" * 65)
    for i in range(3):
        name_a, score_a = sorted_a[i]
        name_b, score_b = sorted_b[i]
        print(f"{name_a[:20]:<20} ({score_a:.2f})   | {name_b[:20]:<20} ({score_b:.2f})")

if __name__ == "__main__":
    main()