# Embedding Model Comparison Report
**Author: Nikola Kriznar**

## Objective
This experiment compares the ranking performance of two different transformer models:
1. **Model A (Default)**: `sentence-transformers/all-MiniLM-L6-v2` (384-dim, Lightweight)
2. **Model B (Alternative)**: `sentence-transformers/all-mpnet-base-v2` (768-dim, Heavyweight)

We want to observe if a significantly larger model changes the similarity "rankings" for a specific target persona.

## Analysis for: Nikola Kriznar

### Statistical Correlation
- **Spearman's Rank Correlation**: **0.3873**
- **P-value**: 0.1246

A Spearman score of ~0.39 indicates a **weak to moderate positive correlation**. This means that while both models agree on the most obvious matches, their interpretation of "middle-tier" similarity differs significantly.

### Top Matches Comparison

| Rank | all-MiniLM-L6-v2 (Smaller) | Score | all-mpnet-base-v2 (Larger) | Score |
| :--- | :--- | :--- | :--- | :--- |
| **1** | Zilong Wang | 0.7545 | Zilong Wang | 0.8343 |
| **2** | Binziya Siddik | 0.7036 | Somto Muotoe | 0.7272 |
| **3** | Bhavik Kantilal Bhagat | 0.6490 | Mohammad Pakdoust | 0.6974 |
| **4** | Sridhar Vadla | 0.6353 | Pawan Lingras | 0.6306 |
| **5** | Md Riad Arifin | 0.6173 | Md Musfiqur Rahman | 0.6261 |

## Qualitative Insights
- **Model Agreement**: Both models identified **Zilong Wang** as the #1 match for Nikola. This suggests very high semantic overlap in their descriptions (specifically keywords like "competitive gaming" and "traveling").
- **Embedding Density**: The `mpnet` (larger) model tends to assign higher raw similarity scores (e.g., 0.83 vs 0.75 for the top match). This indicates a more "confident" clustering in the higher-dimensional space.
- **Semantic Nuance**: `mpnet` ranked **Somto Muotoe** significantly higher (#2 vs not in Top 5 for MiniLM). Somto's description focuses on technology and specific hobbies, which the larger model likely linked more effectively to Nikola's specific mentions of gaming.

## Conclusion
For simple keyword-based similarity, `MiniLM` is highly efficient. However, for identifying nuanced relationships between diverse interests (e.g., relating "gaming" to specific "tech interests"), the `mpnet` model provides a more sophisticated ranking, albeit at a higher computational cost.
