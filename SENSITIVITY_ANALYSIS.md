# Data Sensitivity Analysis Report
**Author: Bhavik Kantilal Bhagat**

## Objective
The goal of this analysis is to quantify how sensitive the `all-MiniLM-L6-v2` transformer model is to specific changes in input text. We categorize changes into "Minor" (semantic preservation via synonyms/phrasing) and "Major" (semantic reversal via antonyms).

## Methodology
1. **Baseline**: Generate embeddings for the original `classmates.csv`.
2. **Modification**: Modify 3 specific student descriptions:
    - **Greg Kirczenow**: Minor change (verbing nouns).
    - **Mohammad Pakdoust**: Major change (preference reversal).
    - **Bhavik Kantilal Bhagat**: Minor change (elaboration).
3. **Comparison**: Calculate the cosine similarity between the original embedding and the modified embedding.

## Results

| Name | Change Type | Original Text | Modified Text | Similarity |
| :--- | :--- | :--- | :--- | :--- |
| **Greg Kirczenow** | Minor | "Swim, bike, run" | "swimming, cycling, running" | **0.8734** |
| **Bhavik Bhagat** | Minor | "Chess, Maths and Music." | "I enjoy playing chess, solving math puzzles and listening to music." | **0.7275** |
| **Mohammad Pakdoust** | Major | "...passionate about outdoor activities..." | "...prefer to stay indoors and avoid outdoor activities..." | **0.5617** |

## Key Findings
- **Semantic Sturdiness**: The model maintains high similarity (>0.70) for minor rephrasing, acknowledging that the "topic" remains the same even if the tokens change.
- **Intent Detection**: The significant drop to ~0.56 for Major changes demonstrates that the model is not just looking at "word overlap" but is sensitive to the actual sentiment and intent of the sentence.
- **Control Group**: Unchanged sentences consistently returned a similarity of **1.0000**, verifying the mathematical integrity of the pipeline.

## Conclusion
The `all-MiniLM-L6-v2` model is suitable for finding similar interests because it is robust enough to handle different writing styles (Minor changes) while being sensitive enough to distinguish between contradictory preferences (Major changes).
