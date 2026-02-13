import csv
import json

import umap
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt


def main():
    project_path = ""  # keep empty if classmates.csv is in same folder

    # ---- Load data ----
    attendees_map = {}
    with open(project_path + "classmates.csv", newline="", encoding="cp1252") as csvfile:
        attendees = csv.reader(csvfile, delimiter=",", quotechar='"')
        next(attendees)  # Skip header
        for row in attendees:
            name, paragraph = row
            attendees_map[paragraph] = name

    # ---- Generate sentence embeddings ----
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    paragraphs = list(attendees_map.keys())
    embeddings = model.encode(paragraphs)

    # ---- Create dictionary: person -> embedding ----
    person_embeddings = {
        attendees_map[paragraph]: embedding for paragraph, embedding in zip(paragraphs, embeddings)
    }

    # ---- Save embeddings (required) ----
    # JSON can’t serialize numpy float32 automatically, so convert to list of floats
    embeddings_out = {k: v.tolist() for k, v in person_embeddings.items()}
    with open(project_path + "embeddings.json", "w", encoding="utf-8") as f:
        json.dump(embeddings_out, f)

    # ---- UMAP 2D projection ----
    reducer = umap.UMAP(random_state=42)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(list(person_embeddings.values()))
    reduced_data = reducer.fit_transform(scaled_data)

    # ---- Plot + labels ----
    x = reduced_data[:, 0]
    y = reduced_data[:, 1]
    labels = list(person_embeddings.keys())

    plt.figure(figsize=(10, 8))
    plt.scatter(x, y)
    for i, name in enumerate(labels):
        plt.annotate(name, (x[i], y[i]), fontsize=8)

    plt.axis("off")
    plt.savefig(project_path + "visualization.png", dpi=800)
    plt.close()

    print("Done. Saved embeddings.json and visualization.png")


if __name__ == "__main__":
    main()
