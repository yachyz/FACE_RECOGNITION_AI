import numpy as np
from sklearn.cluster import DBSCAN, KMeans

def cluster_embeddings(embeddings, eps=0.6, min_samples=2, metric="cosine", method="dbscan", n_clusters=5):
    x = np.vstack(embeddings).astype(np.float32)

    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    x = x / norms

    if method == "dbscan":
        model = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
        labels = model.fit_predict(x)
    elif method == "kmeans":
        model = KMeans(n_clusters=n_clusters, random_state=42)
        labels = model.fit_predict(x)
    else:
        raise ValueError(f"Unknown clustering method: {method}")

    return labels
