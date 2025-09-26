import numpy as np

from tqdm import tqdm


def batched_faiss_search(faiss_index, test_embeddings, num_neighbors, batch_size=10):
    """
    On the local machine, this function works only with batch sizes of 1. However, on the cluster, we can use
    larger batch sizes.
    """
    all_neighbors = []
    for start in tqdm(range(0, test_embeddings.shape[0], batch_size)):
        end = min(start + batch_size, test_embeddings.shape[0])
        batch = test_embeddings[start:end]
        _, neighbors = faiss_index.search(batch, num_neighbors)
        all_neighbors.append(neighbors)
    return np.vstack(all_neighbors)