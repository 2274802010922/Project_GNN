from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data



def _normalize_features(features):
    features = np.asarray(features, dtype=np.float32)
    norms = np.linalg.norm(features, axis=1, keepdims=True) + 1e-12
    return features / norms



def build_knn_edges(features, k=7, similarity_power=2.0):
    """
    Build a symmetric cosine k-NN graph.

    similarity_power > 1.0 sharpens strong edges and weakens noisy ones.
    """
    features = _normalize_features(features)
    num_nodes = len(features)
    if num_nodes < 2:
        raise ValueError("Need at least 2 samples to build a graph.")

    effective_k = min(k + 1, num_nodes)
    neighbors = NearestNeighbors(n_neighbors=effective_k, metric="cosine")
    neighbors.fit(features)
    distances, indices = neighbors.kneighbors(features)

    edge_pairs = set()
    weight_lookup = {}

    for src in range(num_nodes):
        for rank in range(1, effective_k):
            dst = int(indices[src, rank])
            sim = max(0.0, float(1.0 - distances[src, rank]))
            sim = sim ** similarity_power
            edge_pairs.add((src, dst))
            edge_pairs.add((dst, src))
            weight_lookup[(src, dst)] = sim
            weight_lookup[(dst, src)] = sim

    edge_pairs = sorted(edge_pairs)
    edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(
        [weight_lookup[(int(src), int(dst))] for src, dst in edge_pairs],
        dtype=torch.float32,
    )
    return edge_index, edge_weight



def build_full_graph(features, labels, image_paths, class_names, k=7):
    edge_index, edge_weight = build_knn_edges(features, k=k)

    x = torch.tensor(features, dtype=torch.float32)
    x = F.normalize(x, p=2, dim=1)
    y = torch.tensor(labels, dtype=torch.long)

    data = Data(
        x=x,
        y=y,
        edge_index=edge_index,
        edge_weight=edge_weight,
    )
    data.image_paths = list(image_paths)
    data.class_names = list(class_names)
    data.raw_features = torch.tensor(features, dtype=torch.float32)

    print(f"Full graph nodes: {data.num_nodes}")
    print(f"Full graph edges: {data.num_edges}")
    print(f"Average degree: {data.num_edges / max(data.num_nodes, 1):.2f}")
    return data



def build_episode_graph(features, global_labels, episode_indices, episode_class_ids, k=4):
    """Build a local graph for one few-shot episode."""
    episode_indices = np.asarray(episode_indices, dtype=np.int64)
    episode_features = np.asarray(features)[episode_indices]
    episode_global_y = np.asarray(global_labels)[episode_indices]

    edge_index, edge_weight = build_knn_edges(episode_features, k=k)

    class_to_local = {int(class_id): idx for idx, class_id in enumerate(episode_class_ids)}
    episode_local_y = [class_to_local[int(class_id)] for class_id in episode_global_y]

    x = torch.tensor(episode_features, dtype=torch.float32)
    x = F.normalize(x, p=2, dim=1)

    data = Data(
        x=x,
        y=torch.tensor(episode_local_y, dtype=torch.long),
        global_y=torch.tensor(episode_global_y, dtype=torch.long),
        edge_index=edge_index,
        edge_weight=edge_weight,
    )
    data.episode_indices = torch.tensor(episode_indices, dtype=torch.long)
    data.episode_class_ids = torch.tensor(episode_class_ids, dtype=torch.long)
    return data



def build_class_index(labels):
    class_to_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        class_to_indices[int(label)].append(idx)
    return {cls: np.array(indices, dtype=np.int64) for cls, indices in class_to_indices.items()}
