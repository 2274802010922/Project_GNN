from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data


def _build_masks(labels, train_ratio=0.7, val_ratio=0.15, seed=42):
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    n = len(labels)

    train_mask = np.zeros(n, dtype=bool)
    val_mask = np.zeros(n, dtype=bool)
    test_mask = np.zeros(n, dtype=bool)

    indices_by_class = defaultdict(list)
    for idx, label in enumerate(labels):
        indices_by_class[int(label)].append(idx)

    for _, class_indices in indices_by_class.items():
        class_indices = np.array(class_indices)
        rng.shuffle(class_indices)
        m = len(class_indices)

        if m == 1:
            train_count, val_count = 1, 0
        elif m == 2:
            train_count, val_count = 1, 0
        elif m == 3:
            train_count, val_count = 1, 1
        else:
            train_count = max(1, int(round(m * train_ratio)))
            val_count = max(1, int(round(m * val_ratio)))
            if train_count + val_count >= m:
                val_count = max(1, m - train_count - 1)

        test_count = m - train_count - val_count
        if test_count < 0:
            test_count = 0

        train_idx = class_indices[:train_count]
        val_idx = class_indices[train_count:train_count + val_count]
        test_idx = class_indices[train_count + val_count:]

        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True

    if val_mask.sum() == 0:
        remaining = np.where(~train_mask)[0]
        if len(remaining) > 0:
            val_mask[remaining[0]] = True
            test_mask[remaining[0]] = False

    if test_mask.sum() == 0:
        remaining = np.where(~train_mask & ~val_mask)[0]
        if len(remaining) > 0:
            test_mask[remaining[0]] = True

    return (
        torch.tensor(train_mask, dtype=torch.bool),
        torch.tensor(val_mask, dtype=torch.bool),
        torch.tensor(test_mask, dtype=torch.bool),
    )


def build_graph(features, labels, image_paths, class_names, k=5, seed=42):
    """
    Build a k-NN graph using cosine distance.
    Each node is an image feature vector.
    """
    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)

    if len(features) < 2:
        raise ValueError("Need at least 2 images to build a graph.")

    features_norm = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-12)

    effective_k = min(k + 1, len(features))
    nbrs = NearestNeighbors(n_neighbors=effective_k, metric="cosine")
    nbrs.fit(features_norm)
    distances, indices = nbrs.kneighbors(features_norm)

    edge_pairs = set()
    for src in range(len(features)):
        for rank in range(1, effective_k):
            dst = int(indices[src, rank])
            edge_pairs.add((src, dst))
            edge_pairs.add((dst, src))

    edge_pairs = sorted(edge_pairs)
    edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()

    weight_lookup = {}
    for src in range(len(features)):
        for rank in range(1, effective_k):
            dst = int(indices[src, rank])
            sim = float(1.0 - distances[src, rank])
            weight_lookup[(src, dst)] = sim
            weight_lookup[(dst, src)] = sim

    aligned_weights = [weight_lookup[(int(src), int(dst))] for src, dst in edge_pairs]

    x = torch.tensor(features, dtype=torch.float32)
    x = F.normalize(x, p=2, dim=1)

    y = torch.tensor(labels, dtype=torch.long)
    edge_weight = torch.tensor(aligned_weights, dtype=torch.float32)

    train_mask, val_mask, test_mask = _build_masks(labels, seed=seed)

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_weight=edge_weight,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )

    data.image_paths = image_paths
    data.class_names = class_names
    data.raw_features = torch.tensor(features, dtype=torch.float32)

    print(f"Nodes: {data.num_nodes}")
    print(f"Edges: {data.num_edges}")
    print(f"Average degree: {data.num_edges / data.num_nodes:.2f}")
    print(
        f"Split sizes -> train: {int(train_mask.sum())}, "
        f"val: {int(val_mask.sum())}, test: {int(test_mask.sum())}"
    )

    return data
