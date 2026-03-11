import torch
from torch_geometric.data import Data
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def build_graph(features):

    features = np.array(features)

    similarity = cosine_similarity(features)

    edge_index = []

    threshold = 0.9

    for i in range(len(features)):
        for j in range(len(features)):

            if i != j and similarity[i][j] > threshold:
                edge_index.append([i, j])

    edge_index = torch.tensor(edge_index).t().contiguous()

    x = torch.tensor(features, dtype=torch.float)

    data = Data(x=x, edge_index=edge_index)

    print("Nodes:", data.num_nodes)
    print("Edges:", data.num_edges)

    return data
