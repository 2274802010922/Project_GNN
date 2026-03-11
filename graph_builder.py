import torch
from torch_geometric.data import Data


def build_graph(features):

    x = torch.stack(features)

    num_nodes = x.shape[0]

    edges = []

    for i in range(num_nodes - 1):
        edges.append([i, i + 1])
        edges.append([i + 1, i])

    edge_index = torch.tensor(edges).t().contiguous()

    data = Data(x=x, edge_index=edge_index)

    print("Graph nodes:", num_nodes)
    print("Graph edges:", edge_index.shape[1])

    return data
