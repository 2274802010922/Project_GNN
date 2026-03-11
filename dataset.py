import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split

def build_dataloaders(graphs, batch_size=16):

    train_graphs, test_graphs = train_test_split(
        graphs, test_size=0.2, random_state=42
    )

    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=batch_size)

    return train_loader, test_loader
