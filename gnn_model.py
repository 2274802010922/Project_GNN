import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GNN(torch.nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim // 2)
        self.classifier = torch.nn.Linear(hidden_dim // 2, num_classes)
        self.dropout = dropout

    def get_embeddings(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        return x

    def forward(self, x, edge_index, edge_weight=None):
        x = self.get_embeddings(x, edge_index, edge_weight=edge_weight)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.classifier(x)
        return x
