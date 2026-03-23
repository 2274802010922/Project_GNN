import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class FewShotGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=256, output_dim=128, dropout=0.3):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.dropout = dropout

    def get_embeddings(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        x = F.normalize(x, p=2, dim=1)
        return x

    def forward(self, x, edge_index, edge_weight=None):
        return self.get_embeddings(x, edge_index, edge_weight=edge_weight)
