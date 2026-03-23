import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class ResidualGCNBlock(torch.nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.3):
        super().__init__()
        self.conv = GCNConv(in_dim, out_dim)
        self.bn = torch.nn.BatchNorm1d(out_dim)
        self.dropout = dropout
        self.res_proj = torch.nn.Linear(in_dim, out_dim) if in_dim != out_dim else torch.nn.Identity()

    def forward(self, x, edge_index, edge_weight=None):
        residual = self.res_proj(x)
        out = self.conv(x, edge_index, edge_weight=edge_weight)
        out = self.bn(out)
        out = F.relu(out + residual)
        out = F.dropout(out, p=self.dropout, training=self.training)
        return out


class FewShotGNN(torch.nn.Module):
    """
    Stronger but still simple graph encoder.

    Compared with the original 2-layer GCN, this version adds:
    - 3 graph layers
    - BatchNorm
    - residual shortcuts
    - a projection head for cleaner metric-learning embeddings
    """

    def __init__(self, input_dim, hidden_dim=384, output_dim=192, dropout=0.35):
        super().__init__()
        self.block1 = ResidualGCNBlock(input_dim, hidden_dim, dropout=dropout)
        self.block2 = ResidualGCNBlock(hidden_dim, hidden_dim, dropout=dropout)
        self.conv_out = GCNConv(hidden_dim, output_dim)
        self.out_bn = torch.nn.BatchNorm1d(output_dim)
        self.skip_proj = torch.nn.Linear(input_dim, output_dim)
        self.dropout = dropout

    def get_embeddings(self, x, edge_index, edge_weight=None):
        h = self.block1(x, edge_index, edge_weight=edge_weight)
        h = self.block2(h, edge_index, edge_weight=edge_weight)

        out = self.conv_out(h, edge_index, edge_weight=edge_weight)
        out = self.out_bn(out)
        out = out + self.skip_proj(x)
        out = F.relu(out)
        out = F.dropout(out, p=self.dropout, training=self.training)
        out = F.normalize(out, p=2, dim=1)
        return out

    def forward(self, x, edge_index, edge_weight=None):
        return self.get_embeddings(x, edge_index, edge_weight=edge_weight)
