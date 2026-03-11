import torch
import random
import numpy as np

from dataset import build_graph_dataset
from model import GCN
from train import train
from explainer import run_explainer

from torch_geometric.loader import DataLoader


def main():

    # giả sử bạn đã load graphs trước đó
    graphs = torch.load("graphs.pt")

    print("Total graphs:", len(graphs))

    train_loader, test_loader = build_dataloaders(graphs)

    input_dim = graphs[0].num_node_features
    num_classes = len(set([g.y.item() for g in graphs]))

    model = GNN(input_dim, 64, num_classes)

    model = train_model(model, train_loader, test_loader)

    # chạy explainer cho graph đầu
    explanation = run_explainer(model, graphs[0])

    print("Explainer finished")


if __name__ == "__main__":
    main()


