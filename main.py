import torch
import os

from dataset import build_graphs
from model import GNNModel
from train import train
from explainer import explain


def main():

    dataset_path = "dataset"   # folder dataset images

    if os.path.exists("graphs.pt"):
        print("Loading graphs...")
        graphs = torch.load("graphs.pt")

    else:
        print("Building graphs...")
        graphs = build_graphs(dataset_path)
        torch.save(graphs, "graphs.pt")
        print("Graphs saved!")

    num_features = graphs[0].num_node_features
    num_classes = len(set([g.y.item() for g in graphs]))

    model = GNNModel(num_features, 64, num_classes)

    train(model, graphs)

    explanation = explain(model, graphs[0])

    print("Explanation generated")


if __name__ == "__main__":
    main()
