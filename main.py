import torch
from torch_geometric.loader import DataLoader
import random
import numpy as np

from dataset import build_graphs
from model import GCN
from train import train
from explain import create_explainer
from visualize import visualize_graph


def main():

    seed=42

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    graphs,label_map = build_graphs(device)

    random.shuffle(graphs)

    train_size=int(0.8*len(graphs))

    train_graphs=graphs[:train_size]
    test_graphs=graphs[train_size:]

    train_loader = DataLoader(train_graphs,batch_size=8,shuffle=True)
    test_loader = DataLoader(test_graphs,batch_size=8)

    input_dim = graphs[0].x.shape[1]

    model = GCN(input_dim).to(device)

    optimizer = torch.optim.Adam(model.parameters(),lr=0.0005)

    for epoch in range(50):

        loss = train(model,train_loader,optimizer,device)

        print("Epoch",epoch+1,"Loss",loss)

    explainer = create_explainer(model)

    data=test_graphs[0].to(device)

    explanation = explainer(

        x=data.x,
        edge_index=data.edge_index,
        index=0

    )

    visualize_graph(data.cpu(),explanation)


if __name__=="__main__":
    main()
