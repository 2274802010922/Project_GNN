import os
import torch
from torch_geometric.data import Data
from torchvision import models, transforms
from PIL import Image
import torchvision
import torch.nn as nn

def build_graphs(dataset_path):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])

    dataset = torchvision.datasets.ImageFolder(dataset_path, transform=transform)

    # ResNet feature extractor
    resnet = models.resnet18(pretrained=True)
    resnet = nn.Sequential(*list(resnet.children())[:-1])
    resnet = resnet.to(device)
    resnet.eval()

    graphs = []

    for img, label in dataset:

        img = img.unsqueeze(0).to(device)

        with torch.no_grad():
            feature = resnet(img)

        feature = feature.view(-1).cpu()

        # node feature
        x = feature.unsqueeze(1)

        # simple chain graph
        edge_index = []
        for i in range(len(x)-1):
            edge_index.append([i,i+1])
            edge_index.append([i+1,i])

        edge_index = torch.tensor(edge_index).t().contiguous()

        y = torch.tensor([label])

        data = Data(x=x, edge_index=edge_index, y=y)

        graphs.append(data)

    return graphs
