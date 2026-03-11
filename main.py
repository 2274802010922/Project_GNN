import torch
import random
import numpy as np
from torch_geometric.loader import DataLoader

from dataset import build_graph_dataset
from model import GCN
from train import train, test
from explainer import run_explainer

import kagglehub

print("Downloading dataset...")

path = kagglehub.dataset_download("anhduy091100/vaipe-minimal-dataset")

base_path = path + "/vaipe_minimal"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)

seed = 42

np.random.seed(seed)
torch.manual_seed(seed)

# ===== build dataset =====
print("Building graph dataset...")

graphs, label_map, reverse_label_map = build_graph_dataset(base_path, device)

random.shuffle(graphs)

train_size = int(0.8 * len(graphs))

train_graphs = graphs[:train_size]
test_graphs = graphs[train_size:]

train_loader = DataLoader(train_graphs, batch_size=8, shuffle=True)
test_loader = DataLoader(test_graphs, batch_size=8)

print("Train graphs:", len(train_graphs))
print("Test graphs:", len(test_graphs))

# ===== model =====
input_dim = graphs[0].x.shape[1]

model = GCN(input_dim).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

# ===== training =====
best_acc = 0

print("Training model...")

for epoch in range(50):

    loss = train(model, train_loader, optimizer, device)
    acc = test(model, test_loader, device)

    print(f"Epoch {epoch+1}, Loss: {loss:.4f}, Test Acc: {acc:.4f}")

    if acc > best_acc:

        best_acc = acc
        torch.save(model.state_dict(), "best_model.pth")

print("Best Test Accuracy:", best_acc)

# ===== load best model =====
model.load_state_dict(torch.load("best_model.pth"))
model.eval()

# ===== run explainer =====
print("Running GNNExplainer...")

run_explainer(model, test_graphs[0], reverse_label_map, device)