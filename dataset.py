

import kagglehub

# Download latest version
path = kagglehub.dataset_download("anhduy091100/vaipe-minimal-dataset")

print("Path to dataset files:", path)

import os

base_path = path

for item in os.listdir(base_path):
    print(item)

for root, dirs, files in os.walk(base_path):
    print("Folder:", root)
    print("Subfolders:", dirs)
    print("Files:", len(files))
    print("-"*40)

import os

label_path = base_path + "/vaipe_minimal/pills/labels"

print(os.listdir(label_path)[:5])

import json
import os

label_path = base_path + "/vaipe_minimal/pills/labels"

sample_file = os.listdir(label_path)[0]

with open(os.path.join(label_path, sample_file), 'r') as f:
    data = json.load(f)

print(data)

import torch
print(torch.__version__)
try:
    import torch_geometric
    print("PyG installed")
except:
    print("PyG not installed")

Tạo dataset graph

import os
import json
import torch
import torchvision.transforms as T
import torchvision.models as models
from PIL import Image
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

base_path = "/root/.cache/kagglehub/datasets/anhduy091100/vaipe-minimal-dataset/versions/1/vaipe_minimal"
image_dir = os.path.join(base_path, "pills/images")
label_dir = os.path.join(base_path, "pills/labels")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pretrained ResNet18
resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
resnet.fc = torch.nn.Identity()   # remove classifier
resnet = resnet.to(device)
resnet.eval()

transform = T.Compose([
    T.Resize((224,224)),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

graphs = []

# ===== STEP 1: COLLECT ALL LABELS FIRST =====
all_labels = []

for label_file in os.listdir(label_dir):
    with open(os.path.join(label_dir, label_file), 'r') as f:
        annotation = json.load(f)
        for box in annotation["boxes"]:
            all_labels.append(box["label"])

unique_labels = sorted(list(set(all_labels)))
GLOBAL_LABEL_MAP = {lab: i for i, lab in enumerate(unique_labels)}

print("Total classes:", len(GLOBAL_LABEL_MAP))


# ===== STEP 2: CREATE GRAPHS =====
for label_file in os.listdir(label_dir):

    with open(os.path.join(label_dir, label_file), 'r') as f:
        annotation = json.load(f)

    image_path = os.path.join(image_dir, annotation["path"])
    image = Image.open(image_path).convert("RGB")

    node_features = []
    labels = []

    for box in annotation["boxes"]:
        x, y, w, h = box["x"], box["y"], box["w"], box["h"]

        crop = image.crop((x, y, x+w, y+h))
        crop = transform(crop).unsqueeze(0).to(device)

        with torch.no_grad():
            feat = resnet(crop)

        feat = feat.squeeze().cpu()

        geom_feat = torch.tensor([x, y, w, h], dtype=torch.float)

        node_features.append(torch.cat([feat, geom_feat]))
        labels.append(box["label"])

    if len(node_features) < 2:
        continue

    x = torch.stack(node_features)

    # Fully connected edges
    edge_index = []
    for i in range(len(x)):
        for j in range(len(x)):
            if i != j:
                edge_index.append([i, j])

    edge_index = torch.tensor(edge_index).t().contiguous()

    # ✅ USE GLOBAL LABEL MAP HERE
    y = torch.tensor([GLOBAL_LABEL_MAP[l] for l in labels], dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, y=y)
    graphs.append(data)

print("Total graphs:", len(graphs))
print("Example graph:", graphs[0])

!pip install torch-geometric

import kagglehub

# Download latest version
path = kagglehub.dataset_download("anhduy091100/vaipe-minimal-dataset")

print("Path to dataset files:", path)

import os

base_path = path

for item in os.listdir(base_path):
    print(item)

for root, dirs, files in os.walk(base_path):
    print("Folder:", root)
    print("Subfolders:", dirs)
    print("Files:", len(files))
    print("-"*40)

import os

label_path = base_path + "/vaipe_minimal/pills/labels"

print(os.listdir(label_path)[:5])

import json
import os

label_path = base_path + "/vaipe_minimal/pills/labels"

sample_file = os.listdir(label_path)[0]

with open(os.path.join(label_path, sample_file), 'r') as f:
    data = json.load(f)

print(data)

import torch
print(torch.__version__)
try:
    import torch_geometric
    print("PyG installed")
except:
    print("PyG not installed")

#Tạo dataset graph

import os
import json
import torch
import torchvision.transforms as T
import torchvision.models as models
from PIL import Image
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

base_path = "/root/.cache/kagglehub/datasets/anhduy091100/vaipe-minimal-dataset/versions/1/vaipe_minimal"
image_dir = os.path.join(base_path, "pills/images")
label_dir = os.path.join(base_path, "pills/labels")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pretrained ResNet18
resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
resnet.fc = torch.nn.Identity()   # remove classifier
resnet = resnet.to(device)
resnet.eval()

transform = T.Compose([
    T.Resize((224,224)),
    T.ToTensor()
])

graphs = []

# ===== STEP 1: COLLECT ALL LABELS FIRST =====
all_labels = []

for label_file in os.listdir(label_dir):
    with open(os.path.join(label_dir, label_file), 'r') as f:
        annotation = json.load(f)
        for box in annotation["boxes"]:
            all_labels.append(box["label"])

unique_labels = sorted(list(set(all_labels)))
GLOBAL_LABEL_MAP = {lab: i for i, lab in enumerate(unique_labels)}

print("Total classes:", len(GLOBAL_LABEL_MAP))


# ===== STEP 2: CREATE GRAPHS =====
for label_file in os.listdir(label_dir):

    with open(os.path.join(label_dir, label_file), 'r') as f:
        annotation = json.load(f)

    image_path = os.path.join(image_dir, annotation["path"])
    image = Image.open(image_path).convert("RGB")

    node_features = []
    labels = []

    for box in annotation["boxes"]:
        x, y, w, h = box["x"], box["y"], box["w"], box["h"]

        crop = image.crop((x, y, x+w, y+h))
        crop = transform(crop).unsqueeze(0).to(device)

        with torch.no_grad():
            feat = resnet(crop)

        feat = feat.squeeze().cpu()

        geom_feat = torch.tensor([x, y, w, h], dtype=torch.float)

        node_features.append(torch.cat([feat, geom_feat]))
        labels.append(box["label"])

    if len(node_features) < 2:
        continue

    x = torch.stack(node_features)

    # Fully connected edges
    edge_index = []
    for i in range(len(x)):
        for j in range(len(x)):
            if i != j:
                edge_index.append([i, j])

    edge_index = torch.tensor(edge_index).t().contiguous()

    # USE GLOBAL LABEL MAP HERE
    y = torch.tensor([GLOBAL_LABEL_MAP[l] for l in labels], dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, y=y)
    graphs.append(data)

print("Total graphs:", len(graphs))
print("Example graph:", graphs[0])

reverse_label_map = {v: k for k, v in GLOBAL_LABEL_MAP.items()}

node_counts = [g.x.shape[0] for g in graphs]

print("Average nodes:", sum(node_counts)/len(node_counts))
print("Max nodes:", max(node_counts))
print("Min nodes:", min(node_counts))

Train/Test Split

import random
import numpy as np

seed = 42

np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.Random(seed).shuffle(graphs)

train_size = int(0.8 * len(graphs))
train_graphs = graphs[:train_size]
test_graphs = graphs[train_size:]

print("Train graphs:", len(train_graphs))
print("Test graphs:", len(test_graphs))

DataLoader

from torch_geometric.loader import DataLoader

train_loader = DataLoader(train_graphs, batch_size=8, shuffle=True)
test_loader = DataLoader(test_graphs, batch_size=8)

GCN Model

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

# =========================
# GNN MODEL (EMBEDDING MODEL)
# =========================

class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x  # trả về embedding


# =========================
# INIT MODEL
# =========================

input_dim = graphs[0].x.shape[1]   # 516
model = GCN(input_dim).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)


# =========================
# PROTOTYPE LOSS (FEW-SHOT)
# =========================

def prototype_loss(embeddings, labels):

    unique_labels = labels.unique()
    prototypes = []

    for lab in unique_labels:
        mask = labels == lab
        proto = embeddings[mask].mean(dim=0)
        prototypes.append(proto)

    prototypes = torch.stack(prototypes)

    # distance giữa node embedding và prototype
    dists = torch.cdist(embeddings, prototypes)

    # label index mới
    label_map = {lab.item(): i for i, lab in enumerate(unique_labels)}
    new_labels = torch.tensor([label_map[l.item()] for l in labels]).to(device)

    loss = F.cross_entropy(-dists, new_labels)

    return loss


# =========================
# TRAIN FUNCTION
# =========================

def train():

    model.train()
    total_loss = 0

    for data in train_loader:

        data = data.to(device)

        optimizer.zero_grad()

        embeddings = model(data.x, data.edge_index)

        loss = prototype_loss(embeddings, data.y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


# =========================
# TEST FUNCTION
# =========================

def test(loader):

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for data in loader:

            data = data.to(device)

            embeddings = model(data.x, data.edge_index)

            unique_labels = data.y.unique()

            prototypes = []

            for lab in unique_labels:
                mask = data.y == lab
                proto = embeddings[mask].mean(dim=0)
                prototypes.append(proto)

            prototypes = torch.stack(prototypes)

            dists = torch.cdist(embeddings, prototypes)

            preds = dists.argmin(dim=1)

            label_map = {lab.item(): i for i, lab in enumerate(unique_labels)}
            true_labels = torch.tensor([label_map[l.item()] for l in data.y]).to(device)

            correct += (preds == true_labels).sum().item()
            total += data.y.size(0)

    return correct / total


# =========================
# TRAIN LOOP
# =========================
best_acc = 0

for epoch in range(50):

    loss = train()
    acc = test(test_loader)

    print(f"Epoch {epoch+1}, Loss: {loss:.4f}, Test Acc: {acc:.4f}")

    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "best_model.pth")
        print("Best Test Accuracy:", best_acc)

model.load_state_dict(torch.load("best_model.pth"))
model.eval()

GNNExplainer

from torch_geometric.explain import Explainer, GNNExplainer

explainer = Explainer(
    model=model,
    algorithm=GNNExplainer(epochs=200),
    explanation_type='model',
    node_mask_type='attributes',
    edge_mask_type='object',
    model_config=dict(
        mode='multiclass_classification',
        task_level='node',
        return_type='raw',
    ),
)

data = test_graphs[0].to(device)

model.eval()
with torch.no_grad():
  embeddings = model(data.x, data.edge_index)

unique_labels = data.y.unique()

prototypes = []
for lab in unique_labels:
    mask = data.y == lab
    proto = embeddings[mask].mean(dim=0)
    prototypes.append(proto)

prototypes = torch.stack(prototypes)

dists = torch.cdist(embeddings, prototypes)
pred = dists.argmin(dim=1)

print("===== MODEL PREDICTION =====")
print("Node được giải thích:", 0)


pred_class = unique_labels[pred[0]].item()
true_class = data.y[0].item()

print("Predicted:", reverse_label_map[pred_class])
print("Ground truth:", reverse_label_map[true_class])


explanation = explainer(
    x=data.x,
    edge_index=data.edge_index,
    index=0,
)

print("===== KẾT QUẢ GIẢI THÍCH MÔ HÌNH =====")
print("Số node trong graph:", data.x.shape[0])
print("Số cạnh trong graph:", data.edge_index.shape[1])
print("Kích thước ma trận độ quan trọng feature:", explanation.node_mask.shape)
print("Kích thước vector độ quan trọng cạnh:", explanation.edge_mask.shape)

node_mask = explanation.node_mask.detach().cpu()

# Tính độ quan trọng trung bình của từng chiều feature
feature_importance = node_mask.mean(dim=0)

print("\n===== PHÂN TÍCH ĐỘ QUAN TRỌNG FEATURE =====")

topk = torch.topk(feature_importance, 10)

print("10 chiều feature quan trọng nhất:")
for idx, value in zip(topk.indices, topk.values):
    print(f"Feature {idx.item()} - Mức độ quan trọng: {value.item():.4f}")

print("\n===== ĐỘ QUAN TRỌNG THÔNG TIN HÌNH HỌC (Bounding Box) =====")
bbox_importance = feature_importance[-4:]

print(f"Toạ độ x  : {bbox_importance[0].item():.4f}")
print(f"Toạ độ y  : {bbox_importance[1].item():.4f}")
print(f"Chiều rộng: {bbox_importance[2].item():.4f}")
print(f"Chiều cao : {bbox_importance[3].item():.4f}")

print("\n===== PHÂN TÍCH CẠNH ẢNH HƯỞNG =====")

edge_mask = explanation.edge_mask.detach().cpu()

top_edges = torch.topk(edge_mask, min(5, len(edge_mask)))

for idx, value in zip(top_edges.indices, top_edges.values):
    print(f"Cạnh {idx.item()} có mức ảnh hưởng: {value.item():.4f}")

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# Chuyển graph sang CPU
data_cpu = data.cpu()

G = nx.Graph()

# Thêm node
num_nodes = data_cpu.x.shape[0]
for i in range(num_nodes):
    G.add_node(i)

# Thêm edge
edge_index = data_cpu.edge_index
for i in range(edge_index.shape[1]):
    u = edge_index[0, i].item()
    v = edge_index[1, i].item()
    G.add_edge(u, v)

# ===== Node importance =====
node_importance = explanation.node_mask.mean(dim=1).detach().cpu().numpy()

# Chuẩn hóa cho đẹp
node_importance = node_importance / node_importance.max()

# ===== Edge importance =====
edge_importance = explanation.edge_mask.detach().cpu().numpy()
edge_importance = edge_importance / edge_importance.max()

# Layout
pos = nx.spring_layout(G, seed=42)


# Vẽ node
nx.draw_networkx_nodes(
    G,
    pos,
    node_size=3000 * node_importance + 200,
)

# Vẽ edge
threshold = np.percentile(edge_importance, 70)

important_edges = [
    (u, v)
    for (u, v), w in zip(G.edges(), edge_importance)
    if w >= threshold
]

plt.figure(figsize=(8,6))

nx.draw_networkx_nodes(
    G,
    pos,
    node_size=3000 * node_importance + 200,
)

nx.draw_networkx_edges(
    G,
    pos,
    edgelist=important_edges,
    width=3,
)

nx.draw_networkx_labels(G, pos)

plt.title("GNNExplainer - Important Subgraph")
plt.axis("off")

plt.show()

