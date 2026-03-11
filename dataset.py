import os
import json
import torch
import kagglehub
import torchvision.transforms as T
import torchvision.models as models
from PIL import Image
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

# =====================
# DOWNLOAD DATASET
# =====================

path = kagglehub.dataset_download("anhduy091100/vaipe-minimal-dataset")
base_path = os.path.join(path, "vaipe_minimal")

image_dir = os.path.join(base_path, "pills/images")
label_dir = os.path.join(base_path, "pills/labels")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================
# FEATURE EXTRACTOR
# =====================

resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
resnet.fc = torch.nn.Identity()
resnet = resnet.to(device)
resnet.eval()

transform = T.Compose([
    T.Resize((224,224)),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

# =====================
# BUILD GRAPH DATASET
# =====================

def build_graph_dataset():

    graphs = []

    all_labels = []

    for label_file in os.listdir(label_dir):
        with open(os.path.join(label_dir,label_file),'r') as f:
            annotation = json.load(f)

        for box in annotation["boxes"]:
            all_labels.append(box["label"])

    unique_labels = sorted(list(set(all_labels)))
    label_map = {lab:i for i,lab in enumerate(unique_labels)}

    for label_file in os.listdir(label_dir):

        with open(os.path.join(label_dir,label_file),'r') as f:
            annotation = json.load(f)

        image_path = os.path.join(image_dir, annotation["path"])
        image = Image.open(image_path).convert("RGB")

        node_features=[]
        labels=[]

        for box in annotation["boxes"]:

            x,y,w,h = box["x"],box["y"],box["w"],box["h"]

            crop = image.crop((x,y,x+w,y+h))
            crop = transform(crop).unsqueeze(0).to(device)

            with torch.no_grad():
                feat = resnet(crop)

            feat = feat.squeeze().cpu()

            geom = torch.tensor([x,y,w,h],dtype=torch.float)

            node_features.append(torch.cat([feat,geom]))
            labels.append(box["label"])

        if len(node_features)<2:
            continue

        x = torch.stack(node_features)

        edge_index=[]
        for i in range(len(x)):
            for j in range(len(x)):
                if i!=j:
                    edge_index.append([i,j])

        edge_index=torch.tensor(edge_index).t().contiguous()

        y=torch.tensor([label_map[l] for l in labels],dtype=torch.long)

        data=Data(x=x,edge_index=edge_index,y=y)

        graphs.append(data)

    print("Total graphs:",len(graphs))

    return graphs,label_map


# =====================
# TRAIN TEST SPLIT
# =====================

def build_dataloaders(graphs,batch_size=8):

    import random
    random.seed(42)

    random.shuffle(graphs)

    train_size=int(0.8*len(graphs))

    train_graphs=graphs[:train_size]
    test_graphs=graphs[train_size:]

    train_loader=DataLoader(train_graphs,batch_size=batch_size,shuffle=True)
    test_loader=DataLoader(test_graphs,batch_size=batch_size)

    return train_loader,test_loader
