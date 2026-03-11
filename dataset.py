import os
import json
import torch
import torchvision.transforms as T
import torchvision.models as models
from PIL import Image
from torch_geometric.data import Data
import kagglehub


def load_dataset():

    path = kagglehub.dataset_download("anhduy091100/vaipe-minimal-dataset")

    base_path = path + "/vaipe_minimal"

    image_dir = os.path.join(base_path, "pills/images")
    label_dir = os.path.join(base_path, "pills/labels")

    return image_dir, label_dir


def build_graphs(device):

    image_dir, label_dir = load_dataset()

    resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    resnet.fc = torch.nn.Identity()
    resnet = resnet.to(device)
    resnet.eval()

    transform = T.Compose([
        T.Resize((224,224)),
        T.ToTensor()
    ])

    graphs = []

    all_labels = []

    for label_file in os.listdir(label_dir):

        with open(os.path.join(label_dir, label_file),'r') as f:
            annotation = json.load(f)

        for box in annotation["boxes"]:
            all_labels.append(box["label"])

    unique_labels = sorted(list(set(all_labels)))
    GLOBAL_LABEL_MAP = {lab:i for i,lab in enumerate(unique_labels)}

    for label_file in os.listdir(label_dir):

        with open(os.path.join(label_dir,label_file),'r') as f:
            annotation = json.load(f)

        image_path = os.path.join(image_dir,annotation["path"])
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

            geom_feat = torch.tensor([x,y,w,h],dtype=torch.float)

            node_features.append(torch.cat([feat,geom_feat]))
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

        y=torch.tensor([GLOBAL_LABEL_MAP[l] for l in labels])

        data = Data(x=x,edge_index=edge_index,y=y)

        graphs.append(data)

    return graphs, GLOBAL_LABEL_MAP
