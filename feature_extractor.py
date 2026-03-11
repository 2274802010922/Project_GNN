import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os


model = models.resnet18(weights="DEFAULT")
model = torch.nn.Sequential(*list(model.children())[:-1])
model.eval()


transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])


def extract_features(dataset_path):

    features = []

    for root, dirs, files in os.walk(dataset_path):

        for f in files:

            if f.endswith(".jpg"):

                img_path = os.path.join(root, f)

                img = Image.open(img_path).convert("RGB")
                img = transform(img).unsqueeze(0)

                with torch.no_grad():
                    feat = model(img)

                features.append(feat.squeeze())

    print("Total images:", len(features))

    return features
