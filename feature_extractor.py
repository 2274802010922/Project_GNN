import os
import torch
from torchvision import models, transforms
from PIL import Image

def extract_features(dataset_path):

    model = models.resnet18(pretrained=True)
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])

    features = []
    image_paths = []   # ← THÊM DÒNG NÀY

    for root, dirs, files in os.walk(dataset_path):

        for file in files:

            if file.endswith(".png") or file.endswith(".jpg"):

                img_path = os.path.join(root, file)

                img = Image.open(img_path).convert("RGB")
                img = transform(img).unsqueeze(0)

                with torch.no_grad():
                    feature = model(img)

                feature = feature.squeeze().numpy()

                features.append(feature)
                image_paths.append(img_path)   # ← THÊM DÒNG NÀY

    print("Total images:", len(features))

    return features, image_paths
