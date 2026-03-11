import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os


# load pretrained CNN
model = models.resnet18(weights="DEFAULT")

# remove classification layer
model = torch.nn.Sequential(*list(model.children())[:-1])
model.eval()


transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])


def extract_features(dataset_path):

    features = []
    image_paths = []   # FIX: create list

    for root, dirs, files in os.walk(dataset_path):

        for f in files:

            if f.lower().endswith((".jpg",".png",".jpeg")):

                img_path = os.path.join(root, f)

                img = Image.open(img_path).convert("RGB")
                img = transform(img).unsqueeze(0)

                with torch.no_grad():
                    feat = model(img)

                feat = feat.squeeze()

                features.append(feat)
                image_paths.append(img_path)   # FIX: save path

    print("Total images:", len(features))

    return features, image_paths
