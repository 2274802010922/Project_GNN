import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18


class ImagePathDataset(Dataset):
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        return image, image_path


def extract_features(image_paths, batch_size=32, cache_path=None, force_recompute=False):
    """
    Extract ResNet18 penultimate-layer features for all images.
    Returns a NumPy array of shape [N, 512].
    """
    if cache_path is not None and os.path.exists(cache_path) and not force_recompute:
        cached = np.load(cache_path)
        if cached.shape[0] == len(image_paths):
            print(f"Loaded cached features from: {cache_path}")
            print(f"Feature shape: {cached.shape}")
            return cached

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = ResNet18_Weights.DEFAULT
    backbone = resnet18(weights=weights)
    model = torch.nn.Sequential(*list(backbone.children())[:-1]).to(device)
    model.eval()

    transform = weights.transforms()
    dataset = ImagePathDataset(image_paths, transform)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    features = []
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            batch_features = model(images).flatten(start_dim=1)
            features.append(batch_features.cpu())

    features = torch.cat(features, dim=0).numpy().astype(np.float32)

    if cache_path is not None:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.save(cache_path, features)
        print(f"Saved feature cache to: {cache_path}")

    print(f"Extracted features shape: {features.shape}")
    return features
