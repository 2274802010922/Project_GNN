import os
from typing import List

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision.models import (
    ResNet18_Weights,
    ResNet50_Weights,
    resnet18,
    resnet50,
)


_MODEL_REGISTRY = {
    "resnet18": (resnet18, ResNet18_Weights.DEFAULT, 512),
    "resnet50": (resnet50, ResNet50_Weights.DEFAULT, 2048),
}


class ImagePathDataset(Dataset):
    def __init__(self, image_paths: List[str], transform, tta: bool = True):
        self.image_paths = image_paths
        self.transform = transform
        self.tta = tta

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")

        if self.tta:
            original = self.transform(image)
            flipped = self.transform(ImageOps.mirror(image))
            views = torch.stack([original, flipped], dim=0)
        else:
            views = self.transform(image).unsqueeze(0)

        return views, image_path


class FeatureBackbone(torch.nn.Module):
    def __init__(self, model_name: str = "resnet50"):
        super().__init__()
        if model_name not in _MODEL_REGISTRY:
            raise ValueError(f"Unsupported model_name={model_name}. Choose from {list(_MODEL_REGISTRY)}")

        model_fn, weights, feature_dim = _MODEL_REGISTRY[model_name]
        backbone = model_fn(weights=weights)
        backbone.fc = torch.nn.Identity()

        self.model = backbone
        self.weights = weights
        self.feature_dim = feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)



def extract_features(
    image_paths,
    batch_size=16,
    cache_path=None,
    force_recompute=False,
    model_name="resnet50",
    tta=True,
):
    """
    Extract pretrained CNN features for all cropped pill images.

    Improvements over the original version:
    - uses a stronger default backbone (ResNet50)
    - averages features from original + horizontally flipped image (light TTA)
    - L2 normalizes features before returning them
    """
    if cache_path is not None and os.path.exists(cache_path) and not force_recompute:
        cached = np.load(cache_path)
        if cached.shape[0] == len(image_paths):
            print(f"Loaded cached features from: {cache_path}")
            print(f"Feature shape: {cached.shape}")
            return cached.astype(np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = FeatureBackbone(model_name=model_name).to(device)
    backbone.eval()

    transform = backbone.weights.transforms()
    dataset = ImagePathDataset(image_paths, transform=transform, tta=tta)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    features = []
    with torch.inference_mode():
        for views, _ in dataloader:
            # views: [B, V, C, H, W]
            bsz, num_views, c, h, w = views.shape
            views = views.view(bsz * num_views, c, h, w).to(device)
            batch_features = backbone(views)
            batch_features = batch_features.view(bsz, num_views, -1).mean(dim=1)
            batch_features = torch.nn.functional.normalize(batch_features, p=2, dim=1)
            features.append(batch_features.cpu())

    features = torch.cat(features, dim=0).numpy().astype(np.float32)

    if cache_path is not None:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.save(cache_path, features)
        print(f"Saved feature cache to: {cache_path}")

    print(f"Backbone: {model_name}")
    print(f"TTA enabled: {tta}")
    print(f"Extracted features shape: {features.shape}")
    return features
