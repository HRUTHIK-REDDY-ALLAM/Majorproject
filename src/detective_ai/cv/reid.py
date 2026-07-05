"""Person re-identification using MobileNetV2 (CPU-friendly).

Generates 512-dimensional appearance embeddings for person crops,
used to match individuals across different camera views.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from detective_ai.config import settings
from detective_ai.storage.cache import cache

logger = logging.getLogger(__name__)

# Lazy-loaded model singleton (torch/torchvision imported only on first use,
# so the module stays importable without the optional CV dependencies)
_reid_model = None
_transform = None


def _get_reid_model():
    """Lazy-load MobileNetV2 for ReID feature extraction."""
    global _reid_model, _transform
    if _reid_model is None:
        import torch.nn as nn
        from torchvision import models, transforms

        logger.info("Loading MobileNetV2 ReID model (CPU)...")

        # Use MobileNetV2 pre-trained on ImageNet as feature extractor
        base_model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

        # Remove classifier, add a 512-dim embedding head
        _reid_model = nn.Sequential(
            base_model.features,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(1280, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
        )
        _reid_model.eval()

        # Standard ImageNet normalization
        _transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 128)),  # Standard ReID input size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        logger.info("MobileNetV2 ReID model loaded.")
    return _reid_model


def _get_transform():
    """Get the image preprocessing transform."""
    _get_reid_model()  # Ensure model is loaded (which also initializes _transform)
    return _transform


def extract_embedding(
    person_crop: np.ndarray,
    use_cache: bool = True,
) -> list[float]:
    """Extract a 512-dim appearance embedding from a person crop.

    Args:
        person_crop: BGR image of a detected person (numpy array).
        use_cache: Whether to cache the result.

    Returns:
        512-dimensional embedding as list of floats.
    """
    if person_crop is None or person_crop.size == 0:
        return [0.0] * 512

    import torch
    import torch.nn as nn

    # Cache key based on image hash
    if use_cache:
        img_hash = hash(person_crop.tobytes()[:1000])  # partial hash for speed
        cache_key = f"reid:{img_hash}"
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

    model = _get_reid_model()
    transform = _get_transform()

    # Convert BGR to RGB
    rgb_crop = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)

    # Preprocess
    tensor = transform(rgb_crop).unsqueeze(0)  # Add batch dimension

    # Extract embedding
    with torch.no_grad():
        embedding = model(tensor)
        # L2 normalize
        embedding = nn.functional.normalize(embedding, p=2, dim=1)

    result = embedding.squeeze().tolist()

    if use_cache:
        cache.set(cache_key, result, ttl=3600)

    return result


def extract_embeddings_batch(
    person_crops: list[np.ndarray],
    batch_size: int | None = None,
) -> list[list[float]]:
    """Extract embeddings for a batch of person crops.

    Args:
        person_crops: List of BGR person crop images.
        batch_size: Processing batch size (default from config).

    Returns:
        List of 512-dim embeddings.
    """
    if not person_crops:
        return []

    import torch
    import torch.nn as nn

    batch_size = batch_size or settings.reid_batch_size
    model = _get_reid_model()
    transform = _get_transform()

    all_embeddings = []

    for i in range(0, len(person_crops), batch_size):
        batch_crops = person_crops[i : i + batch_size]
        tensors = []

        for crop in batch_crops:
            if crop is None or crop.size == 0:
                # Create zero tensor for invalid crops
                tensors.append(torch.zeros(3, 256, 128))
            else:
                rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                tensors.append(transform(rgb))

        batch_tensor = torch.stack(tensors)

        with torch.no_grad():
            embeddings = model(batch_tensor)
            embeddings = nn.functional.normalize(embeddings, p=2, dim=1)

        for emb in embeddings:
            all_embeddings.append(emb.tolist())

    logger.debug(f"Extracted {len(all_embeddings)} ReID embeddings (batch_size={batch_size})")
    return all_embeddings


def compute_reid_similarity(
    embedding_a: list[float],
    embedding_b: list[float],
) -> float:
    """Compute cosine similarity between two appearance embeddings.

    Args:
        embedding_a: First 512-dim embedding.
        embedding_b: Second 512-dim embedding.

    Returns:
        Cosine similarity score (0-1).
    """
    a = np.array(embedding_a)
    b = np.array(embedding_b)
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return float(max(0.0, dot / norm))
