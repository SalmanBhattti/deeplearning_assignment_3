# inference.py
import os
import sys
import argparse
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from main_model import load_model


def get_inference_transform():
    """Same preprocessing as training — must be identical."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225]
        ),
    ])


def embed_single_image(image_path, model, transform, device):
    """
    Load one image and return its embedding vector.
    """
    img    = Image.open(image_path).convert('RGB')
    tensor = transform(img).unsqueeze(0).to(device)  # (1, 3, 224, 224)

    model.eval()
    with torch.no_grad():
        embedding = model(tensor)

    return embedding.squeeze(0).cpu().numpy()  # (128,)


def embed_multiple_images(image_paths, model, transform, device):
    """
    Load multiple images and return their embedding vectors.
    """
    embeddings = []
    for path in image_paths:
        emb = embed_single_image(path, model, transform, device)
        embeddings.append(emb)
        print(f"  Embedded: {os.path.basename(path)} "
              f"→ shape {emb.shape}")

    return np.stack(embeddings, axis=0)  # (N, 128)


def find_similar(query_path, model, transform,
                 device, saved_emb_path, saved_lbl_path,
                 top_k=5):
    """
    Given a query image path, find top_k most similar
    images from saved embeddings.
    """
    # Load saved embeddings
    db_embeddings = np.load(saved_emb_path)
    db_labels     = np.load(saved_lbl_path)

    # Embed query
    query_emb = embed_single_image(
        query_path, model, transform, device
    )

    # Cosine similarity
    sims    = db_embeddings @ query_emb
    top_idx = np.argsort(sims)[::-1][:top_k]

    print(f"\nTop {top_k} similar images:")
    for rank, idx in enumerate(top_idx):
        print(f"  Rank {rank+1}: "
              f"label={db_labels[idx]}  "
              f"similarity={sims[idx]:.4f}")

    return top_idx, db_labels[top_idx]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run inference on image(s)"
    )
    parser.add_argument(
        '--images', nargs='+', required=True,
        help='Path to one or more images'
    )
    parser.add_argument(
        '--weights', type=str,
        default='./weights/best_contrastive.pth',
        help='Path to trained model weights'
    )
    parser.add_argument(
        '--embedding_dim', type=int, default=128
    )
    args = parser.parse_args()

    device    = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    transform = get_inference_transform()

    print(f"Loading model from {args.weights}...")
    model = load_model(
        args.weights,
        embedding_dim=args.embedding_dim,
        device=device
    )
    model = model.to(device)

    print(f"\nEmbedding {len(args.images)} image(s)...")
    embeddings = embed_multiple_images(
        args.images, model, transform, device
    )

    print(f"\nFinal embedding matrix shape: {embeddings.shape}")
    print(f"First embedding (first 8 values):")
    print(f"  {embeddings[0][:8]}")
    print(f"Embedding norm: {np.linalg.norm(embeddings[0]):.4f}")
    print("\nInference complete!")