# save_embeddings.py
import os
import torch
import numpy as np
from torch.utils.data import DataLoader

from main_model import EmbeddingNet, load_model
from dataset import load_caltech101, get_transforms, BaseDataset


def compute_and_save_embeddings(model, full_dataset, indices,
                                 transform, device, save_path):
    """
    Runs all images through the model and saves embeddings to disk.
    """
    model.eval()

    base_ds = BaseDataset(full_dataset, indices, transform=transform)
    loader  = DataLoader(base_ds, batch_size=64,
                         shuffle=False, num_workers=0)

    all_embeddings = []
    all_labels     = []

    print(f"Computing embeddings for {len(base_ds)} images...")

    with torch.no_grad():
        for i, (imgs, labels) in enumerate(loader):
            imgs = imgs.to(device)
            embs = model(imgs)
            all_embeddings.append(embs.cpu().numpy())
            all_labels.append(labels.numpy())

            if (i + 1) % 10 == 0:
                print(f"  Batch {i+1}/{len(loader)}")

    all_embeddings = np.concatenate(all_embeddings, axis=0)
    all_labels     = np.concatenate(all_labels,     axis=0)

    # Save as numpy arrays
    np.save(save_path + '_embeddings.npy', all_embeddings)
    np.save(save_path + '_labels.npy',     all_labels)

    print(f"Saved embeddings: {all_embeddings.shape}")
    print(f"Saved labels    : {all_labels.shape}")
    print(f"Location        : {save_path}_embeddings.npy")

    return all_embeddings, all_labels


def load_embeddings(save_path):
    """Load saved embeddings from disk."""
    embeddings = np.load(save_path + '_embeddings.npy')
    labels     = np.load(save_path + '_labels.npy')
    print(f"Loaded embeddings: {embeddings.shape}")
    print(f"Loaded labels    : {labels.shape}")
    return embeddings, labels


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create embeddings folder
    os.makedirs('./embeddings', exist_ok=True)

    # Load dataset
    full_ds, train_idx, val_idx, test_idx = load_caltech101(
        root_dir='./data'
    )
    transform = get_transforms(train=False)

    # All 3 experiment modes and their best model paths
    experiments = [
        ('contrastive', './weights/best_contrastive.pth'),
        ('triplet',     './weights/best_triplet.pth'),
        ('hardmining',  './weights/best_hardmining.pth'),
    ]

    for mode, weight_path in experiments:
        print(f"\n{'='*50}")
        print(f"Processing: {mode}")
        print(f"{'='*50}")

        # Load trained model
        model = load_model(weight_path, embedding_dim=128, device=device)
        model = model.to(device)

        # Compute and save for train, val, test splits
        for split_name, indices in [('train', train_idx),
                                     ('val',   val_idx),
                                     ('test',  test_idx)]:
            save_path = f'./embeddings/{mode}_{split_name}'
            compute_and_save_embeddings(
                model, full_ds, indices,
                transform, device, save_path
            )

    print("\nAll embeddings saved successfully!")