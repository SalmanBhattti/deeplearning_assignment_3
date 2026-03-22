# train.py
import os
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from main_model import EmbeddingNet, save_model
from dataset import (load_caltech101, get_transforms,
                     ContrastiveDataset, TripletDataset)
from loss import ContrastiveLoss, TripletLoss, batch_hard_mining


def train_one_epoch(model, loader, optimizer, criterion,
                    mode, device):
    model.train()
    total_loss = 0.0

    for batch_idx, batch in enumerate(loader):
        optimizer.zero_grad()

        if mode == 'contrastive':
            img1, img2, labels = batch
            img1   = img1.to(device)
            img2   = img2.to(device)
            labels = labels.float().to(device)

            emb1 = model(img1)
            emb2 = model(img2)
            loss = criterion(emb1, emb2, labels)

        elif mode == 'triplet':
            anchor, positive, negative = batch
            anchor   = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)

            emb_a = model(anchor)
            emb_p = model(positive)
            emb_n = model(negative)
            loss  = criterion(emb_a, emb_p, emb_n)

        elif mode == 'hardmining':
            anchor, positive, negative = batch
            anchor   = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)

            emb_a = model(anchor)
            emb_p = model(positive)
            emb_n = model(negative)

            loss = batch_hard_mining(
                torch.cat([emb_a, emb_p, emb_n], dim=0),
                torch.cat([
                    torch.zeros(emb_a.size(0)),
                    torch.ones(emb_p.size(0)),
                    torch.ones(emb_n.size(0)) * 2
                ]).long().to(device),
                margin=0.2
            )

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if (batch_idx + 1) % 20 == 0:
            print(f"  Batch {batch_idx+1}/{len(loader)} "
                  f"Loss: {loss.item():.4f}")

    return total_loss / len(loader)


def evaluate(model, full_dataset, indices, transform, device, k=1):
    from dataset import BaseDataset
    model.eval()

    base_ds = BaseDataset(full_dataset, indices, transform=transform)
    loader  = DataLoader(base_ds, batch_size=64, shuffle=False,
                         num_workers=0)

    all_embeddings = []
    all_labels     = []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            embs = model(imgs)
            all_embeddings.append(embs.cpu())
            all_labels.append(labels)

    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_labels     = torch.cat(all_labels,     dim=0)

    # Pairwise cosine similarity
    sim_matrix = torch.mm(all_embeddings, all_embeddings.T)

    correct = 0
    n       = all_embeddings.size(0)

    for i in range(n):
        sims       = sim_matrix[i].clone()
        sims[i]    = -1  # exclude self
        top_k      = sims.topk(k).indices
        top_labels = all_labels[top_k]
        if all_labels[i] in top_labels:
            correct += 1

    return correct / n


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.weights_dir, exist_ok=True)

    full_ds, train_idx, val_idx, test_idx = load_caltech101(
        root_dir=args.data_dir
    )

    train_transform = get_transforms(train=True)
    val_transform   = get_transforms(train=False)

    if args.mode == 'contrastive':
        train_dataset = ContrastiveDataset(
            full_ds, train_idx, transform=train_transform
        )
        criterion = ContrastiveLoss(margin=1.0)
        print("\nExperiment 1: Contrastive Loss — Random Pairs")

    elif args.mode == 'triplet':
        train_dataset = TripletDataset(
            full_ds, train_idx, transform=train_transform
        )
        criterion = TripletLoss(margin=0.2)
        print("\nExperiment 2: Triplet Loss — Random Triplets")

    elif args.mode == 'hardmining':
        train_dataset = TripletDataset(
            full_ds, train_idx, transform=train_transform
        )
        criterion = TripletLoss(margin=0.2)
        print("\nExperiment 3: Triplet Loss — Hard Negative Mining")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )

    model     = EmbeddingNet(embedding_dim=128).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=5, gamma=0.5
    )

    best_recall = 0.0

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 40)

        avg_loss = train_one_epoch(
            model, train_loader, optimizer,
            criterion, args.mode, device
        )

        recall1 = evaluate(
            model, full_ds, val_idx, val_transform, device, k=1
        )
        recall5 = evaluate(
            model, full_ds, val_idx, val_transform, device, k=5
        )

        print(f"  Avg Loss : {avg_loss:.4f}")
        print(f"  Recall@1 : {recall1:.4f}")
        print(f"  Recall@5 : {recall5:.4f}")

        scheduler.step()

        if recall1 > best_recall:
            best_recall = recall1
            save_path   = os.path.join(
                args.weights_dir, f"best_{args.mode}.pth"
            )
            save_model(model, save_path)
            print(f"  New best model saved!")

        ckpt_path = os.path.join(
            args.weights_dir, f"ckpt_{args.mode}_epoch{epoch}.pth"
        )
        save_model(model, ckpt_path)

    print(f"\nTraining complete. Best Recall@1: {best_recall:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Embedding Network")

    parser.add_argument('--mode', type=str, default='contrastive',
                        choices=['contrastive', 'triplet', 'hardmining'],
                        help='Training mode')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Path to dataset')
    parser.add_argument('--weights_dir', type=str, default='./weights',
                        help='Where to save model weights')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')

    args = parser.parse_args()
    train(args)