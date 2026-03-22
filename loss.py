# loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """
    Contrastive Loss:
        L = y * D^2 + (1 - y) * max(0, margin - D)^2
        y = 1 → same class (positive pair)
        y = 0 → different class (negative pair)
    """
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, emb1, emb2, label):
        # Euclidean distance between embeddings
        D = F.pairwise_distance(emb1, emb2, p=2)

        positive_loss = label * D.pow(2)
        negative_loss = (1 - label) * F.relu(self.margin - D).pow(2)

        loss = (positive_loss + negative_loss).mean()
        return loss


class TripletLoss(nn.Module):
    """
    Triplet Loss:
        L = max(0, D(anchor, positive) - D(anchor, negative) + margin)
    """
    def __init__(self, margin=0.2):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        d_pos = F.pairwise_distance(anchor, positive, p=2)
        d_neg = F.pairwise_distance(anchor, negative, p=2)

        loss = F.relu(d_pos - d_neg + self.margin).mean()
        return loss


def batch_hard_mining(embeddings, labels, margin=0.2):
    """
    Batch Hard Mining:
    For each anchor in the batch:
        - hardest positive: same class, maximum distance to anchor
        - hardest negative: different class, minimum distance to anchor
    Then compute triplet loss on these hard triplets.
    """
    device = embeddings.device
    n      = embeddings.size(0)

    # Compute full pairwise distance matrix (n x n)
    dist_matrix = torch.cdist(embeddings, embeddings, p=2)

    # Boolean masks
    labels     = labels.unsqueeze(1)                        # (n, 1)
    same_class = (labels == labels.T)                       # (n, n)
    diff_class = ~same_class                                # (n, n)

    # Mask out diagonal (don't compare anchor to itself)
    eye_mask   = torch.eye(n, dtype=torch.bool, device=device)
    same_class = same_class & ~eye_mask

    # Hardest positive: same class, maximum distance
    # Replace invalid positions with 0 so they don't get picked as max
    pos_dist   = dist_matrix * same_class.float()
    hardest_pos = pos_dist.max(dim=1)[0]                    # (n,)

    # Hardest negative: different class, minimum distance
    # Replace invalid positions with large value so they don't get picked as min
    neg_dist    = dist_matrix.clone()
    neg_dist[~diff_class] = float('inf')
    hardest_neg = neg_dist.min(dim=1)[0]                    # (n,)

    # Triplet loss on hard pairs
    loss = F.relu(hardest_pos - hardest_neg + margin).mean()
    return loss


if __name__ == "__main__":
    import torch

    batch_size    = 8
    embedding_dim = 128

    emb1   = F.normalize(torch.randn(batch_size, embedding_dim), p=2, dim=1)
    emb2   = F.normalize(torch.randn(batch_size, embedding_dim), p=2, dim=1)
    labels = torch.randint(0, 2, (batch_size,)).float()

    print("--- Contrastive Loss ---")
    c_loss = ContrastiveLoss(margin=1.0)
    loss   = c_loss(emb1, emb2, labels)
    print(f"Loss value: {loss.item():.4f}")

    print("\n--- Triplet Loss ---")
    anchor   = F.normalize(torch.randn(batch_size, embedding_dim), p=2, dim=1)
    positive = F.normalize(torch.randn(batch_size, embedding_dim), p=2, dim=1)
    negative = F.normalize(torch.randn(batch_size, embedding_dim), p=2, dim=1)
    t_loss   = TripletLoss(margin=0.2)
    loss     = t_loss(anchor, positive, negative)
    print(f"Loss value: {loss.item():.4f}")

    print("\n--- Batch Hard Mining ---")
    embeddings = F.normalize(torch.randn(batch_size, embedding_dim), p=2, dim=1)
    labels_int = torch.randint(0, 10, (batch_size,))
    loss       = batch_hard_mining(embeddings, labels_int, margin=0.2)
    print(f"Loss value: {loss.item():.4f}")

    print("\nAll good!")