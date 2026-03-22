# dataset.py
import os
import random
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import Caltech101
from collections import defaultdict


def get_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])


def load_caltech101(root_dir="./data", train_ratio=0.70,
                    val_ratio=0.15, seed=42):
    random.seed(seed)

    print("Loading Caltech-101...")
    full_dataset = Caltech101(root=root_dir, download=False)

    class_to_indices = defaultdict(list)
    for idx in range(len(full_dataset)):
        _, label = full_dataset[idx]
        class_to_indices[label].append(idx)

    train_indices, val_indices, test_indices = [], [], []

    for label, indices in class_to_indices.items():
        random.shuffle(indices)
        n       = len(indices)
        n_train = int(n * train_ratio)
        n_val   = int(n * val_ratio)

        train_indices += indices[:n_train]
        val_indices   += indices[n_train:n_train + n_val]
        test_indices  += indices[n_train + n_val:]

    print(f"Classes   : {len(class_to_indices)}")
    print(f"Train     : {len(train_indices)} images")
    print(f"Validation: {len(val_indices)} images")
    print(f"Test      : {len(test_indices)} images")

    return full_dataset, train_indices, val_indices, test_indices


class BaseDataset(Dataset):
    def __init__(self, full_dataset, indices, transform=None):
        self.full_dataset     = full_dataset
        self.indices          = indices
        self.transform        = transform
        self.labels           = []
        self.class_to_indices = defaultdict(list)

        for i, idx in enumerate(indices):
            _, label = full_dataset[idx]
            self.labels.append(label)
            self.class_to_indices[label].append(i)

        self.classes = list(self.class_to_indices.keys())

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        real_idx   = self.indices[i]
        img, label = self.full_dataset[real_idx]
        img        = img.convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

    def _load_image(self, i):
        real_idx   = self.indices[i]
        img, label = self.full_dataset[real_idx]
        img        = img.convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

class ContrastiveDataset(BaseDataset):
    def __getitem__(self, i):
        img1, label1 = self._load_image(i)

        if random.random() < 0.5:
            candidates = [x for x in self.class_to_indices[label1] if x != i]
            if not candidates:
                candidates = self.class_to_indices[label1]
            j          = random.choice(candidates)
            pair_label = 1
        else:
            other_class = random.choice([c for c in self.classes if c != label1])
            j           = random.choice(self.class_to_indices[other_class])
            pair_label  = 0

        img2, _ = self._load_image(j)
        return img1, img2, pair_label


class TripletDataset(BaseDataset):
    def __getitem__(self, i):
        anchor, label = self._load_image(i)

        pos_candidates = [x for x in self.class_to_indices[label] if x != i]
        if not pos_candidates:
            pos_candidates = self.class_to_indices[label]
        pos_i = random.choice(pos_candidates)

        neg_class = random.choice([c for c in self.classes if c != label])
        neg_i     = random.choice(self.class_to_indices[neg_class])

        positive, _ = self._load_image(pos_i)
        negative, _ = self._load_image(neg_i)

        return anchor, positive, negative


if __name__ == "__main__":
    full_ds, train_idx, val_idx, test_idx = load_caltech101(root_dir="./data")

    transform = get_transforms(train=True)

    print("\n--- Testing ContrastiveDataset ---")
    c_ds        = ContrastiveDataset(full_ds, train_idx, transform=transform)
    img1, img2, lbl = c_ds[0]
    print(f"img1: {img1.shape}, img2: {img2.shape}, label: {lbl}")

    print("\n--- Testing TripletDataset ---")
    t_ds            = TripletDataset(full_ds, train_idx, transform=transform)
    anc, pos, neg   = t_ds[0]
    print(f"anchor: {anc.shape}, pos: {pos.shape}, neg: {neg.shape}")

    print("\nAll good!")