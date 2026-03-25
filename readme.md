# Deep Metric Learning for Image Retrieval
## Assignment 3 — Deep Learning Spring 2026

### Requirements
```
pip install -r requirements.txt
```

### Dataset
Dataset is downloaded automatically from torchvision.
Place it in ./data folder or it will download on first run.

### How to Run

#### Step 1 — Train all 3 experiments
```
python train.py --mode contrastive --epochs 5 --batch_size 32 --lr 1e-4
python train.py --mode triplet --epochs 5 --batch_size 32 --lr 1e-4
python train.py --mode hardmining --epochs 5 --batch_size 32 --lr 1e-4
```

#### Step 2 — Save embeddings
```
python save_embeddings.py
```

#### Step 3 — Evaluate and generate retrieval grids
```
python retrieval.py
```

#### Step 4 — Generate t-SNE plots
```
python tsne_plots.py
```

#### Step 5 — Run inference on any image
```
python inference.py --images path/to/image.jpg --weights ./weights/best_contrastive.pth
```

### Results

| Experiment | Loss | Sampling | Recall@1 | Recall@5 |
|---|---|---|---|---|
| Exp-1 | Contrastive | Random Pairs | 82.22% | 92.69% |
| Exp-2 | Triplet | Random Triplets | 81.36% | 92.69% |
| Exp-3 | Triplet | Hard Mining | 43.15% | 59.71% |

### Project Structure
```
project/
├── main_model.py
├── dataset.py
├── loss.py
├── train.py
├── save_embeddings.py
├── retrieval.py
├── inference.py
├── tsne_plots.py
├── weights/
├── embeddings/
├── graphs/
├── requirements.txt
└── README.md
```