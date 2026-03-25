# retrieval.py
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from torchvision.datasets import Caltech101


# ── Embedding I/O ─────────────────────────────────────────────────────────────

def fetch_embeddings(base_path):
    """Read precomputed embeddings and their class labels from disk."""
    feat_matrix = np.load(base_path + '_embeddings.npy')
    class_ids   = np.load(base_path + '_labels.npy')
    print(f"  Fetched {feat_matrix.shape[0]} vectors "
          f"of dim {feat_matrix.shape[1]}")
    return feat_matrix, class_ids


# ── Retrieval Metric ──────────────────────────────────────────────────────────

def recall_at_k(feat_matrix, class_ids, k=1):
    """
    Nearest-neighbour Recall@K.
    For every query we retrieve K candidates (excluding itself)
    and check whether the ground-truth class is among them.
    """
    # Cosine similarity matrix  (N x N)
    affinity = cosine_similarity(feat_matrix)
    np.fill_diagonal(affinity, -np.inf)   # block self-retrieval

    hits  = 0
    total = len(class_ids)

    for q in range(total):
        ranked      = np.argsort(affinity[q])[::-1][:k]
        if class_ids[q] in class_ids[ranked]:
            hits += 1

    score = hits / total
    return score


# ── Retrieval Grid ────────────────────────────────────────────────────────────

def draw_retrieval_grid(q_idx, feat_matrix, class_ids,
                        raw_dataset, split_indices,
                        top_k=5, out_file=None):
    """
    Renders a 1 × (top_k+1) grid:
      col-0        → query image  (blue border)
      col-1..top_k → ranked neighbours
                     green border = same class  /  red = different class
    """
    affinity = cosine_similarity(feat_matrix)
    np.fill_diagonal(affinity, -np.inf)

    q_class   = class_ids[q_idx]
    nb_idx    = np.argsort(affinity[q_idx])[::-1][:top_k]
    cat_names = raw_dataset.categories

    fig, panels = plt.subplots(1, top_k + 1,
                               figsize=(3.2 * (top_k + 1), 3.6))
    fig.patch.set_facecolor('#1a1a2e')

    def _show(ax, pil_img, title, border_col):
        ax.imshow(pil_img)
        ax.set_title(title, fontsize=8, color=border_col,
                     fontweight='bold', pad=4)
        ax.axis('off')
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_edgecolor(border_col)
            s.set_linewidth(4)

    # Query panel
    q_img, _ = raw_dataset[split_indices[q_idx]]
    _show(panels[0],
          q_img,
          f"Query\n{cat_names[q_class]}",
          '#00bfff')

    # Neighbour panels
    for rank, nb in enumerate(nb_idx):
        nb_img, _  = raw_dataset[split_indices[nb]]
        nb_class   = class_ids[nb]
        match      = (nb_class == q_class)
        col        = '#00e676' if match else '#ff1744'
        verdict    = 'Hit' if match else 'Miss'
        _show(panels[rank + 1],
              nb_img,
              f"Rank {rank+1}  [{verdict}]\n{cat_names[nb_class]}",
              col)

    plt.tight_layout(pad=0.4)

    if out_file:
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        plt.savefig(out_file, dpi=110,
                    bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        print(f"  Saved grid → {out_file}")
    plt.close(fig)


# ── Comparison Bar Chart ──────────────────────────────────────────────────────

def bar_chart_recall(score_dict,
                     out_file='./graphs/recall_comparison.png'):
    """
    Grouped bar chart — Recall@1 and Recall@5 for all experiments.
    """
    exp_names = list(score_dict.keys())
    r1_vals   = [score_dict[e]['r1'] for e in exp_names]
    r5_vals   = [score_dict[e]['r5'] for e in exp_names]

    pos   = np.arange(len(exp_names))
    w     = 0.32

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor('#f8f9fa')
    ax.set_facecolor('#f0f2f5')

    b1 = ax.bar(pos - w/2, r1_vals, w,
                label='Recall@1', color='#4361ee', alpha=0.88)
    b2 = ax.bar(pos + w/2, r5_vals, w,
                label='Recall@5', color='#f72585', alpha=0.88)

    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2,
                h + 0.012, f'{h:.3f}',
                ha='center', va='bottom',
                fontsize=8.5, fontweight='bold')

    ax.set_xticks(pos)
    ax.set_xticklabels(
        ['Contrastive\n(Random Pairs)',
         'Triplet\n(Random)',
         'Triplet\n(Hard Mining)'],
        fontsize=10
    )
    ax.set_ylabel('Recall Score', fontsize=11)
    ax.set_title('Retrieval Performance — Recall@K Comparison',
                 fontsize=13, fontweight='bold', pad=12)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    plt.savefig(out_file, dpi=150, bbox_inches='tight')
    print(f"Saved bar chart → {out_file}")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs('./graphs', exist_ok=True)

    print("Loading Caltech-101 for visualization...")
    raw_ds = Caltech101(root='./data', download=False)

    from dataset import load_caltech101
    _, _, _, test_idx = load_caltech101(root_dir='./data')

    modes      = ['contrastive', 'triplet', 'hardmining']
    scoreboard = {}
    np.random.seed(42)

    for mode in modes:
        print(f"\n{'─'*55}")
        print(f" Experiment : {mode.upper()}")
        print(f"{'─'*55}")

        feat, cids = fetch_embeddings(f'./embeddings/{mode}_test')

        r1 = recall_at_k(feat, cids, k=1)
        r5 = recall_at_k(feat, cids, k=5)
        print(f"  Recall@1 = {r1:.4f}   Recall@5 = {r5:.4f}")

        scoreboard[mode] = {'r1': r1, 'r5': r5}

        # 10 retrieval grids
        query_pool = np.random.choice(len(cids), 10, replace=False)
        for i, q in enumerate(query_pool):
            draw_retrieval_grid(
                q_idx         = q,
                feat_matrix   = feat,
                class_ids     = cids,
                raw_dataset   = raw_ds,
                split_indices = test_idx,
                top_k         = 5,
                out_file      = f'./graphs/{mode}_query_{i+1:02d}.png'
            )

    bar_chart_recall(scoreboard)

    # ── Results Table ─────────────────────────────────────────────────
    print(f"\n{'═'*52}")
    print(f"{'Experiment':<18} {'Recall@1':>10} {'Recall@5':>10}")
    print(f"{'─'*52}")
    for m in modes:
        print(f"{m:<18} "
              f"{scoreboard[m]['r1']:>10.4f} "
              f"{scoreboard[m]['r5']:>10.4f}")
    print(f"{'═'*52}")
    print("\nDone! All plots saved in ./graphs/")