# tsne_plots.py
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def plot_tsne(embeddings, labels, title, out_file):
    """
    Reduces embeddings to 2D using t-SNE and saves scatter plot.
    """
    print(f"Running t-SNE for {title}...")

    tsne   = TSNE(n_components=2, perplexity=30,
                  random_state=42, max_iter=1000)
    coords = tsne.fit_transform(embeddings)

    unique_labels = np.unique(labels)
    cmap          = plt.cm.get_cmap('tab20', len(unique_labels))

    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#16213e')

    for i, lbl in enumerate(unique_labels):
        mask = labels == lbl
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=8,
            color=cmap(i),
            alpha=0.7,
            linewidths=0
        )

    ax.set_title(title, fontsize=14,
                 fontweight='bold', color='white', pad=12)
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#444')

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    plt.savefig(out_file, dpi=150,
                bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print(f"Saved → {out_file}")
    plt.close(fig)


if __name__ == "__main__":
    experiments = [
        ('contrastive', 'Plot 1 — Contrastive Loss Embeddings'),
        ('triplet',     'Plot 2 — Triplet Loss (Random) Embeddings'),
        ('hardmining',  'Plot 3 — Triplet Loss (Hard Mining) Embeddings'),
    ]

    for mode, title in experiments:
        emb = np.load(f'./embeddings/{mode}_test_embeddings.npy')
        lbl = np.load(f'./embeddings/{mode}_test_labels.npy')

        plot_tsne(
            embeddings = emb,
            labels     = lbl,
            title      = title,
            out_file   = f'./graphs/tsne_{mode}.png'
        )

    print("\nAll t-SNE plots saved in ./graphs/")