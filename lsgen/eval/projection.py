"""2-D projections of embeddings for plots (descriptive only). UMAP and t-SNE on L2-normalised embeddings."""
import numpy as np
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from lsgen.eval.representation import l2norm


def project(x, cfg):
    """{"UMAP": (n, 2), "t-SNE": (n, 2)}; cfg keys: umap_neighbors, tsne_perplexity, seed, n_jobs."""
    x = l2norm(x)
    z_umap = umap.UMAP(n_neighbors=cfg["umap_neighbors"], metric="cosine", random_state=cfg["seed"]).fit_transform(x)
    xp = PCA(min(50, x.shape[1]), random_state=cfg["seed"]).fit_transform(x)
    z_tsne = TSNE(perplexity=cfg["tsne_perplexity"], init="pca", random_state=cfg["seed"], n_jobs=cfg["n_jobs"]).fit_transform(xp)
    return {"UMAP": z_umap, "t-SNE": z_tsne}


def cached_projection(path, x, cfg):
    """project(x, cfg), stored in an .npz at `path` and reused on later runs (delete it to recompute)."""
    if path.exists():
        return dict(np.load(path))
    proj = project(x, cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **proj)
    return proj
