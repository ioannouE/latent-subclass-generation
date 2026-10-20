"""Pre-check of the DINO-minus-DreamSim identity score (docs/EXPERIMENT_IDSCORE.md): within one make, does subtracting the look-alike
similarity (DreamSim) from the overall similarity (frozen DINO) rank same-subclass pairs higher than DINO alone? Fine labels are used
here only to score the rankings."""
import numpy as np
from sklearn.metrics import roc_auc_score


def pair_zscore(sim):
    """Standardise a square similarity matrix over its off-diagonal entries (the pairs of one make), so two encoders are on one scale."""
    off = ~np.eye(len(sim), dtype=bool)
    return (sim - sim[off].mean()) / sim[off].std()


def identity_score(dino, dreamsim):
    """(n, n) z(DINO cosine) - z(DreamSim cosine) for the L2-normalised embeddings of the images of one make."""
    return pair_zscore(dino @ dino.T) - pair_zscore(dreamsim @ dreamsim.T)


def gate(score, sim, k):
    """score kept only on each row's k most similar other images under sim (the DINO neighbours), -inf elsewhere."""
    s = sim.copy()
    np.fill_diagonal(s, -np.inf)
    keep = np.zeros_like(s, dtype=bool)
    np.put_along_axis(keep, np.argsort(-s, 1)[:, :min(k, len(s) - 1)], True, 1)
    return np.where(keep, score, -np.inf)


def precision_at(score, fine, n=5):
    """Mean share of each image's n highest-scoring other images that have its subclass."""
    s = score.astype(float)
    np.fill_diagonal(s, -np.inf)
    top = np.argsort(-s, 1)[:, :n]
    return float((fine[top] == fine[:, None]).mean())


def pair_auroc(score, fine):
    """AUROC of score for separating same-subclass from different-subclass pairs (each unordered pair once)."""
    i, j = np.triu_indices(len(fine), 1)
    return float(roc_auc_score(fine[i] == fine[j], score[i, j]))


def make_metrics(dino, dreamsim, fine, k=20, n=5):
    """All pre-check numbers for one make; dino, dreamsim are L2-normalised (n_images, d) embeddings."""
    sims = {"dino": dino @ dino.T, "dreamsim": dreamsim @ dreamsim.T}
    sims["identity"] = identity_score(dino, dreamsim)
    out = {f"auroc_{name}": pair_auroc(s, fine) for name, s in sims.items()}
    out |= {f"p{n}_{name}": precision_at(s, fine, n) for name, s in sims.items()}
    out[f"p{n}_gated"] = precision_at(gate(sims["identity"], sims["dino"], k), fine, n)
    out[f"p{n}_chance"] = float(((fine[:, None] == fine[None]).sum(1) - 1).mean() / (len(fine) - 1))
    out["gain_auroc"] = out["auroc_identity"] - out["auroc_dino"]
    out[f"gain_p{n}_gated"] = out[f"p{n}_gated"] - out[f"p{n}_dino"]
    return out
