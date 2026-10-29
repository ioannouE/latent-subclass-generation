"""Error analysis of frozen-encoder subclass neighbours within a make (scripts/error_analysis_neighbours.py): which different-subclass
pairs look most alike, which same-subclass pairs look least alike, and a correct neighbour of each query for reference. Fine labels are
used here only to pick and describe the cases."""
import numpy as np
import pandas as pd

from lsgen.data.labels import load_labels
from lsgen.eval.identity import pair_auroc


def subclass_labels(derived_root, data_root, ids, scheme="make"):
    """Per image: coarse id, coarse name, subclass id, subclass name (scheme "make": make / make-model; "maskcon": body type / class)."""
    lab = load_labels(derived_root, data_root, scheme).loc[ids]
    return lab.make_id.to_numpy(), lab.coarse_name.to_numpy(), lab.fine_id.to_numpy(), lab.fine_name.to_numpy()


def rank_matrix(sim):
    """r[i, j] = position of j in i's neighbour list under sim (1 = nearest, self excluded, ties by index)."""
    s = sim.copy()
    np.fill_diagonal(s, -np.inf)
    return np.argsort(np.argsort(-s, 1, kind="stable"), 1, kind="stable") + 1


def pair_table(make, sub, emb, rank_by="dino"):
    """All unordered same-make pairs (a < b, global indices) of makes with >= 2 subclasses: cosine under each L2-normalised
    embedding in emb, whether the subclass matches, and rank_ab (rank of b in a's list) / rank_ba under emb[rank_by]."""
    rows = []
    for c in np.unique(make):
        idx = np.flatnonzero(make == c)
        if len(np.unique(sub[idx])) < 2:
            continue
        sims = {k: e[idx] @ e[idx].T for k, e in emb.items()}
        rank = rank_matrix(sims[rank_by])
        i, j = np.triu_indices(len(idx), 1)
        rows.append(pd.DataFrame({"make": c, "a": idx[i], "b": idx[j], "same": sub[idx[i]] == sub[idx[j]],
                                  **{k: s[i, j] for k, s in sims.items()}, "rank_ab": rank[i, j], "rank_ba": rank[j, i]}))
    return pd.concat(rows, ignore_index=True)


def as_query(pairs):
    """Orient each pair so `a` is the query: the image that ranks its partner higher (ties: lower index)."""
    out = pairs.copy()
    swap = (pairs.rank_ba < pairs.rank_ab).to_numpy()
    out.loc[swap, ["a", "b", "rank_ab", "rank_ba"]] = pairs.loc[swap, ["b", "a", "rank_ba", "rank_ab"]].to_numpy()
    return out


def failure_cases(pairs, n, key="dino"):
    """False neighbours: the n different-subclass pairs with the highest similarity. Missed neighbours: the n same-subclass pairs with
    the lowest. Ties broken by (a, b); both oriented query first."""
    false = pairs[~pairs.same].sort_values([key, "a", "b"], ascending=[False, True, True], kind="stable").head(n)
    missed = pairs[pairs.same].sort_values([key, "a", "b"], kind="stable").head(n)
    return as_query(false).reset_index(drop=True), as_query(missed).reset_index(drop=True)


def controls(pairs, failures):
    """For each failure (query a, partner b): the query's highest-ranked same-subclass neighbour other than b, or none if it has none.
    Returns (rows, positions in failures they belong to)."""
    same = pairs[pairs.same]
    directed = pd.concat([same, same.rename(columns={"a": "b", "b": "a", "rank_ab": "rank_ba", "rank_ba": "rank_ab"})])
    rows, owner = [], []
    for k, (q, p) in enumerate(zip(failures.a, failures.b)):
        c = directed[(directed.a == q) & (directed.b != p)]
        if len(c):
            rows.append(c.sort_values(["rank_ab", "b"], kind="stable").iloc[0])
            owner.append(k)
    return pd.DataFrame(rows, columns=pairs.columns).reset_index(drop=True), owner


def make_aurocs(make, sub, emb):
    """Per make with >= 2 subclasses: AUROC of the cosine for same vs different subclass pairs, number of images and subclasses."""
    rows = []
    for c in np.unique(make):
        idx = np.flatnonzero(make == c)
        if len(np.unique(sub[idx])) >= 2:
            e = emb[idx]
            rows.append({"make": c, "auroc": pair_auroc(e @ e.T, sub[idx]), "n_images": len(idx), "n_subclasses": len(np.unique(sub[idx]))})
    return pd.DataFrame(rows).sort_values(["auroc", "make"], kind="stable").reset_index(drop=True)
