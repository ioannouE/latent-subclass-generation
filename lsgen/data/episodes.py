"""Use-case-B episodes (docs/PLAN.md, Milestone 2). Frozen before any method runs.

An episode is a support set S (k images of one subclass, or of two subclasses of one make for "mixed"
episodes) and a target T: the remaining test images of the same subclass(es), T and S disjoint.
Single-class episodes store `support` / `target`; mixed episodes store `support_by_class` / `target_by_class`
(the flat sets are their unions; see `flat`).
Fine labels are used here to construct episodes (allowed: episode construction); methods only ever
receive the support images, never the episode's fine ids.

Rules (all enforced below and tested in tests/test_episodes.py):
- conflicting images (metadata `exclude`) never appear in S or T;
- no known near-duplicate pair between S and T (S is redrawn, up to `max_redraws`, then the episode is
  kept and its remaining pairs recorded as `n_near_dup_S_T`);
- every episode has its own integer seed derived from (base seed, family, k, class ids, repetition).
"""
import itertools
import logging

import numpy as np

log = logging.getLogger(__name__)
FAMILIES = {"single_test": 0, "single_train": 1, "mixed": 2}


def episode_seed(base_seed, family, k, *keys):
    return int(np.random.SeedSequence([base_seed, FAMILIES[family], k, *keys]).generate_state(1)[0])


def _n_pairs(S, T, near_dup):
    T = set(T)
    return sum(1 for s in S for t in near_dup.get(s, ()) if t in T)


def _draw(rng, pools, counts, targets, near_dup, max_redraws):
    """Draw len(pools) disjoint supports (sizes `counts`) avoiding near-duplicates with the targets."""
    for attempt in range(max_redraws + 1):
        S = [sorted(rng.choice(p, n, replace=False).tolist()) for p, n in zip(pools, counts)]
        T = [sorted(set(t) - set(s)) for t, s in zip(targets, S)]
        n = _n_pairs([i for s in S for i in s], [i for t in T for i in t], near_dup)
        if n == 0:
            break
    return S, T, n, attempt


def build_single(support_pools, target_pools, make_of, ks, n_episodes, min_target, base_seed, family,
                 near_dup, max_redraws=100):
    """support_pools / target_pools: {fine_id: sorted list of ids}. For family="single_test" they are the
    same test pool (T = pool minus S); for "single_train", S comes from train and T is all test images.
    Returns (episodes, eligibility {k: n eligible classes}, skipped [(k, fine_id, reason)])."""
    episodes, eligible, skipped = [], {}, []
    for k in ks:
        eligible[k] = 0
        for g in sorted(target_pools):
            sp, tp = support_pools.get(g, []), target_pools[g]
            n_target = len(tp) - (k if family == "single_test" else 0)
            if len(sp) < k or n_target < min_target:
                skipped.append((k, g, f"support pool {len(sp)} < k={k}" if len(sp) < k
                                else f"target {n_target} < {min_target}"))
                continue
            eligible[k] += 1
            for rep in range(n_episodes):
                seed = episode_seed(base_seed, family, k, g, rep)
                (S,), (T,), n, att = _draw(np.random.default_rng(seed), [sp], [k], [tp], near_dup, max_redraws)
                episodes.append({"episode_id": f"{family}_k{k}_c{g:03d}_r{rep}", "family": family,
                                 "support_source": "test" if family == "single_test" else "train",
                                 "k": k, "make_id": make_of[g], "fine_ids": [g], "ratio": 1.0, "seed": seed,
                                 "support": S, "target": T, "n_near_dup_S_T": n, "redraws": att})
    return episodes, eligible, skipped


def build_mixed(test_pools, make_of, k, ratios, n_episodes, min_target, max_pairs_per_make, base_seed,
                near_dup, max_redraws=100):
    """Supports from two subclasses (a, b) of the same make: round(r*k) images of a, the rest of b.
    Pairs per make: all if <= max_pairs_per_make, else a seeded sample. T = remaining test images of a and b."""
    episodes, skipped, pairs_used = [], [], {}
    by_make = {}
    for g in sorted(test_pools):
        by_make.setdefault(make_of[g], []).append(g)
    for c, classes in sorted(by_make.items()):
        pairs = list(itertools.combinations(classes, 2))
        if not pairs:
            continue
        if len(pairs) > max_pairs_per_make:
            rng = np.random.default_rng(episode_seed(base_seed, "mixed", k, c, 10**6))
            pairs = [pairs[i] for i in sorted(rng.choice(len(pairs), max_pairs_per_make, replace=False))]
        pairs_used[c] = len(pairs)
        for a, b in pairs:
            for r in ratios:
                na = int(round(r * k))
                counts = [na, k - na]
                if any(len(test_pools[g]) - n < min_target for g, n in zip((a, b), counts)):
                    skipped.append((k, (a, b), r, "target < min_target"))
                    continue
                for rep in range(n_episodes):
                    seed = episode_seed(base_seed, "mixed", k, a, b, int(round(100 * r)), rep)
                    (Sa, Sb), (Ta, Tb), n, att = _draw(np.random.default_rng(seed), [test_pools[a], test_pools[b]],
                                                       counts, [test_pools[a], test_pools[b]], near_dup, max_redraws)
                    episodes.append({"episode_id": f"mixed_k{k}_c{a:03d}-{b:03d}_r{int(round(100 * r)):02d}_{rep}",
                                     "family": "mixed", "support_source": "test", "k": k, "make_id": c,
                                     "fine_ids": [a, b], "ratio": r, "seed": seed,
                                     "support_by_class": {str(a): Sa, str(b): Sb},
                                     "target_by_class": {str(a): Ta, str(b): Tb}, "n_near_dup_S_T": n, "redraws": att})
    return episodes, pairs_used, skipped


def flat(ep, key):
    """ep["support"] / ep["target"] for any family (union over classes for mixed episodes)."""
    return ep[key] if key in ep else sorted(i for ids in ep[f"{key}_by_class"].values() for i in ids)


def near_dup_index(pairs):
    """[(id_a, id_b), ...] -> {id: set of near-duplicate ids} (symmetric)."""
    idx = {}
    for a, b in pairs:
        idx.setdefault(a, set()).add(b)
        idx.setdefault(b, set()).add(a)
    return idx
