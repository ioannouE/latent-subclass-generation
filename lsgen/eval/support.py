"""Use case B metrics B1-B6 (docs/METRICS.md). Per episode: embeddings of the N generated images, of the support S and of
the target T = remaining test images of the subclass. B6 is B1-B4 tabulated per k (see scripts/eval_generation.py)."""
import numpy as np

from lsgen.eval.memorization import nearest_similarity
from lsgen.eval.synthesis import kid, prdc


def hit_rate(probs, fine_id):
    """B1: share of generated images the fine-class evaluator assigns to the support's subclass."""
    return float((probs.argmax(1) == fine_id).mean())


def mean_pairwise_distance(x):
    """Mean cosine distance over all pairs of distinct rows."""
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    return float((1 - x @ x.T)[np.triu_indices(len(x), 1)].mean())


def beyond_support(gen, support, target, k=3):
    """B2: KID and coverage of the samples against T (the part of the subclass the support did not show) and against S.
    Against S the values are NaN when S has too few images (KID needs >= 2, coverage > k)."""
    return {"kid_T": kid(gen, target), "coverage_T": prdc(target, gen, k)["coverage"],
            "kid_S": kid(gen, support), "coverage_S": prdc(support, gen, k)["coverage"]}


def exemplar_dependence(gen, support, target):
    """B3 (SSCD embeddings): mean max similarity of samples to S and to T; ref = the same for real T images against S
    (how close an unseen real image is to the support). ratio_S > 1: samples sit closer to S than real images do."""
    s_gen = nearest_similarity(gen, support)[0].mean()
    ref = nearest_similarity(target, support)[0].mean()
    return {"sim_S": s_gen, "sim_T": nearest_similarity(gen, target)[0].mean(), "sim_T_to_S": ref, "ratio_S": s_gen / ref}


def diversity(gen, target):
    """B4 (CLIP embeddings): pairwise distance among the samples, and relative to the pairwise distance within T."""
    d_gen, d_real = mean_pairwise_distance(gen), mean_pairwise_distance(target)
    return {"div_gen": d_gen, "div_T": d_real, "div_ratio": d_gen / d_real}


def mixture_error(probs, fine_ids, ratio):
    """B5: |predicted share of subclass a - r|; the evaluator chooses between the episode's two subclasses (a, b)."""
    a, b = fine_ids
    return float(abs((probs[:, a] > probs[:, b]).mean() - ratio))
