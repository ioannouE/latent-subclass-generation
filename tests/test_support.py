import numpy as np
import pytest

from lsgen.eval.evaluators import EVALUATORS, check_disjoint
from lsgen.eval.support import beyond_support, diversity, exemplar_dependence, hit_rate, mean_pairwise_distance, mixture_error

rng = np.random.default_rng(0)


def test_hit_rate_and_mixture_error():
    probs = np.eye(3)[[0, 0, 1, 2]]
    assert hit_rate(probs, 0) == 0.5
    assert mixture_error(probs, (0, 1), 0.5) == 0  # rows 0, 1 prefer a; rows 2, 3 do not
    assert np.isclose(mixture_error(probs, (0, 1), 0.2), 0.3) and mixture_error(np.eye(3)[[0] * 4], (0, 1), 1.0) == 0


def test_pairwise_distance_known():
    assert np.isclose(mean_pairwise_distance(np.array([[1.0, 0], [0, 1]])), 1)
    assert np.isclose(mean_pairwise_distance(np.array([[1.0, 0], [2, 0]])), 0)  # cosine: scale-free


def test_prototype_collapse_has_no_diversity_and_exemplar_copy_is_flagged():
    target = rng.normal(size=(30, 16))
    support = rng.normal(size=(5, 16))
    collapsed = np.repeat(target[:1], 16, axis=0)
    assert diversity(collapsed, target)["div_ratio"] == pytest.approx(0, abs=1e-9)
    assert 0.7 < diversity(target[:16], target)["div_ratio"] < 1.3
    copy = exemplar_dependence(np.repeat(support, 4, axis=0)[:16], support, target)
    fresh = exemplar_dependence(target[:16], support, target)
    assert np.isclose(copy["sim_S"], 1) and copy["ratio_S"] > 1.5 and 0.5 < fresh["ratio_S"] < 1.5  # fresh samples ~ real T


def test_beyond_support_distinguishes_target_from_support_copy():
    mu = 2 * np.eye(8)[0]
    target = mu + rng.normal(size=(60, 8))
    support = mu + rng.normal(size=(8, 8))
    fresh = mu + rng.normal(size=(16, 8))
    m = beyond_support(fresh, support, target)
    assert abs(m["kid_T"]) < 0.5 and m["coverage_T"] > 0.3
    assert np.isnan(beyond_support(fresh, support[:1], target)["kid_S"])  # k = 1: KID against S undefined
    far = beyond_support(-mu + rng.normal(size=(16, 8)), support, target)
    assert far["kid_T"] > m["kid_T"] and far["coverage_T"] < m["coverage_T"]


def test_evaluator_disjointness_rule():
    assert any("dinov2_large" in w for w in check_disjoint(["dinov3_large"]))  # same family: flagged, not refused
    assert check_disjoint(["supervised_resnet50"]) == []
    for name in set(EVALUATORS.values()):
        with pytest.raises(ValueError, match="circularity"):
            check_disjoint(["sd_vae", name])
