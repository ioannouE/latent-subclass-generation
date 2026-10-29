import numpy as np

from lsgen.eval.neighbours import controls, failure_cases, pair_table, rank_matrix


def unit(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def test_rank_matrix_excludes_self():
    sim = np.array([[1, .2, .9], [.2, 1, .5], [.9, .5, 1]])
    assert np.array_equal(rank_matrix(sim), [[3, 2, 1], [2, 3, 1], [1, 2, 3]])


def test_cases_stay_within_make_and_controls_pick_the_nearest_correct_neighbour():
    # make 0: subclasses 0, 0, 1, 0; image 2 (subclass 1) sits next to image 0. make 1 is a near copy of make 0 but has one subclass.
    x = unit(np.array([[1, 0], [.8, .6], [.99, .1], [0, 1], [1, 0], [.99, .1]]))
    make, sub = np.array([0, 0, 0, 0, 1, 1]), np.array([0, 0, 1, 0, 2, 2])
    pairs = pair_table(make, sub, {"dino": x})
    assert set(pairs.make) == {0} and len(pairs) == 6
    false, missed = failure_cases(pairs, 1)
    assert (false.a[0], false.b[0], false.rank_ab[0]) == (0, 2, 1) and not false.same[0]   # cosine 0.995, mutual nearest neighbours
    assert (missed.a[0], missed.b[0], missed.rank_ab[0]) == (0, 3, 3) and missed.same[0]   # cosine 0
    for failures in (false, missed):  # query 0's same-subclass neighbours: 1 (rank 2), 3 (rank 3, the missed partner)
        ctrl, owner = controls(pairs, failures)
        assert owner == [0] and (ctrl.a[0], ctrl.b[0], ctrl.rank_ab[0]) == (0, 1, 2) and ctrl.same[0]


def test_query_is_the_image_that_ranks_its_partner_higher():
    # image 1 is image 0's 2nd neighbour, but image 0 is image 1's 1st: the pair is reported with query 1
    x = unit(np.array([[1, 0], [.9, .44], [.99, -.1], [-.5, 1]]))
    pairs = pair_table(np.zeros(4, int), np.array([0, 1, 2, 2]), {"dino": x})
    false, _ = failure_cases(pairs, 2)
    row = false[(false[["a", "b"]].min(1) == 0) & (false[["a", "b"]].max(1) == 1)].iloc[0]
    assert (row.a, row.b, row.rank_ab, row.rank_ba) == (1, 0, 1, 2)
