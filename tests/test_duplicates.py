import numpy as np
import pytest
import torch

from lsgen.data.duplicates import duplicate_table, hamming_min, hamming_pairs, resolve_device, within_table


@pytest.mark.skipif(torch.cuda.is_available(), reason="checks the no-GPU guard")
def test_sscd_refuses_cpu_without_flag():
    with pytest.raises(RuntimeError, match="GPU"):
        resolve_device("cuda")
    assert resolve_device("cuda", allow_cpu=True).type == "cpu"


def test_hamming_pairs_known():
    top = 0xFF << 56
    q = np.array([0x0, 0xF, top | 0x1], dtype=np.uint64)
    r = np.array([0x1, 0xF0, top], dtype=np.uint64)
    # distances: q0: [1, 4, 8]  q1: [3, 8, 12]  q2: [8, 13, 1]
    assert {tuple(p) for p in hamming_pairs(q, r, 1).tolist()} == {(0, 0, 1), (2, 2, 1)}
    assert {tuple(p) for p in hamming_pairs(q, r, 3).tolist()} == {(0, 0, 1), (1, 0, 3), (2, 2, 1)}
    assert hamming_min(q, r).tolist() == [1, 3, 1]


def test_duplicate_table_flags_both_criteria():
    rng = np.random.default_rng(0)
    e_tr = rng.normal(size=(5, 64)).astype(np.float32)
    e_tr /= np.linalg.norm(e_tr, axis=1, keepdims=True)
    e_te = np.stack([e_tr[2], -e_tr[0]])
    h_tr = rng.integers(0, 2**63, 5, dtype=np.uint64)
    h_te = np.array([rng.integers(0, 2**63, dtype=np.uint64), h_tr[4] ^ np.uint64(0b11)], dtype=np.uint64)
    assert hamming_min(h_te[:1], h_tr)[0] > 2
    df, nn = duplicate_table(["t0", "t1"], [f"r{i}" for i in range(5)], h_te, h_tr, e_te, e_tr, 2, 0.5)
    assert set(zip(df.test_id, df.train_id)) == {("t0", "r2"), ("t1", "r4")}
    d = df.set_index("train_id")
    assert d.loc["r2", "flag_sscd"] and not d.loc["r2", "flag_phash"]
    assert d.loc["r4", "flag_phash"] and d.loc["r4", "phash_hamming"] == 2
    assert nn.sscd_nn_train_id[0] == "r2" and abs(nn.sscd_nn_sim[0] - 1) < 1e-5


def test_threshold_is_inclusive_and_within_table_excludes_self():
    e = np.eye(4, dtype=np.float32)
    e[1] = [0.75, np.sqrt(1 - 0.75**2), 0, 0]  # cos(e0, e1) = 0.75 exactly
    h = np.array([0, 2**40, 2**50, 2**60], dtype=np.uint64)  # all pairwise Hamming = 1 or 2
    w = within_table(["a", "b", "c", "d"], h, e, 0, 0.75)
    assert set(zip(w.id_a, w.id_b)) == {("a", "b")} and w.flag_sscd.all()
    w = within_table(["a", "b", "c", "d"], h, e, 2, 0.99)
    assert len(w) == 6 and w.flag_phash.all() and not w.flag_sscd.any()
