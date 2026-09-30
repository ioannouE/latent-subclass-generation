import numpy as np

from lsgen.data.duplicates import duplicate_table, hamming_min, hamming_pairs


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
