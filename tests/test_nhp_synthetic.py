"""NHP on synthetic features (CPU, minutes): code correctness, not evidence that the idea works on cars. The backbone is the identity;
the toy has planted subclasses (including single-subclass makes) and strong colour-like / pose-like nuisance shifts (tests/synthetic_nhp.py).
Thresholds were fixed from the chance level of each probe before the final run; `fit` settings are the toy's, not the real run's."""
import functools

import numpy as np
import pytest
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score as ari

from lsgen.eval.representation import cluster, cluster_space, deltas_per_make, hungarian_acc, l2norm, select_k
from lsgen.methods.repr.losses import supcon_loss
from lsgen.methods.repr.nhp import (Head, active_prototypes, adversary_loss, assignment_loss, hierarchy_loss, l0_penalty, nuisance_loss,
                                    usage_loss, usage_share)
from tests.synthetic_nhp import DEFAULTS, N_COLOUR, N_POSE, Toy, embed, fit

pytestmark = pytest.mark.filterwarnings("ignore:User provided device_type of 'cuda'")  # forward_fp16 on a CPU-only machine

STEPS = 800
USAGE, L0 = 0.5, 1.0  # toy weights of the anti-collapse (usage) and L0 terms; the number of active prototypes is sensitive to both


@pytest.fixture(scope="module")
def toy():
    return Toy()


@functools.lru_cache(maxsize=None)
def trained(name):
    """Fitted NHP variants on the toy (cached across tests)."""
    variants = {"full": {}, "no_invariance": dict(invariance=False), "no_hierarchy": dict(w_hierarchy=0.0),
                "dense": dict(w_usage=USAGE), "sparse": dict(sparse=True, w_usage=USAGE, w_l0=L0),
                "make_only": dict(w_assign=0.0, w_usage=0.0, w_hierarchy=0.0, w_nuis=0.0)}
    return fit(Toy(), steps=STEPS, **variants[name])


def z_sub(name, toy):
    return embed(trained(name), toy.real())[1]


def probe(z, y, seed=0):
    """Held-out accuracy of a linear probe on L2-normalised z (random half split)."""
    z, y = l2norm(z.numpy()), y.numpy()
    idx = np.random.default_rng(seed).permutation(len(z))
    a, b = idx[:len(z) // 2], idx[len(z) // 2:]
    return LogisticRegression(max_iter=1000).fit(z[a], y[a]).score(z[b], y[b])


def recovery(z, toy):
    """Per-make k-means (oracle K, the evaluation pipeline's PCA space) ACC and ARI against the planted subclasses, macro over makes with K_c >= 2."""
    make, sub = toy.make.numpy(), toy.sub_id.numpy()
    space, acc, adj = cluster_space(np.asarray(z), make), [], []
    for m, k in enumerate(toy.k_per_make):
        if k >= 2:
            i, c = make == m, cluster(space[make == m], k)
            acc.append(hungarian_acc(sub[i], c))
            adj.append(ari(sub[i], c))
    return float(np.mean(acc)), float(np.mean(adj))


def active_counts(name, toy, data=None):
    ob, data = trained(name), toy if data is None else data
    head, z = ob.net[1], embed(trained(name), data.real())[1]
    with torch.no_grad():
        p = (head.logits(z, data.make) / ob.tau_s + head.log_gate(data.make, sample=False)).softmax(1)
    return active_prototypes(usage_share(p, data.make, len(data.k_per_make)), DEFAULTS["active_min"]).tolist()


# 1. shapes, determinism, finite gradients
def loss_terms(seed=0):
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(64, 8, generator=g, requires_grad=True)
    make = torch.randint(0, 3, (64,), generator=g)
    head = Head(8, 16, (8, 8, 4), 3, 5, sparse=True, adv_classes=(N_COLOUR, N_POSE))
    logit_s, logit_t = (head.logits(z, make) / t + head.log_gate(make, sample=s) for t, s in ((0.1, True), (0.04, False)))
    return z, {
        "make": lambda: supcon_loss(z, make, 0.1),
        "assign": lambda: assignment_loss(logit_s, logit_t),
        "usage": lambda: usage_loss(logit_s.softmax(1), make),
        "hierarchy": lambda: hierarchy_loss(z, logit_s.softmax(1), make, 0.1, 0.1),
        "nuis": lambda: nuisance_loss(z),
        "l0": lambda: l0_penalty(head.log_alpha),
        "adversary": lambda: adversary_loss(head, z, torch.stack([torch.randint(0, c, (64,), generator=g) for c in head.adv_classes], 1))}


@pytest.mark.parametrize("term", ["make", "assign", "usage", "hierarchy", "nuis", "l0", "adversary"])
def test_each_loss_is_finite_with_finite_gradients(term):
    z, terms = loss_terms()
    loss = terms[term]()
    assert loss.shape == () and loss.dtype == torch.float32 and torch.isfinite(loss)
    loss.backward()
    assert z.grad is None or (torch.isfinite(z.grad).all())


def test_shapes_determinism_and_joint_gradient(toy):
    runs = [fit(toy, steps=5, sparse=True, adversary=True, w_adversary=0.1, seed=3) for _ in range(2)]
    idx = torch.arange(256)
    runs[0].loss([*toy.views(idx, True), toy.make[idx]])  # a fresh graph: the joint loss of every term
    for a, b in zip(runs[0].net[1].parameters(), runs[1].net[1].parameters()):
        assert torch.equal(a, b)  # same seed, same parameters
    zm, zs, zn = embed(runs[0], toy.real())
    assert (zm.shape, zs.shape, zn.shape) == ((len(toy), 8), (len(toy), 8), (len(toy), 4)) and zs.dtype == torch.float32
    assert set(runs[0].terms) == {"make", "assign", "usage", "hierarchy", "nuis", "l0", "adversary"}
    total = sum(runs[0].weights[k] * v for k, v in runs[0].terms.items())
    total.backward()
    grads = [p.grad for p in runs[0].net[1].parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_adversary_reverses_the_gradient():
    head = Head(8, 16, (8, 8, 4), 3, 5, adv_classes=(2,))
    y = torch.randint(0, 2, (32, 1))
    z = torch.randn(32, 8, requires_grad=True)
    adversary_loss(head, z, y).backward()
    plain = torch.randn(32, 8).requires_grad_()
    plain.data = z.data.clone()
    torch.nn.functional.cross_entropy(head.adversary(plain), y[:, 0]).backward()
    assert torch.allclose(z.grad, -plain.grad, atol=1e-6)


# 2. each term decreases when optimised alone
@pytest.mark.parametrize("term", ["make", "assign", "usage", "hierarchy", "nuis", "l0"])
def test_each_term_decreases_when_optimised_alone(toy, term):
    only = {f"w_{k}": 1.0 if k == term else 0.0 for k in ("make", "assign", "usage", "hierarchy", "nuis", "l0", "adversary")}
    extra = dict(margins=(1.0, 1.0)) if term == "hierarchy" else {}  # with the default margins the hinge is already satisfied at initialisation
    before = fit(toy, steps=1, sparse=term == "l0", **only, **extra)
    after = fit(toy, steps=300, sparse=term == "l0", **only, **extra)
    assert after.log[-1][term] < before.log[0][term] - 1e-3, (before.log[0][term], after.log[-1][term])


# 3. invariance: nuisance probes, subclass probe
def test_invariance_removes_nuisance_but_keeps_subclass(toy):
    on, off = z_sub("full", toy), z_sub("no_invariance", toy)
    chance_colour, chance_pose = 1 / N_COLOUR, 1 / N_POSE
    colour_on, pose_on = probe(on, toy.colour), probe(on, toy.pose)
    colour_off, pose_off = probe(off, toy.colour), probe(off, toy.pose)
    print(f"\nprobe accuracy (chance colour {chance_colour:.2f}, pose {chance_pose:.2f}): invariance on colour {colour_on:.3f} pose {pose_on:.3f}; "
          f"off colour {colour_off:.3f} pose {pose_off:.3f}; subclass probe on {probe(on, toy.sub_id):.3f} off {probe(off, toy.sub_id):.3f}")
    assert colour_on < chance_colour + 0.10 and pose_on < chance_pose + 0.10
    assert colour_off > chance_colour + 0.30 and pose_off > chance_pose + 0.10
    assert probe(on, toy.sub_id) > 0.9


# 4. sparsity
def test_sparsity_gives_single_subclass_makes_one_prototype(toy):
    planted, dense, sparse = np.array(toy.k_per_make), np.array(active_counts("dense", toy)), np.array(active_counts("sparse", toy))
    ks = {}
    for name in ("dense", "sparse"):
        z = cluster_space(z_sub(name, toy).numpy(), toy.make.numpy())
        ks[name] = np.mean([select_k(z[toy.make.numpy() == m]) - 1 for m in np.flatnonzero(planted == 1)])
    print(f"\nactive prototypes per make: planted {planted.tolist()}, without gates {dense.tolist()}, with gates {sparse.tolist()}; "
          f"K-hat error on the single-subclass makes: without gates {ks['dense']:.2f}, with gates {ks['sparse']:.2f}")
    assert (sparse[planted == 1] == 1).all()
    assert abs(sparse[planted > 1].mean() - planted[planted > 1].mean()) <= 1


# 5. hierarchy
def test_hierarchy_order_and_margin_term(toy):
    def ratios(name):
        d = deltas_per_make(z_sub(name, toy).numpy(), toy.make.numpy(), toy.sub_id.numpy())
        d = d[d.K_c >= 2]
        return d, (d.d0 / d.d1).mean() + (d.d1 / d.d2).mean()
    d, with_margin = ratios("full")
    _, without = ratios("no_hierarchy")
    print(f"\nDelta0 / Delta1 + Delta1 / Delta2 (macro): with the margin term {with_margin:.3f}, without {without:.3f}")
    assert ((d.d0 < d.d1) & (d.d1 < d.d2)).all()
    assert with_margin < without


# 6. recovery
def test_recovery_beats_raw_features_and_make_only_head(toy):
    nhp, raw = recovery(z_sub("full", toy), toy), recovery(toy.real().numpy(), toy)
    make_only = recovery(embed(trained("make_only"), toy.real())[0], toy)
    print(f"\nper-make k-means (ACC, ARI): NHP z_sub {nhp}, raw features {raw}, make-only SupCon head {make_only}")
    assert nhp[0] > raw[0] + 0.1 and nhp[1] > raw[1] + 0.1
    assert nhp[0] > make_only[0] + 0.1 and nhp[1] > make_only[1] + 0.1


# 7. negative control
@pytest.mark.xfail(strict=True, reason="measured failure of the design on this control, see the report: a continuous signal without clusters is "
                                       "still partitioned into confident, view-consistent assignments")
def test_negative_control_finds_no_confident_subclasses():
    toy = Toy(structured=False)
    ob = fit(toy, steps=STEPS)
    head, z = ob.net[1], embed(ob, toy.real())[1]
    with torch.no_grad():
        p = (head.logits(z, toy.make) / ob.tau_s).softmax(1)
    assert p.max(1).values.mean() < 0.6  # assignments stay diffuse, as in the high-entropy regime
