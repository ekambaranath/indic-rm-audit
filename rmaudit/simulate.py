"""Simulation: does the analysis recover known effects, and how many seeds do we need?

Generative model (logit scale). For seed i in domain d and language l:

    P(correct) = sigmoid(mu + a_i + b[d, l]),   a_i ~ N(0, sd_item^2) shared across the three languages

The shared item effect makes the same seed correlated across languages (as in real data), which is why the
analysis bootstraps whole seeds. We compute the TRUE accuracy-point effects by integrating over a_i, then check
(1) coverage of the 95% interval and (2) power = share of simulated studies whose interval excludes 0.
"""

from __future__ import annotations

import numpy as np

from .analysis import EFFECTS, compute_effects
from .schema import DOMAINS, LANGS


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def true_accuracy(mu: float, sd_item: float, b: dict, n_mc: int = 400_000, seed: int = 123) -> dict:
    a = np.random.default_rng(seed).normal(0, sd_item, n_mc)
    return {d: np.array([_sigmoid(mu + a + b[(d, l)]).mean() for l in LANGS]) for d in DOMAINS}


def simulate_study(n_seeds: int, mu: float, sd_item: float, b: dict, rng) -> dict[str, np.ndarray]:
    mats = {}
    for d in DOMAINS:
        a = rng.normal(0, sd_item, size=(n_seeds, 1))
        p = _sigmoid(mu + a + np.array([b[(d, l)] for l in LANGS])[None, :])
        mats[d] = (rng.random(p.shape) < p).astype(float)
    return mats


def _boot_effects(mats, B, rng):
    boot = {}
    for d, m in mats.items():
        n = m.shape[0]
        idx = rng.integers(0, n, size=(B, n))
        boot[d] = m[idx].mean(axis=1)
    return compute_effects(boot)


def coverage_and_power(n_seeds: int, mu: float, sd_item: float, b: dict, n_sims: int = 400, B: int = 300,
                       seed: int = 0, effect: str = "interaction[hi]") -> dict:
    assert effect in EFFECTS
    rng = np.random.default_rng(seed)
    truth = float(compute_effects(true_accuracy(mu, sd_item, b))[effect])
    cover = excl = 0
    est_all = []
    for _ in range(n_sims):
        mats = simulate_study(n_seeds, mu, sd_item, b, rng)
        est = float(compute_effects({d: m.mean(axis=0) for d, m in mats.items()})[effect])
        boot = _boot_effects(mats, B, rng)[effect]
        lo, hi = np.quantile(boot, [0.025, 0.975])
        cover += lo <= truth <= hi
        excl += (lo > 0) or (hi < 0)
        est_all.append(est)
    return {"n_seeds": n_seeds, "true_effect_pts": 100 * truth, "coverage": cover / n_sims,
            "power": excl / n_sims, "mean_estimate_pts": 100 * float(np.mean(est_all))}


def scenario(mu: float, lang_logit: float, context_logit: float, interaction_logit: float) -> dict:
    """Effects in logit units: Hindi/Telugu both get lang_logit (negative = worse); India content gets
    context_logit in every language; interaction_logit applies only to India content in hi/te."""
    b = {}
    for d in DOMAINS:
        for l in LANGS:
            v = 0.0
            if l != "en":
                v += lang_logit
            if d == "india":
                v += context_logit
                if l != "en":
                    v += interaction_logit
            b[(d, l)] = v
    return b


def power_table(n_grid=(50, 100, 150, 200, 300), mu: float = 2.0, sd_item: float = 1.0, lang_logit: float = -0.4,
                context_logit: float = -0.3, interaction_logits=(-0.25, -0.5, -0.75), n_sims: int = 400,
                B: int = 300) -> list[dict]:
    rows = []
    for il in interaction_logits:
        b = scenario(mu, lang_logit, context_logit, il)
        for n in n_grid:
            r = coverage_and_power(n, mu, sd_item, b, n_sims=n_sims, B=B, seed=n)
            r["interaction_logit"] = il
            rows.append(r)
    return rows


def margin_vs_accuracy_power(n_seeds: int, mu: float = 1.2, sd_item: float = 0.7, sd_eps: float = 0.7,
                             lang_shift: float = -0.15, context_shift: float = -0.1,
                             interaction_shift: float = -0.2, n_sims: int = 300, B: int = 300, seed: int = 0,
                             effect: str = "interaction[hi]") -> dict:
    """Same simulated data analysed two ways: 0/1 accuracy vs standardised continuous margin.

    margin[i,d,l] = mu + shift[d,l] + a_i + e_il, a_i shared across languages (sd_item), e_il independent.
    Accuracy = share of margins > 0. Power = share of simulations whose 95% bootstrap CI excludes 0.
    """
    b = scenario(0.0, lang_shift, context_shift, interaction_shift)
    rng = np.random.default_rng(seed)
    hit_acc = hit_mar = 0
    for _ in range(n_sims):
        mats_m = {}
        for d in DOMAINS:
            a = rng.normal(0, sd_item, size=(n_seeds, 1))
            e = rng.normal(0, sd_eps, size=(n_seeds, len(LANGS)))
            mats_m[d] = mu + np.array([b[(d, l)] for l in LANGS])[None, :] + a + e
        sd = np.std(np.concatenate([m.ravel() for m in mats_m.values()]))
        mats_acc = {d: (m > 0).astype(float) for d, m in mats_m.items()}
        mats_z = {d: m / sd for d, m in mats_m.items()}
        for mats, which in ((mats_acc, "acc"), (mats_z, "mar")):
            lo, hi = np.quantile(_boot_effects(mats, B, rng)[effect], [0.025, 0.975])
            if lo > 0 or hi < 0:
                if which == "acc":
                    hit_acc += 1
                else:
                    hit_mar += 1
    return {"n_seeds": n_seeds, "power_accuracy": hit_acc / n_sims, "power_margin": hit_mar / n_sims}
