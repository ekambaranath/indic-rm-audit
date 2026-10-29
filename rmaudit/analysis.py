"""Pre-registered analysis: separating a *language* effect from an *India-context* effect.

For one reward model, accuracy = share of seeds where score(chosen) > score(rejected) (ties count 0.5).
The same seeds appear in every language, so observations are clustered by seed. All uncertainty comes from a
cluster bootstrap that resamples whole seeds (independently within each domain).

Contrasts (positive = accuracy drops), for each non-English language L in {hi, te}:

    language_gap_generic[L] = acc(generic, en) - acc(generic, L)     language effect on neutral content
    language_gap_india[L]   = acc(india,   en) - acc(india,   L)     language effect on India content
    context_gap_en          = acc(generic, en) - acc(india,   en)     India-content effect in English
    context_gap[L]          = acc(generic, L)  - acc(india,   L)
    interaction[L]          = language_gap_india[L] - language_gap_generic[L]
                              > 0 means the language drop is larger on India content (the two compound)
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .schema import DOMAINS, LANGS

NON_EN = ("hi", "te")
EFFECTS = (["context_gap_en"]
           + [f"language_gap_generic[{l}]" for l in NON_EN]
           + [f"language_gap_india[{l}]" for l in NON_EN]
           + [f"context_gap[{l}]" for l in NON_EN]
           + [f"interaction[{l}]" for l in NON_EN])


def load_scores(path: str | Path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def _correct(r: dict) -> float:
    d = r["s_chosen"] - r["s_rejected"]
    return 1.0 if d > 0 else (0.5 if d == 0 else 0.0)


def build_matrices(rows: list[dict], model: str) -> dict[str, np.ndarray]:
    """domain -> array [n_seeds, 3] of correctness in LANGS order, keeping only seeds present in all 3."""
    cell: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)  # (domain, seed) -> lang -> correct
    for r in rows:
        if r["model"] == model:
            cell[(r["domain"], r["seed_id"])][r["lang"]] = _correct(r)
    out = {}
    for d in DOMAINS:
        mat = [[v[l] for l in LANGS] for (dd, _), v in sorted(cell.items()) if dd == d and all(l in v for l in LANGS)]
        out[d] = np.array(mat, dtype=float).reshape(-1, len(LANGS))
    return out


def build_margin_matrices(rows: list[dict], model: str) -> dict[str, np.ndarray]:
    """Like build_matrices but with the standardised score margin (s_chosen - s_rejected) / SD(margin).

    The SD is taken over all cells of this model, so effects are in model-specific SD units and comparable across
    models. A continuous outcome keeps information that 0/1 accuracy throws away, so it has more power.
    """
    cell: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    allm = []
    for r in rows:
        if r["model"] == model:
            m = r["s_chosen"] - r["s_rejected"]
            cell[(r["domain"], r["seed_id"])][r["lang"]] = m
            allm.append(m)
    sd = float(np.std(allm)) or 1.0
    out = {}
    for d in DOMAINS:
        mat = [[v[l] / sd for l in LANGS] for (dd, _), v in sorted(cell.items()) if dd == d and all(l in v for l in LANGS)]
        out[d] = np.array(mat, dtype=float).reshape(-1, len(LANGS))
    return out


def compute_effects(acc: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """acc[domain] has shape (..., 3) in LANGS order. Works on a single estimate or a bootstrap batch."""
    g, i = acc["generic"], acc["india"]
    ix = {l: k for k, l in enumerate(LANGS)}
    e = {"context_gap_en": g[..., ix["en"]] - i[..., ix["en"]]}
    for l in NON_EN:
        e[f"language_gap_generic[{l}]"] = g[..., ix["en"]] - g[..., ix[l]]
        e[f"language_gap_india[{l}]"] = i[..., ix["en"]] - i[..., ix[l]]
        e[f"context_gap[{l}]"] = g[..., ix[l]] - i[..., ix[l]]
        e[f"interaction[{l}]"] = e[f"language_gap_india[{l}]"] - e[f"language_gap_generic[{l}]"]
    return e


def cluster_bootstrap(mats: dict[str, np.ndarray], B: int = 4000, seed: int = 0) -> dict[str, np.ndarray]:
    """Bootstrap distribution of per-cell accuracy: domain -> array [B, 3]."""
    rng = np.random.default_rng(seed)
    out = {}
    for d, m in mats.items():
        n = m.shape[0]
        idx = rng.integers(0, n, size=(B, n))
        out[d] = m[idx].mean(axis=1)
    return out


def _ci(x: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    a = (1 - level) / 2
    return float(np.quantile(x, a)), float(np.quantile(x, 1 - a))


def length_baseline(rows: list[dict], model: str) -> dict[str, dict[str, float]]:
    """Accuracy of 'always prefer the longer answer' by cell. Shows how much a pure length shortcut would score."""
    acc: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in rows:
        if r["model"] != model:
            continue
        d = r["ntok_chosen"] - r["ntok_rejected"]
        acc[(r["domain"], r["lang"])].append(1.0 if d > 0 else (0.5 if d == 0 else 0.0))
    return {f"{d}/{l}": float(np.mean(v)) for (d, l), v in sorted(acc.items())}


def token_inflation(rows: list[dict], model: str) -> dict[str, float]:
    """Mean tokens per chosen answer by language, relative to English: how much longer the same content is."""
    tok: dict[str, list[int]] = defaultdict(list)
    for r in rows:
        if r["model"] == model:
            tok[r["lang"]].append(r["ntok_chosen"])
    base = np.mean(tok["en"]) if tok.get("en") else float("nan")
    return {l: float(np.mean(v) / base) for l, v in tok.items()}


def by_error_type(rows: list[dict], model: str) -> dict[str, dict[str, tuple[float, int]]]:
    acc: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in rows:
        if r["model"] == model:
            acc[(r["error_type"], r["lang"])].append(_correct(r))
    out: dict[str, dict[str, tuple[float, int]]] = defaultdict(dict)
    for (et, l), v in acc.items():
        out[et][l] = (float(np.mean(v)), len(v))
    return out


def mean_margin(rows: list[dict], model: str) -> dict[str, float]:
    m: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in rows:
        if r["model"] == model:
            m[(r["domain"], r["lang"])].append(r["s_chosen"] - r["s_rejected"])
    return {f"{d}/{l}": float(np.mean(v)) for (d, l), v in sorted(m.items())}


def analyze_model(rows: list[dict], model: str, B: int = 4000, seed: int = 0) -> dict:
    mats = build_matrices(rows, model)
    if any(m.shape[0] == 0 for m in mats.values()):
        raise ValueError(f"{model}: need seeds with all three languages in both domains")
    est = {d: m.mean(axis=0) for d, m in mats.items()}
    boot = cluster_bootstrap(mats, B, seed)
    eff_est = compute_effects(est)
    eff_boot = compute_effects(boot)
    effects = {}
    for k in EFFECTS:
        lo, hi = _ci(eff_boot[k])
        effects[k] = {"est": float(eff_est[k]), "lo": lo, "hi": hi}
    acc = {d: {l: {"est": float(est[d][j]), "lo": _ci(boot[d][:, j])[0], "hi": _ci(boot[d][:, j])[1]}
               for j, l in enumerate(LANGS)} for d in DOMAINS}
    mm = build_margin_matrices(rows, model)
    m_est = compute_effects({d: m.mean(axis=0) for d, m in mm.items()})
    m_boot = compute_effects(cluster_bootstrap(mm, B, seed + 1))
    margin_effects = {}
    for k in EFFECTS:
        lo, hi = _ci(m_boot[k])
        margin_effects[k] = {"est": float(m_est[k]), "lo": lo, "hi": hi}
    return {"model": model, "n_seeds": {d: int(m.shape[0]) for d, m in mats.items()}, "acc": acc,
            "effects": effects, "margin_effects": margin_effects, "length_baseline": length_baseline(rows, model),
            "token_inflation": token_inflation(rows, model), "mean_margin": mean_margin(rows, model),
            "by_error_type": by_error_type(rows, model)}


def interpret(res: dict) -> list[str]:
    """Rule-based reading following PREREGISTRATION.md. Intervals that include 0 mean 'not distinguishable from
    zero at this sample size', never 'no effect'."""
    e = res["effects"]
    lines = []
    ctx = e["context_gap_en"]
    lines.append("India content is harder in English" if ctx["lo"] > 0 else
                 "No India-content gap detectable in English" if ctx["hi"] >= 0 >= ctx["lo"] else
                 "India content is *easier* in English (unexpected)")
    for l in NON_EN:
        lg, li, ia = e[f"language_gap_generic[{l}]"], e[f"language_gap_india[{l}]"], e[f"interaction[{l}]"]
        if lg["lo"] > 0 and ia["lo"] > 0:
            s = "language drop on neutral content AND a larger drop on India content (effects compound)"
        elif lg["lo"] > 0 and ia["hi"] >= 0 >= ia["lo"]:
            s = "language drop on neutral content, similar drop on India content (language effect, no evidence of compounding)"
        elif lg["hi"] >= 0 >= lg["lo"] and ia["lo"] > 0:
            s = "no language drop on neutral content, but a drop on India content (context, not language)"
        elif lg["hi"] >= 0 >= lg["lo"] and li["lo"] > 0:
            s = "India content drops in this language while neutral content does not"
        else:
            s = "no language drop distinguishable from zero at this sample size"
        lines.append(f"[{l}] {s}")
    return lines


def to_markdown(results: list[dict]) -> str:
    out = []
    for res in results:
        n = res["n_seeds"]
        out.append(f"### {res['model']}  (seeds: india={n['india']}, generic={n['generic']})\n")
        out.append("| domain | en | hi | te |\n|---|---|---|---|")
        for d in DOMAINS:
            cells = [f"{100 * res['acc'][d][l]['est']:.1f} [{100 * res['acc'][d][l]['lo']:.1f}, "
                     f"{100 * res['acc'][d][l]['hi']:.1f}]" for l in LANGS]
            out.append(f"| {d} | " + " | ".join(cells) + " |")
        out.append("\nAccuracy in %, 95% cluster-bootstrap interval in brackets.\n")
        out.append("| effect (points) | estimate | 95% interval |\n|---|---|---|")
        for k in EFFECTS:
            v = res["effects"][k]
            out.append(f"| {k} | {100 * v['est']:+.1f} | [{100 * v['lo']:+.1f}, {100 * v['hi']:+.1f}] |")
        out.append("\nSecondary (more powerful) analysis: standardised score margin, effects in SD units.\n")
        out.append("| effect (SD units) | estimate | 95% interval |\n|---|---|---|")
        for k in EFFECTS:
            v = res["margin_effects"][k]
            out.append(f"| {k} | {v['est']:+.2f} | [{v['lo']:+.2f}, {v['hi']:+.2f}] |")
        out.append("\nReading: " + "; ".join(interpret(res)) + ".\n")
        lb = res["length_baseline"]
        out.append("Length-only baseline accuracy (prefer more tokens): "
                   + ", ".join(f"{k} {100 * v:.0f}%" for k, v in lb.items()))
        out.append("Token inflation vs English: " + ", ".join(f"{k} x{v:.2f}" for k, v in res["token_inflation"].items()))
        out.append("")
    return "\n".join(out)


def plot_interaction(results: list[dict], path: str | Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.6), squeeze=False)
    colors = {"generic": "#6B7785", "india": "#C0504D"}
    for ax, res in zip(axes[0], results):
        for d in DOMAINS:
            y = [100 * res["acc"][d][l]["est"] for l in LANGS]
            lo = [100 * (res["acc"][d][l]["est"] - res["acc"][d][l]["lo"]) for l in LANGS]
            hi = [100 * (res["acc"][d][l]["hi"] - res["acc"][d][l]["est"]) for l in LANGS]
            ax.errorbar(range(3), y, yerr=[lo, hi], marker="o", capsize=3, color=colors[d], label=d)
        ax.set_xticks(range(3))
        ax.set_xticklabels(["English", "Hindi", "Telugu"])
        ax.set_ylabel("pairwise accuracy (%)")
        ax.set_title(res["model"].split("/")[-1], fontsize=9)
        ax.axhline(50, color="#bbb", lw=0.8, ls=":")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=8, title="content", title_fontsize=8)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
