import json

import numpy as np
import pytest

from rmaudit import analysis, review, simulate
from rmaudit.schema import Item, LANGS, load_jsonl, make_id, save_jsonl, usable_items, validate
from rmaudit.scoring import MockRM, score_with_cache


def mk(seed, domain="india", lang="en", status="verified", url="http://x", **kw):
    d = dict(id=make_id(seed, lang), seed_id=seed, domain=domain, lang=lang, error_type="wrong_entity",
             prompt="Q?", chosen="Delhi is the capital.", rejected="Mumbai is the capital.", status=status,
             source_url=url if lang == "en" else "")
    d.update(kw)
    return Item(**d)


def full_seed(seed, domain="india"):
    return [mk(seed, domain, "en"), mk(seed, domain, "hi", "post_edited"), mk(seed, domain, "te", "post_edited")]


# ---------------- schema ----------------
def test_validate_strict_ok_and_failures():
    items = full_seed("a")
    assert validate(items, strict=True)[0] == []
    bad = [mk("b"), mk("b", lang="hi", status="machine_translated")]
    errs = validate(bad, strict=True)[0]
    assert any("missing languages" in e for e in errs)
    assert any("not final" in e for e in errs)
    assert any("no source_url" in e for e in validate([mk("c", url="")] + full_seed("c")[1:], strict=True)[0])


def test_validate_identical_and_duplicate():
    it = mk("a", rejected="Delhi is the capital.")
    assert any("identical" in e for e in validate([it])[0])
    assert any("duplicate" in e for e in validate([mk("a"), mk("a")])[0])


def test_roundtrip_unicode(tmp_path):
    it = mk("a", lang="te", status="post_edited", prompt="భారతదేశ రాజధాని ఏది?")
    p = tmp_path / "x.jsonl"
    save_jsonl([it], p)
    assert load_jsonl(p)[0].prompt == "భారతదేశ రాజధాని ఏది?"


def test_usable_drops_incomplete_and_rejected():
    items = full_seed("a") + [mk("b")] + full_seed("c")
    items[-1].status = "rejected"
    assert {i.seed_id for i in usable_items(items)} == {"a", "c"} - {"c"} | {"a"} or True
    assert {i.seed_id for i in usable_items(items)} == {"a"}


# ---------------- review round trip ----------------
def test_review_roundtrip(tmp_path):
    items = review.init_langs([mk("a"), mk("b", "generic")])
    assert len(items) == 6
    csvp = tmp_path / "r.csv"
    review.export_review(items, csvp)
    import csv
    rows = list(csv.DictReader(open(csvp, encoding="utf-8-sig")))
    for r in rows:
        if r["lang"] == "hi" and r["seed_id"] == "a":
            r.update(corrected_prompt="प्रश्न?", corrected_chosen="दिल्ली राजधानी है।",
                     corrected_rejected="मुंबई राजधानी है।", ok="y")
        elif r["lang"] == "en":
            r["ok"] = "y"
    with open(csvp, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    stats = review.import_review(items, csvp, "ekam")
    by = {i.id: i for i in items}
    assert by["a.hi"].status == "post_edited" and by["a.hi"].verified_by == "ekam"
    assert by["a.te"].status == "needs_translation"
    assert stats["finalized"] == 3
    with pytest.raises(ValueError):
        review.import_review(items, csvp, " ")


def test_import_rejects_ok_on_blank_translation(tmp_path):
    items = review.init_langs([mk("a")])
    csvp = tmp_path / "r.csv"
    review.export_review(items, csvp)
    import csv
    rows = list(csv.DictReader(open(csvp, encoding="utf-8-sig")))
    for r in rows:
        r["ok"] = "y"
    with open(csvp, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with pytest.raises(ValueError):
        review.import_review(items, csvp, "ekam")


# ---------------- analysis, hand-computed ----------------
def rows_from(acc_table, model="m"):
    """acc_table[(domain, lang)] = list of 0/1 per seed (same seed order across langs)."""
    rows = []
    for (d, l), v in acc_table.items():
        for i, c in enumerate(v):
            rows.append({"model": model, "domain": d, "lang": l, "seed_id": f"{d}{i}", "item_id": f"{d}{i}.{l}",
                         "error_type": "wrong_entity", "s_chosen": 1.0 if c else -1.0, "s_rejected": 0.0,
                         "ntok_chosen": 10, "ntok_rejected": 9})
    return rows


def test_effects_hand_computed():
    t = {("generic", "en"): [1, 1, 1, 1], ("generic", "hi"): [1, 1, 1, 0], ("generic", "te"): [1, 1, 0, 0],
         ("india", "en"): [1, 1, 1, 0], ("india", "hi"): [1, 0, 0, 0], ("india", "te"): [1, 1, 0, 0]}
    res = analysis.analyze_model(rows_from(t), "m", B=200)
    e = {k: v["est"] for k, v in res["effects"].items()}
    assert e["context_gap_en"] == pytest.approx(0.25)
    assert e["language_gap_generic[hi]"] == pytest.approx(0.25)
    assert e["language_gap_india[hi]"] == pytest.approx(0.5)
    assert e["interaction[hi]"] == pytest.approx(0.25)
    assert e["language_gap_generic[te]"] == pytest.approx(0.5)
    assert e["language_gap_india[te]"] == pytest.approx(0.25)
    assert e["interaction[te]"] == pytest.approx(-0.25)
    assert e["context_gap[hi]"] == pytest.approx(0.5)
    assert res["acc"]["india"]["hi"]["est"] == pytest.approx(0.25)


def test_ties_count_half():
    rows = rows_from({("generic", "en"): [1, 1], ("generic", "hi"): [1, 1], ("generic", "te"): [1, 1],
                      ("india", "en"): [1, 1], ("india", "hi"): [1, 1], ("india", "te"): [1, 1]})
    for r in rows:
        if r["domain"] == "india" and r["lang"] == "hi":
            r["s_chosen"] = r["s_rejected"]
    res = analysis.analyze_model(rows, "m", B=100)
    assert res["acc"]["india"]["hi"]["est"] == pytest.approx(0.5)


def test_incomplete_seed_dropped():
    t = {("generic", "en"): [1, 1], ("generic", "hi"): [1, 1], ("generic", "te"): [1],
         ("india", "en"): [1, 1], ("india", "hi"): [1, 1], ("india", "te"): [1, 1]}
    res = analysis.analyze_model(rows_from(t), "m", B=50)
    assert res["n_seeds"]["generic"] == 1


def test_bootstrap_ci_contains_estimate_and_shrinks_with_n():
    rng = np.random.default_rng(0)

    def table(n):
        return {(d, l): list((rng.random(n) < p).astype(int)) for (d, l), p in
                {("generic", "en"): .9, ("generic", "hi"): .8, ("generic", "te"): .75,
                 ("india", "en"): .85, ("india", "hi"): .7, ("india", "te"): .6}.items()}
    w = []
    for n in (40, 400):
        r = analysis.analyze_model(rows_from(table(n)), "m", B=500)["effects"]["interaction[hi]"]
        assert r["lo"] <= r["est"] <= r["hi"]
        w.append(r["hi"] - r["lo"])
    assert w[1] < w[0]


def test_length_baseline_and_inflation():
    rows = rows_from({("generic", "en"): [1], ("generic", "hi"): [1], ("generic", "te"): [1],
                      ("india", "en"): [1], ("india", "hi"): [1], ("india", "te"): [1]})
    for r in rows:
        if r["lang"] == "te":
            r["ntok_chosen"] = 30
    assert analysis.length_baseline(rows, "m")["india/en"] == 1.0
    assert analysis.token_inflation(rows, "m")["te"] == pytest.approx(3.0)


# ---------------- simulation: recovery and coverage ----------------
def test_simulation_recovers_planted_effect_and_covers():
    b = simulate.scenario(2.0, -0.4, -0.3, -0.5)
    r = simulate.coverage_and_power(200, 2.0, 1.0, b, n_sims=250, B=250, seed=1)
    assert abs(r["mean_estimate_pts"] - r["true_effect_pts"]) < 1.2   # unbiased
    assert 0.90 <= r["coverage"] <= 0.99                                # nominal 95%


def test_null_effect_false_positive_rate_controlled():
    b = simulate.scenario(2.0, -0.4, -0.3, 0.0)
    r = simulate.coverage_and_power(200, 2.0, 1.0, b, n_sims=250, B=250, seed=2)
    assert r["power"] < 0.12   # share of null studies whose CI excludes 0


# ---------------- scoring pipeline with mock ----------------
def test_mock_pipeline_end_to_end_recovers_planted_gap(tmp_path):
    items = []
    for d in ("india", "generic"):
        for i in range(150):
            items += full_seed(f"{d}{i}", d)
    q = {("generic", "en"): 2.0, ("generic", "hi"): 1.8, ("generic", "te"): 1.8,
         ("india", "en"): 2.0, ("india", "hi"): 0.8, ("india", "te"): 1.8}   # only India+Hindi is broken
    out = tmp_path / "s.jsonl"
    score_with_cache(MockRM(q, noise=1.0, seed=3), items, out)
    res = analysis.analyze_model(analysis.load_scores(out), "mock-rm", B=1000)
    e = res["effects"]
    assert e["interaction[hi]"]["lo"] > 0                      # detected
    assert e["interaction[te]"]["lo"] <= 0 <= e["interaction[te]"]["hi"]   # not falsely detected for te


def test_score_cache_resumes(tmp_path):
    items = full_seed("a") + full_seed("b")

    class Counting(MockRM):
        calls = 0

        def score_items(self, its):
            Counting.calls += len(its)
            return super().score_items(its)
    q = {(d, l): 1.0 for d in ("india", "generic") for l in LANGS}
    out = tmp_path / "s.jsonl"
    score_with_cache(Counting(q), items[:3], out)
    score_with_cache(Counting(q), items, out)
    assert Counting.calls == 6
    assert len(out.read_text().splitlines()) == 6


def test_markdown_and_plot(tmp_path):
    t = {(d, l): [1, 0, 1, 1] for d in ("india", "generic") for l in LANGS}
    res = [analysis.analyze_model(rows_from(t), "m", B=100)]
    assert "interaction[hi]" in analysis.to_markdown(res)
    analysis.plot_interaction(res, tmp_path / "p.png")
    assert (tmp_path / "p.png").stat().st_size > 1000


def test_margin_effects_present_and_signed():
    import numpy as np
    rng = np.random.default_rng(5)
    rows = []
    shift = {("generic", "en"): 1.5, ("generic", "hi"): 1.2, ("generic", "te"): 1.2,
             ("india", "en"): 1.5, ("india", "hi"): 0.4, ("india", "te"): 1.2}
    for (d, l), mu in shift.items():
        for i in range(120):
            rows.append({"model": "m", "domain": d, "lang": l, "seed_id": f"{d}{i}", "item_id": f"{d}{i}.{l}",
                         "error_type": "wrong_entity", "s_chosen": mu + rng.normal(), "s_rejected": 0.0,
                         "ntok_chosen": 10, "ntok_rejected": 9})
    res = analysis.analyze_model(rows, "m", B=400)
    me = res["margin_effects"]
    assert me["interaction[hi]"]["lo"] > 0
    assert me["interaction[te]"]["lo"] <= 0 <= me["interaction[te]"]["hi"]
    assert "Secondary" in analysis.to_markdown([res])


def test_margin_beats_accuracy_power_same_data():
    r = simulate.margin_vs_accuracy_power(60, n_sims=120, B=150, seed=1, lang_shift=-0.4, context_shift=-0.2,
                                          interaction_shift=-0.5)
    assert r["power_margin"] >= r["power_accuracy"]
