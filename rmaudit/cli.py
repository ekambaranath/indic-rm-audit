"""Command line: python -m rmaudit <command> ..."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import analysis, review, simulate
from .schema import load_jsonl, save_jsonl, usable_items, validate
from .scoring import HFSequenceRM, MockRM, score_with_cache


def _cmd_validate(a):
    items = load_jsonl(a.data)
    errs, warns = validate(items, strict=a.strict)
    for w in warns[:10]:
        print("warn :", w)
    for e in errs:
        print("ERROR:", e)
    print(f"{len(items)} rows, {len(errs)} errors, {len(warns)} warnings (strict={a.strict}); showing first 10 warnings")
    return 1 if errs else 0


def _cmd_init_langs(a):
    items = review.init_langs(load_jsonl(a.data))
    save_jsonl(items, a.data)
    print(f"{len(items)} rows now in {a.data}")
    return 0


def _cmd_export(a):
    n = review.export_review(load_jsonl(a.data), a.out, tuple(a.status) if a.status else None)
    print(f"wrote {n} rows to {a.out}")
    return 0


def _cmd_import(a):
    items = load_jsonl(a.data)
    stats = review.import_review(items, a.csv, a.reviewer)
    save_jsonl(items, a.data)
    print(stats)
    return 0


def _cmd_score(a):
    items = usable_items(load_jsonl(a.data))
    if a.strict:
        errs, _ = validate(items, strict=True)
        if errs:
            print("\n".join(errs[:20]), file=sys.stderr)
            print("refusing to score: strict validation failed (use --no-strict only for pilots)", file=sys.stderr)
            return 1
    if a.mock:
        scorer = MockRM({(d, l): 1.0 for d in ("india", "generic") for l in ("en", "hi", "te")})
    else:
        scorer = HFSequenceRM.from_pretrained(a.model, dtype=a.dtype, batch_size=a.batch_size,
                                              load_in_4bit=a.load_in_4bit)
    rows = score_with_cache(scorer, items, a.out)
    print(f"{len(rows)} rows scored by {scorer.name} -> {a.out}")
    return 0


def _cmd_analyze(a):
    rows = analysis.load_scores(a.scores)
    models = sorted({r["model"] for r in rows})
    results = [analysis.analyze_model(rows, m, B=a.boot) for m in models]
    md = analysis.to_markdown(results)
    Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    (Path(a.out_dir) / "results.md").write_text(md, encoding="utf-8")
    (Path(a.out_dir) / "results.json").write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
    analysis.plot_interaction(results, Path(a.out_dir) / "interaction.png")
    print(md)
    return 0


def _cmd_power(a):
    rows = simulate.power_table(n_grid=tuple(a.n), n_sims=a.sims)
    print("interaction_logit  n_seeds  true_effect(pts)  coverage  power")
    for r in rows:
        print(f"{r['interaction_logit']:>17}  {r['n_seeds']:>7}  {r['true_effect_pts']:>16.1f}  "
              f"{r['coverage']:>8.2f}  {r['power']:>5.2f}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="rmaudit")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("validate"); s.add_argument("data"); s.add_argument("--strict", action="store_true")
    s.set_defaults(f=_cmd_validate)
    s = sub.add_parser("init-langs"); s.add_argument("data"); s.set_defaults(f=_cmd_init_langs)
    s = sub.add_parser("export-review"); s.add_argument("data"); s.add_argument("--out", required=True)
    s.add_argument("--status", nargs="*"); s.set_defaults(f=_cmd_export)
    s = sub.add_parser("import-review"); s.add_argument("data"); s.add_argument("csv")
    s.add_argument("--reviewer", required=True); s.set_defaults(f=_cmd_import)
    s = sub.add_parser("score"); s.add_argument("data"); s.add_argument("--model", default="")
    s.add_argument("--out", default="results/scores.jsonl"); s.add_argument("--dtype", default="float32")
    s.add_argument("--batch-size", type=int, default=8); s.add_argument("--load-in-4bit", action="store_true")
    s.add_argument("--mock", action="store_true", help="fake scorer, pipeline test only")
    s.add_argument("--no-strict", dest="strict", action="store_false", help="allow unverified data (pilots only)")
    s.set_defaults(f=_cmd_score, strict=True)
    s = sub.add_parser("analyze"); s.add_argument("scores"); s.add_argument("--out-dir", default="results")
    s.add_argument("--boot", type=int, default=4000); s.set_defaults(f=_cmd_analyze)
    s = sub.add_parser("power"); s.add_argument("--n", type=int, nargs="+", default=[50, 100, 150, 200, 300])
    s.add_argument("--sims", type=int, default=400); s.set_defaults(f=_cmd_power)

    a = p.parse_args(argv)
    return a.f(a)


if __name__ == "__main__":
    sys.exit(main())
