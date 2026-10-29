# indic-rm-audit: Language or Context?

**Question.** Reward models do worse on Hindi and Telugu. Is that because of the *language*, or because the test
content is *India-specific* and the model never learned it, in any language?

Existing work measures the first half (multilingual accuracy drops). Nobody I found separates it from the second.
This project does, with a 2 x 3 factorial design on minimal preference pairs.

|            | English | Hindi | Telugu |
|------------|---------|-------|--------|
| **generic** facts (control) | same content | same content | same content |
| **India-specific** facts    | same content | same content | same content |

Each seed is one (prompt, chosen, rejected) triple where `rejected` differs by one checkable error. The same seed
exists in all three languages, so a difference between languages cannot come from different content.

- language gap = acc(en) - acc(L) within a domain
- context gap = acc(generic) - acc(india) within a language
- interaction = language gap on India content minus language gap on generic content

## Status (honest)
- **Done and tested:** schema + validation, spreadsheet review round trip, scorer for HF scalar reward models,
  cluster-bootstrap analysis, simulation, CLI, 18 tests (hand-computed effects, ties, CI coverage, end-to-end
  recovery of a planted effect, a tiny-random-model check of the HF code path including batch invariance and a NaN guard).
- **Not done:** no real model has been scored and no translated data exists. `data/pilot_seeds_en.jsonl` holds 32
  *draft* English seeds written by an AI and unverified. **There are no findings in this repo yet.**
- Model IDs in `notebooks/colab_run.md` were not verified; the scorer fails loudly if one is wrong.

## What the simulation says (results/power_simulation.txt)
CI coverage is about 95%, so the intervals are honest. But power is low: with one model, accuracy near 85% and
200 seeds per domain, a 9.5-point interaction is detected only about 43% of the time, and a 5-point language gap
about 37%. So the plan is: at least 150 seeds per domain (more is better), several models, report effect sizes with
intervals rather than significance, and use the continuous score margin as a more powerful secondary analysis.

## Margin analysis (added)
The score margin (chosen minus rejected, in SD units) uses information that 0/1 accuracy discards. On identical
simulated data it roughly doubles power (results/margin_power.txt). At 50 questions per domain only large effects
(about 0.4 SD) are detectable (power ~0.7-0.8); at 150 per domain even modest effects become detectable.

## Related work (non-exhaustive search, Oct 2026)
HelpSteer3's multilingual subset covers 13 languages and no Indian language. M-RewardBench machine-translates
prompts, includes Hindi but not Telugu, and does not test cultural preference. RELIC is machine translated with
Hindi/Telugu only as auxiliary. Samiksha studies LLM-judge agreement, not reward models. HelpSteer2 is English with US annotators.

## Workflow
```
pip install -r requirements.txt && python -m pytest -q
python -m rmaudit init-langs data/seeds.jsonl
python -m rmaudit export-review data/seeds.jsonl --out review/r1.csv   # fill in Google Sheets
python -m rmaudit import-review data/seeds.jsonl review/r1.csv --reviewer "<name>"
python -m rmaudit validate data/seeds.jsonl --strict
python -m rmaudit score data/seeds.jsonl --model <hf-id> --out results/scores.jsonl   # GPU
python -m rmaudit analyze results/scores.jsonl --out-dir results
```
`score` refuses unverified data unless you pass `--no-strict` (pilots only). See `PREREGISTRATION.md`
(fill it in before scoring) and `docs/annotation_protocol.md`.

## Threats to validity
Translation quality; Hindi/Telugu tokenise to more tokens (token inflation is reported, plus a length-only
baseline); "India-specific" vs "generic" difficulty may differ for reasons other than context (error types are
balanced and a difficulty check is done on pilot data only); factual-error pairs are one slice of preference,
not helpfulness or safety; scalar RMs only.
