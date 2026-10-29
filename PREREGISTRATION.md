# Pre-registration (write the date and commit hash BEFORE any real model is scored)

Registered: ____-__-__   Commit: ________

## Question
Reward models (RMs) score worse on Hindi/Telugu. Is that because of the **language**, or because the
test content is **India-specific** (and RMs lack that knowledge in any language)?

## Design
2 x 3 factorial on minimal preference pairs. Domain {india, generic} x language {en, hi, te}.
Every seed exists in all three languages with identical content. English is fact-checked against a cited source;
Hindi/Telugu are machine translated and post-edited by a native speaker (see docs/annotation_protocol.md).
Unit of analysis: seed. Accuracy = share of seeds with score(chosen) > score(rejected); ties = 0.5.

## Hypotheses (directions are my prior, not a promise)
- H1 language gap: acc(generic, en) > acc(generic, L) for L in {hi, te}.
- H2 context gap: acc(generic, en) > acc(india, en).
- H3 interaction: the language gap is larger on India content than on generic content.
- H4 confound check: report (not test) token inflation per language and the length-only baseline.
- H5 exploratory: does the pattern change with model size within one family.

## Estimation
Cluster bootstrap over seeds (4000 resamples, independently within each domain), 95% percentile intervals.
No p-value thresholds. An interval containing 0 is reported as "not distinguishable from zero at this n",
never as "no effect". Effects are reported per model; pooling across models is only descriptive.
CO-PRIMARY analysis on the continuous margin s_chosen - s_rejected (standardised within model, effects in SD units),
because binary accuracy has low power (results/power_simulation.txt, results/margin_power.txt). Both are reported for
every model; neither is chosen after seeing results.

## Exclusions
Only seeds failing the annotation protocol (rejected by a reviewer, or an English fact that cannot be sourced).
Decided on the content, before looking at any model score. Pilot seeds used to tune difficulty are removed
from the main analysis.

## Stopping rule
Fixed sample: as many verified seeds per domain as the annotation budget allows, target >= 150 per domain. If only ~50 per domain are verified the paper is framed as a pilot: only large effects
(about 0.4 SD or more) are detectable, and null results are reported as inconclusive, not as evidence of no effect.
No peeking at model results to decide when to stop.

## Deviations
Any change after the registration date is listed here with the reason.
