# Annotation protocol

**English seed (domain india or generic)**
1. One factual question with a single checkable answer. India items need India-specific knowledge; generic items must not.
2. `chosen` is correct; `rejected` is identical except for one error of the stated `error_type`.
3. Keep chosen/rejected within ~30% length of each other (reward models are length biased).
4. Check the fact in an authoritative source (government/statute text, official site, encyclopaedia citing primary sources). Paste the URL in `source_url`. Unverifiable -> `ok = n`.
5. Balance error types across both domains so domain is not confounded with error type.

**Hindi / Telugu**
1. Paste machine translation into `corrected_*` (set `mt_system`). Keep named entities, numbers and dates unchanged.
2. A native speaker post-edits for correctness and natural register, keeping the content the same as English. Use standard script, no English mixing unless the English term is normal usage.
3. Check the translated `rejected` still contains exactly the same error and nothing extra.
4. `ok = y` stamps `status=post_edited` and `verified_by`.

**Agreement check**
A second reviewer independently re-checks about 20% of items (random). Report percent agreement and Cohen's kappa
(use the Concord library). Disagreements are resolved by discussion; the rate goes in the README.

**Round trip**
`python -m rmaudit init-langs data/seeds.jsonl` -> `export-review` -> fill the CSV in Google Sheets -> `import-review --reviewer <name>` -> `validate --strict`.
