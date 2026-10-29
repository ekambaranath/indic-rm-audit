"""Data schema and validation for the 2 x 3 reward-model audit.

Design: every *seed* is one minimal preference pair (a prompt, a correct "chosen" answer, and a
"rejected" answer that differs by one specific, checkable error). Each seed exists in three
languages with the SAME content, so differences between languages cannot come from different content.

    domain  : "india"   India-specific facts (law, institutions, culture, geography)
              "generic" matched control facts with no India-specific knowledge
    lang    : "en" | "hi" | "te"

Statuses (a seed is only used in the main analysis when every language of it is usable):
    needs_translation  blank row waiting for a translation
    draft_unverified   written/drafted but not yet fact-checked
    machine_translated machine translation, not yet checked by a native speaker
    verified           English item whose fact was checked against a cited source
    post_edited        translation checked and corrected by a native speaker
    rejected           reviewer rejected it; the whole seed is dropped
"""

from __future__ import annotations

import difflib
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

LANGS = ("en", "hi", "te")
DOMAINS = ("india", "generic")
STATUSES = ("needs_translation", "draft_unverified", "machine_translated", "verified", "post_edited", "rejected")
ERROR_TYPES = ("wrong_entity", "wrong_number", "wrong_date", "wrong_procedure", "wrong_attribution",
               "wrong_location")
FINAL_STATUS = {"en": ("verified",), "hi": ("post_edited", "verified"), "te": ("post_edited", "verified")}


@dataclass
class Item:
    id: str
    seed_id: str
    domain: str
    lang: str
    error_type: str
    prompt: str
    chosen: str
    rejected: str
    status: str = "draft_unverified"
    source_url: str = ""
    verified_by: str = ""
    mt_system: str = ""
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def make_id(seed_id: str, lang: str) -> str:
    return f"{seed_id}.{lang}"


def load_jsonl(path: str | Path) -> list[Item]:
    items = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            items.append(Item(**json.loads(line)))
    return items


def save_jsonl(items: list[Item], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it.to_dict(), ensure_ascii=False) + "\n")


def edit_ratio(a: str, b: str) -> float:
    """1.0 = identical, 0.0 = nothing in common (character level)."""
    return difflib.SequenceMatcher(None, a, b).ratio()


def validate(items: list[Item], strict: bool = False) -> tuple[list[str], list[str]]:
    """Return (errors, warnings).

    strict=True is the gate for the main analysis: every seed must exist in all three languages, English must
    be `verified` (fact checked) and hi/te must be `post_edited` or `verified` (checked by a native speaker).
    """
    errors: list[str] = []
    warns: list[str] = []
    seen: set[str] = set()
    by_seed: dict[str, dict[str, Item]] = defaultdict(dict)

    for it in items:
        if it.id in seen:
            errors.append(f"{it.id}: duplicate id")
        seen.add(it.id)
        if it.id != make_id(it.seed_id, it.lang):
            errors.append(f"{it.id}: id must be '<seed_id>.<lang>'")
        if it.lang not in LANGS:
            errors.append(f"{it.id}: lang {it.lang!r} not in {LANGS}")
        if it.domain not in DOMAINS:
            errors.append(f"{it.id}: domain {it.domain!r} not in {DOMAINS}")
        if it.status not in STATUSES:
            errors.append(f"{it.id}: status {it.status!r} not in {STATUSES}")
        if it.error_type not in ERROR_TYPES:
            errors.append(f"{it.id}: error_type {it.error_type!r} not in {ERROR_TYPES}")
        blank_ok = it.status in ("needs_translation", "rejected")
        for fld in ("prompt", "chosen", "rejected"):
            if not getattr(it, fld).strip() and not blank_ok:
                errors.append(f"{it.id}: empty {fld}")
        if it.chosen.strip() and it.chosen.strip() == it.rejected.strip():
            errors.append(f"{it.id}: chosen and rejected are identical")
        if it.chosen.strip() and it.rejected.strip():
            if edit_ratio(it.chosen, it.rejected) < 0.5:
                warns.append(f"{it.id}: chosen/rejected differ a lot (edit ratio < 0.5); minimal pairs avoid "
                             "lexical shortcuts")
            lc, lr = len(it.chosen), len(it.rejected)
            if abs(lc - lr) / max(lc, lr) > 0.3:
                warns.append(f"{it.id}: chosen/rejected lengths differ by more than 30%; reward models have a "
                             "length bias")
        by_seed[it.seed_id][it.lang] = it

    for seed, langs in by_seed.items():
        domains = {i.domain for i in langs.values()}
        types = {i.error_type for i in langs.values()}
        if len(domains) > 1:
            errors.append(f"{seed}: domain differs across languages")
        if len(types) > 1:
            errors.append(f"{seed}: error_type differs across languages")
        if strict:
            missing = [l for l in LANGS if l not in langs]
            if missing:
                errors.append(f"{seed}: missing languages {missing}")
            for lang, it in langs.items():
                if it.status == "rejected":
                    errors.append(f"{seed}: contains a rejected item ({it.id}); drop the whole seed")
                elif it.status not in FINAL_STATUS[lang]:
                    errors.append(f"{it.id}: status {it.status!r} is not final for {lang} "
                                  f"(need one of {FINAL_STATUS[lang]})")
            if "en" in langs and not langs["en"].source_url.strip():
                errors.append(f"{seed}: English item has no source_url")
    return errors, warns


def usable_items(items: list[Item]) -> list[Item]:
    """Items from seeds that exist in all three languages with no rejected member."""
    by_seed: dict[str, dict[str, Item]] = defaultdict(dict)
    for it in items:
        by_seed[it.seed_id][it.lang] = it
    keep = {s for s, d in by_seed.items()
            if all(l in d for l in LANGS) and all(d[l].status != "rejected" for l in LANGS)
            and all(d[l].chosen.strip() and d[l].rejected.strip() and d[l].prompt.strip() for l in LANGS)}
    return [it for it in items if it.seed_id in keep]
