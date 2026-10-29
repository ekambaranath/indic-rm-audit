"""Spreadsheet round-trip for fact-checking and translation post-editing.

Workflow (works in Google Sheets or Excel; no custom tool needed):

    1. `init-langs`      adds blank hi/te rows (status needs_translation) for every English seed
    2. `export-review`   writes a CSV with the English reference columns next to each row
    3. you fill the CSV:   corrected_* = your text (required for translation rows),
                           source_url  = official source that confirms the English fact,
                           ok          = y (checked, final) or n (reject the whole seed)
    4. `import-review`   applies the edits and sets statuses, stamping --reviewer

Paste machine translation into corrected_* first if you like (set mt_system), then post-edit it.
"""

from __future__ import annotations

import csv
from pathlib import Path

from .schema import LANGS, Item, make_id

COLS = ["id", "seed_id", "domain", "lang", "error_type", "en_prompt", "en_chosen", "en_rejected",
        "prompt", "chosen", "rejected", "status", "source_url", "mt_system", "notes", "ok",
        "corrected_prompt", "corrected_chosen", "corrected_rejected"]


def init_langs(items: list[Item], langs: tuple[str, ...] = LANGS) -> list[Item]:
    """Add a blank row for every (seed, lang) that does not exist yet."""
    have = {i.id for i in items}
    out = list(items)
    for it in [i for i in items if i.lang == "en"]:
        for lang in langs:
            if lang == "en" or make_id(it.seed_id, lang) in have:
                continue
            out.append(Item(id=make_id(it.seed_id, lang), seed_id=it.seed_id, domain=it.domain, lang=lang,
                            error_type=it.error_type, prompt="", chosen="", rejected="",
                            status="needs_translation"))
    return out


def export_review(items: list[Item], path: str | Path, only_status: tuple[str, ...] | None = None) -> int:
    en = {i.seed_id: i for i in items if i.lang == "en"}
    rows = []
    for it in items:
        if only_status and it.status not in only_status:
            continue
        ref = en.get(it.seed_id)
        rows.append({
            "id": it.id, "seed_id": it.seed_id, "domain": it.domain, "lang": it.lang,
            "error_type": it.error_type,
            "en_prompt": ref.prompt if ref else "", "en_chosen": ref.chosen if ref else "",
            "en_rejected": ref.rejected if ref else "",
            "prompt": it.prompt, "chosen": it.chosen, "rejected": it.rejected, "status": it.status,
            "source_url": it.source_url, "mt_system": it.mt_system, "notes": it.notes, "ok": "",
            "corrected_prompt": "", "corrected_chosen": "", "corrected_rejected": "",
        })
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:  # utf-8-sig so Excel shows Telugu correctly
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def import_review(items: list[Item], path: str | Path, reviewer: str) -> dict:
    """Apply a filled review CSV. Returns counts. Text edits apply even when `ok` is blank."""
    if not reviewer.strip():
        raise ValueError("--reviewer is required so every checked item is attributable")
    by_id = {i.id: i for i in items}
    stats = {"edited": 0, "finalized": 0, "rejected": 0, "unknown_id": 0}
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            it = by_id.get(row["id"])
            if it is None:
                stats["unknown_id"] += 1
                continue
            edited = False
            for fld in ("prompt", "chosen", "rejected"):
                new = (row.get(f"corrected_{fld}") or "").strip()
                if new and new != getattr(it, fld):
                    setattr(it, fld, new)
                    edited = True
            for fld in ("source_url", "mt_system", "notes"):
                new = (row.get(fld) or "").strip()
                if new != getattr(it, fld):
                    setattr(it, fld, new)
            ok = (row.get("ok") or "").strip().lower()
            if ok in ("y", "yes"):
                if it.lang != "en" and not (it.prompt and it.chosen and it.rejected):
                    raise ValueError(f"{it.id}: marked ok but translation is incomplete")
                it.status = "verified" if it.lang == "en" else "post_edited"
                it.verified_by = reviewer
                stats["finalized"] += 1
            elif ok in ("n", "no"):
                it.status = "rejected"
                it.verified_by = reviewer
                stats["rejected"] += 1
            elif edited and it.status == "needs_translation":
                it.status = "machine_translated"
            if edited:
                stats["edited"] += 1
    return stats
