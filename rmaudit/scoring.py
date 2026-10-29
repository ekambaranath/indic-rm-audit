"""Reward-model scorers. A scorer turns Items into per-item score rows.

Scalar (Bradley-Terry style) reward models score each response independently, so there is no position bias
to randomize away: we compare score(chosen) with score(rejected).
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Protocol

from .schema import Item


class Scorer(Protocol):
    name: str

    def score_items(self, items: list[Item]) -> list[dict]: ...


def _row(it: Item, model: str, sc: float, sr: float, tc: int, tr: int) -> dict:
    return {"item_id": it.id, "seed_id": it.seed_id, "domain": it.domain, "lang": it.lang,
            "error_type": it.error_type, "model": model, "s_chosen": sc, "s_rejected": sr,
            "ntok_chosen": tc, "ntok_rejected": tr, "nchar_chosen": len(it.chosen),
            "nchar_rejected": len(it.rejected)}


class HFSequenceRM:
    """A Hugging Face reward model exposed as AutoModelForSequenceClassification with one label.

    Tested with a tiny random model offline; real checkpoints (e.g. Skywork-Reward-V2 family) are the user's
    first GPU run. If a score is not finite the run stops with advice instead of silently producing garbage
    (this is how fp16 overflow shows up on some models).
    """

    def __init__(self, model, tokenizer, name: str, device: str = "cpu", batch_size: int = 8):
        self.model, self.tok, self.name, self.device, self.bs = model, tokenizer, name, device, batch_size
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        if getattr(self.model.config, "pad_token_id", None) is None:
            self.model.config.pad_token_id = self.tok.pad_token_id
        self.model.eval()

    @classmethod
    def from_pretrained(cls, model_id: str, dtype: str = "float32", device: str | None = None,
                        batch_size: int = 8, load_in_4bit: bool = False):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        kw: dict = {"num_labels": 1}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
            kw["device_map"] = {"": 0}
        else:
            kw["dtype"] = getattr(torch, dtype)
        try:
            model = AutoModelForSequenceClassification.from_pretrained(model_id, **kw)
        except TypeError:  # older transformers used torch_dtype
            kw["torch_dtype"] = kw.pop("dtype", None)
            model = AutoModelForSequenceClassification.from_pretrained(model_id, **kw)
        if not load_in_4bit:
            model.to(device)
        tok = AutoTokenizer.from_pretrained(model_id)
        return cls(model, tok, model_id, device, batch_size)

    def _texts(self, prompt: str, response: str) -> str:
        conv = [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}]
        return self.tok.apply_chat_template(conv, tokenize=False)

    def _score(self, texts: list[str]) -> tuple[list[float], list[int]]:
        import torch

        out, counts = [], []
        for i in range(0, len(texts), self.bs):
            chunk = texts[i:i + self.bs]
            enc = self.tok(chunk, return_tensors="pt", padding=True, add_special_tokens=False)
            counts += [int(m.sum()) for m in enc["attention_mask"]]
            enc = {k: v.to(self.model.device) for k, v in enc.items()}
            with torch.inference_mode():
                logits = self.model(**enc).logits
            vals = logits.float().reshape(len(chunk), -1)[:, 0].cpu().tolist()
            for v in vals:
                if not math.isfinite(v):
                    raise FloatingPointError(
                        "non-finite reward score. Re-run with --dtype float32 (fp16 can overflow on some "
                        "models) and report it if it persists.")
            out += vals
        return out, counts

    def score_items(self, items: list[Item]) -> list[dict]:
        tc = [self._texts(it.prompt, it.chosen) for it in items]
        tr = [self._texts(it.prompt, it.rejected) for it in items]
        sc, nc = self._score(tc)
        sr, nr = self._score(tr)
        return [_row(it, self.name, a, b, c, d) for it, a, b, c, d in zip(items, sc, sr, nc, nr)]


class MockRM:
    """Deterministic fake reward model with *known* language/context effects, for tests and the pipeline demo.

    quality[(domain, lang)] is the average margin by which the chosen answer outscores the rejected one;
    noise is Gaussian per response. Never report results from this class as findings.
    """

    def __init__(self, quality: dict[tuple[str, str], float], noise: float = 1.0, seed: int = 0,
                 name: str = "mock-rm"):
        self.quality, self.noise, self.seed, self.name = quality, noise, seed, name

    def _gauss(self, key: str) -> float:
        h = hashlib.sha256(f"{self.seed}|{key}".encode()).digest()
        u1 = (int.from_bytes(h[:8], "big") + 1) / (2 ** 64 + 1)
        u2 = (int.from_bytes(h[8:16], "big") + 1) / (2 ** 64 + 1)
        return math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)

    def score_items(self, items: list[Item]) -> list[dict]:
        rows = []
        for it in items:
            q = self.quality[(it.domain, it.lang)]
            sc = q + self.noise * self._gauss(it.seed_id + "|c|" + it.lang)
            sr = self.noise * self._gauss(it.seed_id + "|r|" + it.lang)
            rows.append(_row(it, self.name, sc, sr, len(it.chosen.split()), len(it.rejected.split())))
        return rows


def score_with_cache(scorer: Scorer, items: list[Item], out_path: str | Path) -> list[dict]:
    """Score items, appending to a JSONL cache so an interrupted Colab session can resume."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done: dict[str, dict] = {}
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if r["model"] == scorer.name:
                    done[r["item_id"]] = r
    todo = [it for it in items if it.id not in done]
    for i in range(0, len(todo), 64):
        rows = scorer.score_items(todo[i:i + 64])
        with open(out_path, "a", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                done[r["item_id"]] = r
    return [done[it.id] for it in items]
