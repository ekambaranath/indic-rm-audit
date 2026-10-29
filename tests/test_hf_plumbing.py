"""Offline check of the real HF code path with a tiny random Qwen2 reward model (no downloads)."""
import pytest

torch = pytest.importorskip("torch")
tokenizers = pytest.importorskip("tokenizers")
transformers = pytest.importorskip("transformers")

from rmaudit.schema import Item
from rmaudit.scoring import HFSequenceRM

TEMPLATE = ("{% for m in messages %}<|{{ m['role'] }}|>{{ m['content'] }}<|end|>{% endfor %}")


@pytest.fixture(scope="module")
def rm():
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForSequenceClassification
    tk = Tokenizer(models.BPE(unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    corpus = ["The capital of India is New Delhi.", "भारत की राजधानी नई दिल्ली है।", "భారతదేశ రాజధాని న్యూఢిల్లీ."] * 20
    tk.train_from_iterator(corpus, trainers.BpeTrainer(vocab_size=400, special_tokens=["<unk>", "<pad>", "<eos>", "<|user|>", "<|assistant|>", "<|end|>"],
                                                       initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="<unk>", pad_token="<pad>", eos_token="<eos>")
    tok.chat_template = TEMPLATE
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=2, num_key_value_heads=2, num_labels=1, pad_token_id=tok.pad_token_id)
    return HFSequenceRM(Qwen2ForSequenceClassification(cfg), tok, "tiny", batch_size=1)


def items():
    t = ["The capital of India is New Delhi.", "भारत की राजधानी नई दिल्ली है।", "భారతదేశ రాజధాని న్యూఢిల్లీ."]
    return [Item(id=f"s{i}.en", seed_id=f"s{i}", domain="india", lang="en", error_type="wrong_entity",
                 prompt="Q?" * (i + 1), chosen=t[i % 3], rejected=t[(i + 1) % 3] + " extra" * i) for i in range(5)]


def test_scores_finite_and_batch_invariant(rm):
    its = items()
    a = rm.score_items(its)
    rm.bs = 3
    b = rm.score_items(its)
    rm.bs = 1
    for x, y in zip(a, b):
        assert abs(x["s_chosen"] - y["s_chosen"]) < 1e-4      # padding must not change a score
        assert abs(x["s_rejected"] - y["s_rejected"]) < 1e-4
        assert x["ntok_chosen"] > 0


def test_nan_guard(rm, monkeypatch):
    its = items()[:1]
    orig = rm.model.forward

    def bad(*a, **k):
        o = orig(*a, **k)
        o.logits = o.logits * float("nan")
        return o
    monkeypatch.setattr(rm.model, "forward", bad)
    with pytest.raises(FloatingPointError):
        rm.score_items(its)
