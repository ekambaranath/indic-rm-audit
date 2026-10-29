Colab (T4) cells. Each block is one cell; lines starting with ! are shell.

```
!git clone https://github.com/ekambaranath/indic-rm-audit.git
%cd indic-rm-audit
!pip -q install -r requirements.txt
!python -m pytest -q
```
```
# Skywork model IDs are unverified: if one fails to load, search Hugging Face for the exact name and fix it here.
MODELS = ["Skywork/Skywork-Reward-V2-Qwen3-0.6B", "Skywork/Skywork-Reward-V2-Qwen3-1.7B", "Skywork/Skywork-Reward-V2-Qwen3-4B"]
for m in MODELS:
    !python -m rmaudit score data/seeds.jsonl --model {m} --out results/scores.jsonl
```
```
!python -m rmaudit analyze results/scores.jsonl --out-dir results
```
For 8B, add `--load-in-4bit --dtype float16`.
