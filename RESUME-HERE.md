# Resume here

Paused 2026-08-07. M0–M7 all built, run and measured. Nothing is committed —
the work is uncommitted on `main` and survives a reboot on disk.

## Restart the environment

```bash
cd ~/zexp1/aitrain1
systemctl status ollama          # installed as a service, starts on boot
ollama list                      # gemma3:12b, gemma3n:e4b, embeddinggemma
```

Nothing needs rebuilding. `zlearn.db`, `vectors.npy` and the trained adapter are
all on disk. Sanity check in one command:

```bash
PYTHONPATH=src .venv/bin/python src/rag.py --grade 11 "What is an empty set?"
```

## Open items, in priority order

1. **Drop the real corpus into `corpus/`** (from Backblaze), then rerun
   `inventory.py` -> `chunk.py` -> `embed.py`. Expect the extraction rate to
   fall well below the current 100%, which only looks good because the 13
   placeholder files were already-cleaned markdown.
2. **Replace `eval/golden.jsonl`** with 50 hand-written questions. The current
   33 were machine-authored, which the governing doc forbids for a real
   evaluation.
3. **Fork `zexp1/zsm1`** — blocked on which of the 30 orgs is the destination.
4. **Commit** — not done, was never asked for.
5. **The operator's learning pass.** `docs/01-references1/CONCEPT-LEDGER.md` is
   still empty and must be filled in his own words, not Claude's.

## What was measured

See `docs/02-runs/`. Headline: recall@1 97%, judge mean 4.52/5, grounded 33/33,
QLoRA adapter 1.32 -> 2.65 of 5 on CBSE format adherence.

## Environment notes

Beyond the documented stack, this box needed `python3-venv`, `build-essential`
and `python3-dev` installed via apt — bitsandbytes JIT-compiles CUDA kernels
through Triton and fails without a C compiler and Python headers. That cost two
failed training runs to diagnose.

Python here is 3.14; torch 2.13.0+cu130 works on it. `google/gemma-3-1b-it` is
gated on Hugging Face and returns 401 without a licence acceptance —
`unsloth/gemma-3-1b-it` is a public mirror of the same weights and was used.
