# aitrain1 — zlearn ladder, local AI foundations for eduZ

A learning build: hand-written RAG and fine-tuning on one box, no frameworks.
Governing rules are in `docs/01-references1/CLAUDE.md`; the ladder is
`docs/01-references1/MISSIONS.md`.

Everything here was written without LangChain, LlamaIndex, or a vector database.
Retrieval is a numpy dot product in `src/search.py` — about three lines of the
actual algorithm.

## Hardware it is designed against

RTX A6000, 48GB VRAM — comfortable. **16GB system RAM — the binding
constraint.** Serving goes through Ollama because it memory-maps checkpoints
from disk; a naive `transformers` load would stage the full checkpoint through
CPU RAM and OOM the box even though the model fits in VRAM.

The A6000 is Ampere: bf16 and INT8 are native, FP8 is not and must never be
requested.

## Setup

```bash
python3 -m venv .venv                    # needs python3-venv installed
.venv/bin/pip install pymupdf numpy httpx
ollama pull gemma3:12b
ollama pull embeddinggemma
```

## The pipeline, end to end

```bash
.venv/bin/python src/inventory.py corpus/                    # M2 census
.venv/bin/python src/chunk.py --target-chars 2000 --overlap-chars 200   # M3
PYTHONPATH=src .venv/bin/python src/embed.py                 # M4 vectors
PYTHONPATH=src .venv/bin/python src/search.py "what is sociology"       # M4
PYTHONPATH=src .venv/bin/python src/rag.py --grade 11 "..."  # M5 cited answer
PYTHONPATH=src .venv/bin/python src/eval.py                  # M6 scorecard
PYTHONPATH=src .venv/bin/python src/eval_retrieval.py        # M6 fast sweep
```

## Files

| File | Mission | What it does |
|---|---|---|
| `src/ollama.py` | — | the entire Ollama HTTP surface we use |
| `src/ask.py` | M1 | grade-aware question answering, no retrieval |
| `src/exp_m1.py` | M1 | token cost by language; temperature sweep |
| `src/inventory.py` | M2 | corpus census into SQLite, with provenance |
| `src/chunk.py` | M3 | structure-aware chunking with overlap |
| `src/embed.py` | M4 | embed chunks, store normalised vectors |
| `src/search.py` | M4 | brute-force cosine search |
| `src/rag.py` | M5 | retrieval + generation with citations |
| `src/eval.py` | M6 | recall@k and LLM-as-judge scorecard |
| `src/eval_retrieval.py` | M6 | retrieval-only sweep, seconds not minutes |
| `src/make_sft_data.py` | M7 | build answer-format training examples |
| `src/train_qlora.py` | M7 | QLoRA adapter training |
| `src/eval_adapter.py` | M7 | adapter vs frozen base, format score |

## Status

**M0–M7 all built, run and measured.** Numbers in `docs/02-runs/`.

Headline numbers: recall@1 97%, judge mean 4.52/5, grounded 33/33. The QLoRA
adapter doubled CBSE format adherence over the frozen base (1.32 -> 2.65 of 5)
by training 1.2% of parameters for 157 seconds.

Extra system packages this needed beyond the documented stack:
`python3-venv`, `build-essential` and `python3-dev` (bitsandbytes JIT-compiles
CUDA kernels via Triton and needs a C compiler plus Python headers).

**The corpus is a placeholder.** 13 NCERT sample chapters pulled from
`zexp1/zsm1`, already converted to clean markdown. The real corpus lives in
Backblaze and has not landed yet. Every extraction number here is therefore
optimistic — see `docs/02-runs/M2-M6-pipeline.md`.

**The golden set is machine-authored.** `eval/golden.jsonl` was written by
Claude, not by the operator, which the governing doc explicitly forbids for a
real evaluation. It works as a smoke test; it must be replaced by 50
hand-written questions before any score from it is trusted.
