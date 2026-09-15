# M0 — First token (run log)

Box: pk31, RTX A6000 48GB, 16GB system RAM. Ollama 0.32.6. Driver 595.84.

Model tags used: `gemma3n:e4b`, `gemma3:12b` (the `gemma-4:*` tags in MISSIONS.md
do not exist in the Ollama registry).

## Measured

| | e4b | 12b @ 131k ctx | 12b @ 4k ctx |
|---|---|---|---|
| On-disk size | 7.5 GB | 8.1 GB | 8.1 GB |
| VRAM used (total, incl. 1.1GB desktop) | 4,814 MiB | 18,220 MiB | 10,160 MiB |
| VRAM attributable to model | ~3.7 GB | ~17.1 GB | ~9.1 GB |
| Generation rate (warm) | 103.4 tok/s | 69.1 tok/s | — |
| Prompt eval rate (warm) | 691.8 tok/s | — | — |
| Cold load duration | 36.5 s | 13.4 s | — |

## Findings

Cold vs warm is a disk-read effect, not a model effect: the first e4b run
generated at 4.89 tok/s and the identical second run at 103.4 tok/s. Only the
page cache changed between them.

KV cache dominates VRAM at long context. The same 12B weights occupy ~9 GB at a
4k context window and ~17 GB at 131k. The weights did not change; the per-token
attention cache did. Context length is therefore a VRAM decision, not just a
capability decision.

System RAM never spiked during load — Ollama mmaps the checkpoint from disk
rather than staging it through CPU memory, which is what makes an 8 GB
checkpoint loadable on a box with ~8 GB of RAM free.
