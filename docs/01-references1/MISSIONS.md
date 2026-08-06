# MISSIONS — zero to competent, in order

Eight missions. Each is one Claude Code session. `/clear` between them.

Each mission has an **exit gate**. Do not advance past an unmet gate — the later missions assume the earlier concepts are actually held, not merely passed through.

Estimated: M0–M2 in day one, M3–M5 in day two, M6 in day three, M7 when M6's scorecard runs.

---

## M0 — First token

**Objective:** get a model answering on the box, and understand what just happened.

**Build:** nothing. Install Ollama, pull `gemma-4:e4b` and `gemma-4:12b`, run both.

**Concepts to explain before/while doing:**
- What a model file actually contains (weights, config, tokenizer) and why it is one big blob
- Quantisation tags — what `q4_K_M` vs `q8_0` vs `bf16` mean in bytes and in quality
- The VRAM arithmetic: parameters x bytes-per-parameter + KV cache. Do the sum out loud for both models.
- Why the 48GB card can hold these easily but the 16GB of system RAM would have choked a naive loader

**Experiment:** run `nvidia-smi` before, during, and after. Note VRAM used against the arithmetic prediction. They should roughly match — if they do not, work out why.

**Exit gate:** operator can state, without looking, roughly how much VRAM a 12B model at 8-bit needs, and why.

---

## M1 — Talking to it in code

**Objective:** replace the chat box with Python, and meet the knobs.

**Build:** `ask.py` — takes a question and a grade level on the command line, returns a grade-appropriate answer. Hits the Ollama HTTP API directly with `httpx`. No client library.

**Concepts:**
- System prompt vs user prompt — what the model actually sees
- Tokens: run the same sentence in English and Hindi, compare token counts. This matters commercially and he should see it early.
- Temperature, top_p — what they do to the sampling distribution
- Context window: what happens as it fills
- Structured output: asking for strict JSON and getting it

**Experiment:** ask one CBSE question five times at `temperature=0`, then five times at `temperature=1.0`. Print all ten. Then ask the same question with `grade=4` and `grade=11` system prompts.

**Exit gate:** operator can predict, before running, whether a given task wants temperature 0 or higher, and say why.

---

## M2 — The corpus, honestly

**Objective:** find out what he actually has. This is the unglamorous mission and the most valuable one.

**Build:** `inventory.py` — walks the CBSE download directory, writes a SQLite table with one row per file: path, size, page count, detected grade, detected subject, extractable-text yes/no, first 200 characters of extracted text.

**Concepts:**
- Text-layer PDFs vs scanned-image PDFs, and why the difference decides everything downstream
- Extraction is not availability — a file can exist, open fine, and yield nothing
- OCR is a separate and much more expensive problem. Scope it, do not solve it yet.
- Provenance: why grade and subject must be captured now, not reconstructed later

**Experiment:** print the honest summary — total files, how many yield text, how many are image-only, how many have no determinable grade. Expect the number to be worse than he hopes.

**Exit gate:** a manifest table exists, and he can say out loud "I have N usable chapters covering X, and M files that need OCR before they are worth anything."

---

## M3 — Chunking

**Objective:** turn documents into retrievable units without destroying meaning.

**Build:** `chunk.py` — reads usable files from the manifest, splits into chunks, writes to SQLite with full provenance columns.

**Concepts:**
- Why you cannot retrieve a whole textbook — context windows, cost, and signal dilution
- Chunk size as a tradeoff: too small loses context, too large drowns the answer in noise
- Overlap, and what it buys
- Why splitting on paragraph or heading boundaries beats splitting on character count
- What breaks: tables, equations, diagrams-with-captions, multi-column layouts

**Experiment:** chunk one chapter three ways — ~200, ~500, ~1000 tokens. Print three chunks from each that would be needed to answer a real question. Look at them. Form an opinion.

**Exit gate:** chunks in SQLite with provenance, and he has a stated preference on chunk size with a reason attached.

---

## M4 — Embeddings and search, no LLM

**Objective:** semantic retrieval, built by hand, with no generation involved.

**Build:** `embed.py` (embed all chunks via Ollama's embedding endpoint, store vectors in a `.npy` file) and `search.py` (brute-force cosine similarity in numpy, return top-k with provenance).

**Concepts:**
- What an embedding is: text to a fixed-length vector where nearness means relatedness
- Cosine similarity — write the formula out, it is three lines of numpy
- Why brute force is completely fine at this scale, and precisely what a vector database would add (at a few thousand chunks: nothing)
- Why this mission has no LLM in it — retrieval quality is measurable on its own, and it caps everything downstream

**Experiment:** search `"photosynthesis"`, then `"how do plants make their food"`, then `"पौधे भोजन कैसे बनाते हैं"`. Compare results. Then search something absent from the corpus and look at what comes back anyway — nearest-neighbour always returns *something*.

**Exit gate:** top-5 returns in well under a second, and he can explain why the plain-language query found the right chapter without sharing a keyword with it.

---

## M5 — RAG

**Objective:** wire M4 into M1, with citations.

**Build:** `rag.py` — retrieve top-k chunks for a question, put them in the system prompt with source labels, generate an answer that cites file and page.

**Concepts:**
- Context stuffing, and the budget it consumes
- Citation discipline — making the model point at the chunk it used
- Hallucination: what it looks like, and why retrieval reduces but never eliminates it
- The hard ceiling: generation quality cannot exceed retrieval quality

**Experiment — the important one:** ask a question whose answer is definitively *not* in the corpus. Watch what the model does with five irrelevant retrieved chunks. Then add an instruction permitting it to say the corpus does not cover this and answer from general knowledge, flagged as such. Compare. This experiment is the whole beyond-syllabus benchmark in miniature.

**Exit gate:** answers carry citations, and unsupported answers are visibly flagged rather than silently fabricated.

---

## M6 — Evaluation

**Objective:** stop guessing. This mission is the actual professional skill.

**Build:** a 50-question golden set (he writes the questions, from real CBSE material, across at least three grades and two subjects, including some beyond-syllabus and some in Hindi), plus `eval.py` producing a scorecard.

**Concepts:**
- Retrieval metrics vs generation metrics — recall@5 on retrieval, quality on generation, measured separately because they fail differently
- LLM-as-judge: using the 12B to grade answers against references, and where that judge is biased
- Why 50 hand-written questions beat 5,000 generated ones
- The loop: change one variable, re-run, read the number

**Experiment:** run the eval. Then change chunk size from M3's choice to a different one, re-embed, re-run. Watch the score move. This is the moment the whole thing stops being magic.

**Exit gate:** one command prints a scorecard, and he has changed exactly one variable and observed the effect.

---

## M7 — First fine-tune (gated)

**Do not start until M6's scorecard runs.**

**Objective:** change model *behaviour*, not model *knowledge*.

**Build:** a QLoRA adapter on a small Gemma 4 variant, trained on ~500 curated examples of CBSE-style answer formatting — step structure, marks-appropriate length, exam register.

**Concepts:**
- What LoRA actually modifies, and why the base weights stay frozen
- **Why format and not facts** — syllabus content changes yearly and varies by board; it belongs in retrieval, never in weights
- QLoRA: quantised base plus small trainable adapters, and why that fits 48GB comfortably
- Overfitting, and how the M6 scorecard catches it
- Pre-tokenising to Arrow shards on disk, because 16GB of RAM will not hold the dataset

**Experiment:** score the adapter against the base model on the M6 golden set. If it does not beat the base, say so and diagnose — a failed fine-tune that is correctly diagnosed is a better outcome than an unmeasured one that looks fine.

**Exit gate:** a number, either way.

---

## What is deliberately not here

- OCR of the scanned CBSE files — real, expensive, separate mission
- Serving to multiple concurrent users, batching, vLLM
- The full markdown corpus
- Anything touching Gitea, k3s, or the wider estate

These come after the ladder, not during it.
