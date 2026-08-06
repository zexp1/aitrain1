# zlearn — local AI foundations, eduZ K-12 India

This file governs every Claude Code session in this workspace. Read it fully before the first action of any session.

---

## 1. What this workstream is

A **learning build**, not a delivery build.

The operator is a seasoned founder with deep strategy, finance, and hands-on software-building grounding, but **no prior hands-on AI/ML experience**. The objective is that by the end of this ladder he can reason about AI systems unaided — chunking, embeddings, retrieval, evaluation, fine-tuning — without needing to ask what a term means.

**A working artifact the operator does not understand is a failed session.** Optimise for his comprehension, not for the artifact's sophistication.

---

## 2. Working agreement — non-negotiable

1. **Explain before you build.** Before writing code that introduces a new concept, give a plain-language explanation in 10 lines or fewer: what it is, why it is needed *here*, and what would visibly break without it. Then write the code.

2. **No frameworks in the first pass.** Forbidden: LangChain, LlamaIndex, Haystack, Chroma, Weaviate, Pinecone, Qdrant, any agent framework, any orchestration library. These abstractions *are the thing being learned* — importing them defeats the exercise. If a task genuinely needs one later, say so explicitly and explain what it does that we cannot do in 30 lines of our own code.

3. **Small, readable files.** No source file over ~150 lines. Explicit over clever. No list comprehensions three levels deep. Comments explain **why**, never **what**.

4. **He runs it, you don't.** After writing a script, stop. Tell him the exact command. Wait for his actual output before proceeding. Do not run it yourself and report success.

5. **One concept per session.** When the session's concept is complete, say so plainly and tell him to `/clear` before the next mission. Do not let a session sprawl across two missions.

6. **Every run prints numbers.** Tokens/sec, VRAM used, chunk count, extraction success rate, recall@5, latency. Learning without measurement is vibes. If a script produces no measurable output, it is incomplete.

7. **Correct him.** If he asks for something built on a wrong mental model, say so directly and explain the correction before building. Do not politely build the wrong thing. He has explicitly asked for this.

8. **Debrief every session** in exactly this shape, five lines:
   - Built:
   - Concept it taught:
   - Number that moved:
   - Still fuzzy:
   - Next:

---

## 3. Hardware — design against this

| | |
|---|---|
| Box | `pk31`, Ubuntu Server |
| CPU | Ryzen 5900X |
| GPU | NVIDIA RTX A6000, **48GB VRAM** |
| System RAM | **16GB — this is the binding constraint** |

Rules that follow from the hardware:

- **RAM is the bottleneck, not the GPU.** Stream from disk; never load a full dataset into memory. Prefer generators over lists, SQLite over in-memory dicts, memory-mapped reads over `read()`.
- **Never use naive `transformers` checkpoint loading** — it stages the full checkpoint through CPU RAM and will OOM the box even when the model fits in VRAM. Use Ollama (mmap from disk) for all serving in this workstream.
- **A6000 is Ampere.** bf16 and INT8 are native. **FP8 is not** — never suggest FP8 kernels or quantisation.
- **No Docker stacks, no k3s, no Postgres, no Redis** in this workstream. One Python venv, local files, SQLite. Infrastructure is a different project.

---

## 4. The stack — do not expand without asking

| Tool | Role |
|---|---|
| Ollama | model serving (generation + embeddings) |
| Python 3.11+, single venv | everything else |
| `pymupdf` | PDF text extraction |
| `numpy` | vector maths |
| `sqlite3` (stdlib) | chunk store, manifest, eval results |
| `httpx` | talking to the Ollama API |

Anything beyond this list requires explicit justification of what it does that we cannot write ourselves in a short function. Default answer is no.

---

## 5. Domain context

eduZ is an AI-first K-12 edtech venture for India, aimed at underserved learners, with a Hindi-belt focus.

The working corpus for this ladder is **raw CBSE material the operator has downloaded**. It is deliberately partial — some files complete, some incomplete, some likely scanned images rather than text. **Treat the mess as signal, not noise.** Real deployment data looks exactly like this, and learning to characterise a broken corpus is one of the actual skills on offer here.

Two standing benchmarks. Every artifact gets asked both:

1. **Beyond-syllabus:** can it help a student with a general question, not only a question whose answer sits in the retrieved chapter? A system that can only recite the textbook has failed the brief.
2. **India K-12 fit:** board vocabulary, exam answer formats, grade-appropriate language, tolerance for Hindi and Hinglish input.

---

## 6. Data handling rules

- **Never assume a file is complete.** Always report what was found against what was expected, and make the gap visible in output.
- **Every chunk carries provenance:** source file, page number, and where determinable — grade, subject, chapter. If a field cannot be determined, record `unknown`. Never infer or guess a provenance value.
- **The audience is minors.** Any generation path must keep age-appropriateness in scope. Flag it if a design decision weakens that.

---

## 7. Session protocol

- One mission per session. Read `MISSIONS.md`, confirm which mission is active, execute only that one.
- At the end of a mission, append the operator's own words to `CONCEPT-LEDGER.md` — prompt him for them; do not write his explanations for him.
- If the operator asks to skip ahead past an unmet gate in `MISSIONS.md`, say no and explain what the skipped mission would have given him.
