"""M5 — retrieval-augmented generation with citations.

Retrieve first, then generate with the retrieved text in the prompt and an
instruction to cite. Retrieval quality is a hard ceiling on answer quality: the
model cannot ground a claim in a chunk that was never fetched.

    python src/rag.py --grade 11 "What is the scope of sociology?"
    python src/rag.py --no-fallback "..."   # forbid general knowledge
"""
import argparse
import sqlite3

import ollama
import search as search_mod

SYSTEM = """You are a teacher for an Indian CBSE student in class {grade}.
Answer using the SOURCES below.

Rules:
- Cite the source number for every factual claim, like [S2].
- If the sources do not contain the answer, say exactly:
  "The corpus does not cover this."
  {fallback}
- Never invent a citation. A claim with no source gets no bracket.
- The reader is a minor: keep the answer age-appropriate.
- Answer in the language of the QUESTION, not the language of the sources.
  An English question gets an English answer even when the sources or the
  subject matter are Indian. Only a question actually written in Hindi or
  Hinglish gets a Hindi or Hinglish answer.
"""

FALLBACK_ON = ("Then answer from your own general knowledge, and prefix that "
               "part with [GENERAL KNOWLEDGE - NOT FROM CORPUS].")
FALLBACK_OFF = "Then stop. Do not answer from general knowledge."


def build_context(hits, budget_chars=6000):
    """Pack retrieved chunks into the prompt until the budget runs out.

    Context is not free — every chunk here is prompt tokens we pay for and KV
    cache we allocate, so the budget is explicit rather than 'send everything'.
    """
    parts, used = [], 0
    for i, h in enumerate(hits, 1):
        block = (f"[S{i}] source: {h['path']} | class {h['grade']} | "
                 f"{h['subject']} | section: {h['heading']}\n{h['text']}")
        if used + len(block) > budget_chars:
            break
        parts.append(block)
        used += len(block)
    return "\n\n".join(parts), len(parts), used


def answer(question, grade=9, k=5, allow_fallback=True, db="zlearn.db",
           table="chunks", quiet=False):
    matrix, ids = search_mod.load()
    con = sqlite3.connect(db)
    hits = search_mod.search(question, matrix, ids, con, table, k)
    con.close()

    context, used_n, used_chars = build_context(hits)
    system = SYSTEM.format(
        grade=grade,
        fallback=FALLBACK_ON if allow_fallback else FALLBACK_OFF)
    prompt = f"SOURCES:\n{context}\n\nQUESTION: {question}"

    text, stats = ollama.generate(prompt, system=system, temperature=0.0)

    if not quiet:
        print(f"retrieved {len(hits)} chunks, used {used_n} "
              f"({used_chars:,} chars) | top score {hits[0]['score']:.3f}"
              if hits else "retrieved nothing")
        print(f"\n{text.strip()}\n")
        print("SOURCES USED:")
        for i, h in enumerate(hits[:used_n], 1):
            print(f"  [S{i}] {h['path']} #{h['heading'][:50]} "
                  f"(score {h['score']:.3f})")
        print(f"\n[{stats['prompt_tokens']} prompt tok -> "
              f"{stats['output_tokens']} out tok @ "
              f"{stats['tokens_per_sec']:.1f} tok/s]")
    return text, hits, stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("question")
    p.add_argument("--grade", type=int, default=9)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--db", default="zlearn.db")
    p.add_argument("--table", default="chunks")
    p.add_argument("--no-fallback", action="store_true")
    args = p.parse_args()
    answer(args.question, args.grade, args.k, not args.no_fallback,
           args.db, args.table)


if __name__ == "__main__":
    main()
