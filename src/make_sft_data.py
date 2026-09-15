"""M7 data — build CBSE-style answer-FORMAT training examples.

The adapter must teach shape, not syllabus facts: step structure, marks-
appropriate length, exam register. Syllabus content changes every year and
varies by board, so it belongs in retrieval where it can be swapped, never
baked into weights where it can only be retrained out.

Each example is generated from a real corpus chunk, so the register is grounded
in actual NCERT prose rather than invented.

    python src/make_sft_data.py --n 500
"""
import argparse
import json
import random
import sqlite3

import ollama

# Marks drive length and structure in CBSE — this is exactly the behaviour the
# adapter should learn.
FORMATS = [
    (1, "a one-mark answer: a single precise sentence, no preamble"),
    (2, "a two-mark answer: two short numbered points"),
    (3, "a three-mark answer: three numbered points, each one sentence"),
    (5, "a five-mark answer: a one-line definition, then four numbered points, "
        "then a one-line conclusion"),
]

MAKER = """You write CBSE exam questions and model answers for class {grade}.

From the PASSAGE below, write ONE exam question worth {marks} mark(s) and its
model answer formatted as {shape}.

Rules:
- The question must be answerable from the passage alone.
- Use CBSE board vocabulary and exam register.
- Age-appropriate for class {grade}.
- Return JSON only: {{"question": "...", "answer": "..."}}"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--db", default="zlearn.db")
    p.add_argument("--out", default="eval/sft_format.jsonl")
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    random.seed(args.seed)
    con = sqlite3.connect(args.db)
    chunks = con.execute(
        "SELECT id, grade, subject, text FROM chunks "
        "WHERE char_count > 600 ORDER BY id").fetchall()
    con.close()
    if not chunks:
        raise SystemExit("no chunks — run chunk.py first")

    written = failed = 0
    with open(args.out, "w", encoding="utf-8") as f:
        while written < args.n:
            cid, grade, subject, text = random.choice(chunks)
            marks, shape = random.choice(FORMATS)
            grade_label = grade if grade != "unknown" else "9"

            raw, _ = ollama.generate(
                f"PASSAGE:\n{text[:2500]}",
                system=MAKER.format(grade=grade_label, marks=marks, shape=shape),
                temperature=0.7, fmt="json")
            try:
                d = json.loads(raw)
                q, a = d["question"].strip(), d["answer"].strip()
            except (json.JSONDecodeError, KeyError, AttributeError):
                failed += 1
                continue
            if not q or not a:
                failed += 1
                continue

            f.write(json.dumps({
                "instruction": f"Answer this class {grade_label} CBSE "
                               f"{subject} question worth {marks} mark(s).",
                "input": q,
                "output": a,
                "marks": marks,
                "grade": grade_label,
                "subject": subject,
                "source_chunk": cid,
            }, ensure_ascii=False) + "\n")
            f.flush()                     # so a long run is observable on disk
            written += 1
            print(f"\r  wrote {written}/{args.n} (rejected {failed})",
                  end="", flush=True)

    print(f"\nwrote      {written} examples -> {args.out}")
    print(f"rejected   {failed} (unparseable model output)")


if __name__ == "__main__":
    main()
