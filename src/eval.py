"""M6 — the scorecard. Retrieval and generation measured separately.

They are separated because they fail differently. Retrieval can fetch the wrong
chapter (recall@k catches that). Generation can be handed the right chapter and
still write nonsense, or be handed nothing and fabricate (the judge catches
that). A single blended score would hide which half broke.

    python src/eval.py
    python src/eval.py --k 3 --table chunks_800 --vecs v800.npy --label small
"""
import argparse
import json
import sqlite3
import time

import ollama
import rag
import search as search_mod

JUDGE_SYSTEM = """You grade a student-facing answer against a reference answer.
Return JSON only: {"score": 0-5, "grounded": true/false, "reason": "one line"}

score 5 = fully correct and complete; 3 = partially correct; 0 = wrong or empty.
grounded = true only if factual claims carry [S#] citations OR the answer
correctly states the corpus does not cover this and flags general knowledge.
Judge only correctness against the reference, not style or length."""


def judge(question, answer_text, reference):
    prompt = (f"QUESTION: {question}\n\nREFERENCE ANSWER: {reference}\n\n"
              f"ANSWER TO GRADE: {answer_text}")
    raw, _ = ollama.generate(prompt, system=JUDGE_SYSTEM, temperature=0.0,
                             fmt="json")
    try:
        d = json.loads(raw)
        return int(d.get("score", 0)), bool(d.get("grounded", False)), \
            str(d.get("reason", ""))[:80]
    except (json.JSONDecodeError, ValueError, TypeError):
        return 0, False, "judge returned unparseable output"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--golden", default="eval/golden.jsonl")
    p.add_argument("--db", default="zlearn.db")
    p.add_argument("--table", default="chunks")
    p.add_argument("--vecs", default="vectors.npy")
    p.add_argument("--ids", default="vector_ids.npy")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--label", default="default")
    p.add_argument("--skip-judge", action="store_true")
    args = p.parse_args()

    with open(args.golden, encoding="utf-8") as f:
        golden = [json.loads(line) for line in f if line.strip()]

    matrix, ids = search_mod.load(args.vecs, args.ids)
    con = sqlite3.connect(args.db)

    hit_at_k = miss = 0
    scores, grounded_flags, rows = [], [], []
    started = time.time()

    for item in golden:
        hits = search_mod.search(item["question"], matrix, ids, con,
                                 args.table, args.k)
        paths = [h["path"] for h in hits]
        top = hits[0]["score"] if hits else 0.0

        # recall@k: for in-corpus questions only. Beyond-syllabus questions have
        # no correct chunk to find, so scoring them here would be meaningless.
        expect = item.get("expect_path")
        if expect:
            got = expect in paths
            hit_at_k += got
        else:
            got = None
            miss += 1

        context, used_n, _ = rag.build_context(hits)
        system = rag.SYSTEM.format(grade=item.get("grade", 9),
                                   fallback=rag.FALLBACK_ON)
        answer_text, _ = ollama.generate(
            f"SOURCES:\n{context}\n\nQUESTION: {item['question']}",
            system=system, temperature=0.0)

        if args.skip_judge:
            score, is_grounded, reason = 0, False, "skipped"
        else:
            score, is_grounded, reason = judge(
                item["question"], answer_text, item["reference"])
        scores.append(score)
        grounded_flags.append(is_grounded)
        rows.append((item["id"], item["kind"], got, top, score, is_grounded,
                     reason))
        print(f"\r  scored {len(rows)}/{len(golden)}", end="", flush=True)

    elapsed = time.time() - started
    con.close()

    in_corpus = [r for r in rows if r[2] is not None]
    beyond = [r for r in rows if r[2] is None]

    print(f"\n\n=== SCORECARD [{args.label}] "
          f"k={args.k} table={args.table} ===")
    print(f"questions              {len(golden)}  "
          f"({len(in_corpus)} in-corpus, {len(beyond)} beyond-syllabus)")
    print(f"\nRETRIEVAL")
    print(f"  recall@{args.k}              "
          f"{hit_at_k}/{len(in_corpus)}  "
          f"({hit_at_k / max(len(in_corpus), 1) * 100:.0f}%)")
    print(f"  mean top score       {sum(r[3] for r in in_corpus) / max(len(in_corpus), 1):.3f} "
          f"(in-corpus)")
    print(f"  mean top score       {sum(r[3] for r in beyond) / max(len(beyond), 1):.3f} "
          f"(beyond-syllabus)  <- separation is the signal")
    print(f"\nGENERATION (judge: {ollama.GEN_MODEL})")
    print(f"  mean score /5        {sum(scores) / len(scores):.2f}")
    print(f"  grounded             {sum(grounded_flags)}/{len(rows)} "
          f"({sum(grounded_flags) / len(rows) * 100:.0f}%)")
    for kind in sorted({r[1] for r in rows}):
        sub = [r for r in rows if r[1] == kind]
        print(f"    {kind:20s} {sum(r[4] for r in sub) / len(sub):.2f}/5  "
              f"(n={len(sub)})")
    print(f"\nelapsed                {elapsed:.0f}s "
          f"({elapsed / len(golden):.1f}s/question)")

    worst = sorted(rows, key=lambda r: r[4])[:5]
    print("\nWORST 5 (look here first):")
    for r in worst:
        print(f"  q{r[0]:<3d} {r[1]:18s} score {r[4]} "
              f"retrieved={r[2]} top={r[3]:.3f}  {r[6]}")


if __name__ == "__main__":
    main()
