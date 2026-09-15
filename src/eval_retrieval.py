"""M6 helper — retrieval-only sweep. No generation, so it runs in seconds.

recall@5 saturates on a small corpus: with only 13 files, five guesses almost
always include the right one. recall@1 is the harsher question — did the single
best chunk come from the right chapter — and that is where chunking choices
actually show up.

    python src/eval_retrieval.py
"""
import json
import sqlite3

import search as search_mod

VARIANTS = [
    ("chunk_800", "chunks_800", "v800.npy", "ids800.npy"),
    ("chunk_2000", "chunks", "vectors.npy", "vector_ids.npy"),
]
KS = (1, 3, 5)


def main():
    with open("eval/golden.jsonl", encoding="utf-8") as f:
        golden = [json.loads(line) for line in f if line.strip()]
    in_corpus = [g for g in golden if g.get("expect_path")]
    beyond = [g for g in golden if not g.get("expect_path")]

    print(f"{len(in_corpus)} in-corpus questions, {len(beyond)} beyond-syllabus\n")
    header = "variant".ljust(12) + "".join(f"recall@{k}".rjust(11) for k in KS)
    print(header + "sep(in-beyond)".rjust(16))
    print("-" * len(header + "sep(in-beyond)".rjust(16)))

    for label, table, vecs, ids_file in VARIANTS:
        matrix, ids = search_mod.load(vecs, ids_file)
        con = sqlite3.connect("zlearn.db")

        maxk = max(KS)
        cache = {}
        for g in golden:
            hits = search_mod.search(g["question"], matrix, ids, con, table,
                                     maxk)
            cache[g["id"]] = hits

        cells = []
        for k in KS:
            got = sum(1 for g in in_corpus
                      if g["expect_path"] in [h["path"] for h in cache[g["id"]][:k]])
            cells.append(f"{got}/{len(in_corpus)} ({got / len(in_corpus) * 100:.0f}%)")

        mean_in = sum(cache[g["id"]][0]["score"] for g in in_corpus) / len(in_corpus)
        mean_out = sum(cache[g["id"]][0]["score"] for g in beyond) / len(beyond)

        print(label.ljust(12) + "".join(c.rjust(11) for c in cells)
              + f"{mean_in - mean_out:+.3f}".rjust(16))
        con.close()

    print("\nsep = mean top score on in-corpus questions minus beyond-syllabus.")
    print("A large gap means a score threshold could reject out-of-corpus")
    print("questions before generation ever runs.")


if __name__ == "__main__":
    main()
