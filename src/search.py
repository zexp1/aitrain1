"""M4 — brute-force semantic search over the stored vectors.

Cosine similarity is the whole algorithm. Because the stored vectors are
pre-normalised, cosine(a, b) reduces to a single dot product, and searching the
entire corpus is one matrix multiply. At a few thousand chunks a vector database
would add nothing but a dependency and a daemon.

    python src/search.py "how do plants make their food" --k 5
"""
import argparse
import sqlite3
import time

import numpy as np

import ollama

DB = "zlearn.db"


def load(vecs="vectors.npy", ids="vector_ids.npy"):
    return np.load(vecs), np.load(ids)


def search(query, matrix, ids, con, table="chunks", k=5, grade=None):
    qvec = np.asarray(ollama.embed([query], kind="query")[0],
                      dtype=np.float32)
    qvec /= np.linalg.norm(qvec)

    scores = matrix @ qvec                      # cosine similarity, all at once
    order = np.argsort(-scores)

    hits = []
    for idx in order:
        cid = int(ids[idx])
        row = con.execute(
            f"SELECT path, grade, subject, language, heading, text "
            f"FROM {table} WHERE id=?", (cid,)).fetchone()
        if row is None:
            continue
        if grade and row[1] != str(grade):
            continue
        hits.append({"id": cid, "score": float(scores[idx]), "path": row[0],
                     "grade": row[1], "subject": row[2], "language": row[3],
                     "heading": row[4], "text": row[5]})
        if len(hits) >= k:
            break
    return hits


def main():
    p = argparse.ArgumentParser()
    p.add_argument("query")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--db", default=DB)
    p.add_argument("--table", default="chunks")
    p.add_argument("--grade")
    args = p.parse_args()

    matrix, ids = load()
    con = sqlite3.connect(args.db)

    started = time.time()
    hits = search(args.query, matrix, ids, con, args.table, args.k, args.grade)
    elapsed = (time.time() - started) * 1000

    print(f'query: "{args.query}"   [{len(ids)} chunks searched in '
          f'{elapsed:.0f}ms]')
    for i, h in enumerate(hits, 1):
        print(f"\n{i}. score {h['score']:.3f}  class {h['grade']} "
              f"{h['subject']} ({h['language']})")
        print(f"   {h['path']}  #{h['heading'][:60]}")
        print(f"   {' '.join(h['text'].split())[:200]}...")
    con.close()


if __name__ == "__main__":
    main()
