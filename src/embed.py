"""M4 — embed every chunk and store the vectors in a .npy file.

An embedding turns text into a fixed-length vector positioned so that related
meanings land near each other. Nothing is generated here — this mission has no
LLM in it, because retrieval quality is measurable on its own and it caps
everything downstream.

    python src/embed.py
"""
import argparse
import sqlite3
import time

import numpy as np

import ollama

DB = "zlearn.db"
VECS = "vectors.npy"
IDS = "vector_ids.npy"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--db", default=DB)
    p.add_argument("--table", default="chunks")
    p.add_argument("--vecs", default=VECS)
    p.add_argument("--ids", default=IDS)
    p.add_argument("--batch", type=int, default=16)
    args = p.parse_args()

    con = sqlite3.connect(args.db)
    rows = con.execute(
        f"SELECT id, text FROM {args.table} ORDER BY id").fetchall()
    con.close()
    if not rows:
        raise SystemExit("no chunks — run chunk.py first")

    ids, vectors = [], []
    started = time.time()
    # Batched so the request stays small; 16GB of RAM will not hold the whole
    # corpus as Python strings plus float lists at once on a real corpus.
    for i in range(0, len(rows), args.batch):
        batch = rows[i:i + args.batch]
        vecs = ollama.embed([t for _, t in batch], kind="document")
        ids.extend(cid for cid, _ in batch)
        vectors.extend(vecs)
        print(f"\r  embedded {len(ids)}/{len(rows)}", end="", flush=True)

    elapsed = time.time() - started
    arr = np.asarray(vectors, dtype=np.float32)
    # Pre-normalise once, so cosine similarity at query time is a plain dot
    # product — the expensive part is done here, not on every search.
    arr /= np.linalg.norm(arr, axis=1, keepdims=True)

    np.save(args.vecs, arr)
    np.save(args.ids, np.asarray(ids, dtype=np.int64))

    print(f"\nchunks       {len(ids)}")
    print(f"dimensions   {arr.shape[1]}")
    print(f"elapsed      {elapsed:.1f}s  ({len(ids) / elapsed:.1f} chunks/s)")
    print(f"on disk      {arr.nbytes / 1e6:.1f} MB  -> {args.vecs}")


if __name__ == "__main__":
    main()
