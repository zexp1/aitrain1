"""M3 — split usable files into retrievable chunks, with provenance.

Splits on structure first (headings, then blank lines), and only falls back to
hard character cuts when a single block is oversized. Splitting mid-sentence on
a character count is what destroys meaning; paragraph boundaries are free
semantic hints the author already put there.

    python src/chunk.py --target-chars 2000 --overlap-chars 200
"""
import argparse
import re
import sqlite3

DB = "zlearn.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
  id INTEGER PRIMARY KEY,
  path TEXT, grade TEXT, subject TEXT, language TEXT,
  page INTEGER,            -- unknown for markdown; kept for PDF provenance
  heading TEXT,
  ord INTEGER,             -- position within the file
  char_count INTEGER,
  text TEXT
);
CREATE INDEX IF NOT EXISTS chunks_path ON chunks(path);
"""

HEADING_RE = re.compile(r"^#{1,6}\s+(.*)$", re.M)


def blocks(text):
    """Yield (heading, block_text) walking the document in order."""
    positions = [(m.start(), m.group(1).strip()) for m in HEADING_RE.finditer(text)]
    if not positions:
        yield "unknown", text
        return
    if positions[0][0] > 0:
        yield "unknown", text[:positions[0][0]]
    for i, (start, heading) in enumerate(positions):
        end = positions[i + 1][0] if i + 1 < len(positions) else len(text)
        yield heading, text[start:end]


def split_block(block, target, overlap):
    """Paragraph-greedy packing, with a hard cut only for oversized paragraphs."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", block) if p.strip()]
    out, cur = [], ""
    for para in paras:
        while len(para) > target:               # a single huge paragraph
            out.append(para[:target])
            para = para[target - overlap:]
        if len(cur) + len(para) + 2 <= target:
            cur = f"{cur}\n\n{para}" if cur else para
        else:
            if cur:
                out.append(cur)
            # Overlap carries the tail of the previous chunk forward, so a fact
            # that straddles a boundary still appears whole in one chunk.
            tail = cur[-overlap:] if cur and overlap else ""
            cur = f"{tail}\n\n{para}".strip() if tail else para
    if cur:
        out.append(cur)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--db", default=DB)
    p.add_argument("--target-chars", type=int, default=2000)
    p.add_argument("--overlap-chars", type=int, default=200)
    p.add_argument("--table", default="chunks")
    args = p.parse_args()

    con = sqlite3.connect(args.db)
    con.executescript(SCHEMA.replace("chunks", args.table))
    con.execute(f"DELETE FROM {args.table}")

    rows = con.execute(
        "SELECT path, grade, subject, language FROM files WHERE has_text=1"
    ).fetchall()

    total = 0
    for path, grade, subject, language in rows:
        with open(path, encoding="utf-8", errors="replace") as f:
            text = f.read()
        ordinal = 0
        for heading, block in blocks(text):
            for piece in split_block(block, args.target_chars, args.overlap_chars):
                piece = piece.strip()
                if len(piece) < 50:            # scraps carry no retrievable signal
                    continue
                con.execute(
                    f"INSERT INTO {args.table} (path, grade, subject, language, "
                    "page, heading, ord, char_count, text) "
                    "VALUES (?,?,?,?,?,?,?,?,?)",
                    (path, grade, subject, language, None, heading, ordinal,
                     len(piece), piece))
                ordinal += 1
                total += 1

    con.commit()
    stats = con.execute(
        f"SELECT COUNT(*), AVG(char_count), MIN(char_count), MAX(char_count) "
        f"FROM {args.table}").fetchone()
    print(f"target={args.target_chars} overlap={args.overlap_chars} "
          f"table={args.table}")
    print(f"chunks       {stats[0]}")
    print(f"chars avg    {stats[1]:.0f}   min {stats[2]}   max {stats[3]}")
    print(f"est. tokens  ~{stats[1] / 4:.0f} avg per chunk (rough: 4 chars/token)")
    print(f"files        {len(rows)}  ->  {total / len(rows):.1f} chunks/file")
    con.close()


if __name__ == "__main__":
    main()
