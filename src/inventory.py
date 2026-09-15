"""M2 — walk the corpus and write one honest row per file.

Reports what was FOUND against what was EXPECTED. A file that opens fine and
yields no text is the failure mode this mission exists to make visible.

    python src/inventory.py corpus/
"""
import argparse
import os
import re
import sqlite3
import sys

DB = "zlearn.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS files (
  path TEXT PRIMARY KEY,
  size_bytes INTEGER,
  kind TEXT,            -- pdf | markdown | other
  pages INTEGER,        -- NULL for non-PDF
  grade TEXT,           -- 'unknown' when not determinable, never guessed
  subject TEXT,
  language TEXT,
  has_text INTEGER,     -- 1 if extraction yielded usable characters
  char_count INTEGER,
  head TEXT             -- first 200 extracted characters
);
"""

# Provenance comes from the directory layout: <lang>/class<N>/<subject>/file.md
# Anything not matching that shape is recorded as unknown, not inferred.
GRADE_RE = re.compile(r"class[_-]?(\d{1,2})\b", re.I)


def provenance(path, root):
    """Locate the class<N> segment and read language/subject around it.

    Position-based parsing broke on the real corpus, which nests an extra batch
    directory above the language. The class segment is the reliable anchor.
    """
    rel = os.path.relpath(path, root)
    parts = rel.split(os.sep)
    grade = subject = language = "unknown"
    for i, part in enumerate(parts):
        m = GRADE_RE.fullmatch(part)
        if not m:
            continue
        grade = m.group(1)
        if i >= 1:
            language = parts[i - 1]
        if i + 1 < len(parts) - 1:          # a dir after class, not the file
            subject = parts[i + 1]
        break
    return grade, subject, language


def read_pdf(path):
    """Returns (page_count, text). Empty text means image-only — needs OCR."""
    import fitz
    text = []
    with fitz.open(path) as doc:
        for page in doc:
            text.append(page.get_text())
        return doc.page_count, "\n".join(text)


def read_markdown(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        return None, f.read()


def walk(root):
    for dirpath, _, names in os.walk(root):
        for name in sorted(names):
            yield os.path.join(dirpath, name)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root")
    p.add_argument("--db", default=DB)
    args = p.parse_args()

    if not os.path.isdir(args.root):
        sys.exit(f"not a directory: {args.root}")

    con = sqlite3.connect(args.db)
    con.executescript(SCHEMA)

    seen = usable = image_only = no_grade = 0
    for path in walk(args.root):
        ext = os.path.splitext(path)[1].lower()
        if ext == ".pdf":
            kind, reader = "pdf", read_pdf
        elif ext in (".md", ".markdown", ".txt"):
            kind, reader = "markdown", read_markdown
        else:
            continue

        seen += 1
        try:
            pages, text = reader(path)
        except Exception as e:                      # a corrupt file is data too
            pages, text = None, ""
            print(f"  ! unreadable {path}: {e}", file=sys.stderr)

        stripped = text.strip()
        has_text = 1 if len(stripped) > 200 else 0
        usable += has_text
        if kind == "pdf" and not has_text:
            image_only += 1

        grade, subject, language = provenance(path, args.root)
        if grade == "unknown":
            no_grade += 1

        con.execute(
            "INSERT OR REPLACE INTO files VALUES (?,?,?,?,?,?,?,?,?,?)",
            (path, os.path.getsize(path), kind, pages, grade, subject,
             language, has_text, len(stripped), stripped[:200]))

    con.commit()

    print(f"\n=== Corpus census: {args.root} ===")
    print(f"files seen                 {seen}")
    print(f"yield usable text          {usable}"
          f"  ({usable / seen * 100:.0f}%)" if seen else "")
    print(f"open but yield nothing     {seen - usable}   <- needs OCR or is broken")
    print(f"  of which image-only PDFs {image_only}")
    print(f"no determinable grade      {no_grade}")

    print("\nBy grade / subject (usable only):")
    for row in con.execute(
            "SELECT language, grade, subject, COUNT(*), SUM(char_count) "
            "FROM files WHERE has_text=1 GROUP BY 1,2,3 ORDER BY 1,2,3"):
        print(f"  {row[0]:8s} class {row[1]:2s} {row[2]:16s} "
              f"{row[3]} file(s)  {row[4]:,} chars")
    con.close()


if __name__ == "__main__":
    main()
