# M2–M6 — pipeline run log

All numbers from the placeholder corpus (13 NCERT sample chapters from
`zexp1/zsm1`, 488KB of clean markdown). They will change when the real
Backblaze corpus lands, and mostly in the unflattering direction.

## M1 — knobs

| | |
|---|---|
| Token cost, same sentence | English 23 tok, Hindi 29 (1.26x), Hinglish 31 (1.35x) |
| temperature=0.0, 5 runs | 1 distinct answer |
| temperature=1.0, 5 runs | 4 distinct answers |
| grade=4 vs grade=11, same question | 60 vs 1045 output tokens |

Hindi costing 1.26x for identical meaning is a direct commercial fact: the same
answer costs more and fills the context window faster for a Hindi-belt user.

## M2 — corpus census

13 files, 13 yield text (100%), 0 image-only, 0 with undeterminable grade.

**This 100% is not a real number.** These files were already converted to clean
markdown by an earlier pipeline. Raw CBSE PDFs are where the extraction rate
falls apart, and that measurement has not happened yet.

Provenance comes from the directory layout (`<lang>/class<N>/<subject>/`) and is
recorded as `unknown` when the path does not match, never guessed.

## M3 — chunking

| target chars | chunks | avg chars | chunks/file |
|---|---|---|---|
| 800 | 667 | 601 | 51.3 |
| 2000 | 357 | 1089 | 27.5 |
| 4000 | 273 | 1380 | 21.0 |

Chosen: 2000 with 200 overlap, on the M6 evidence below.

## M4 — embeddings

embeddinggemma, 768 dimensions, 357 chunks in 6.1s (58 chunks/s), 1.1MB on disk.
Search over the full corpus takes 150ms–2s, dominated by embedding the query,
not by the similarity scan.

**Bug found and fixed:** EmbeddingGemma is trained with task prefixes
(`task: search result | query:` for queries, `title: ... | text:` for
documents). Without them, retrieval was visibly wrong — "what is sociology"
returned unrelated chunks at 0.32. With them it returns the correct chapter at
0.54. Same model, same corpus, same code path; only the prefix changed.

## M5 — RAG

| question | top score | behaviour |
|---|---|---|
| "What is the scope of sociology?" (in corpus) | 0.664 | cited [S1]–[S4] correctly |
| "Who won the 1983 Cricket World Cup?" (not in corpus) | 0.127 | said corpus does not cover it, then answered flagged as general knowledge |
| same, `--no-fallback` | 0.127 | refused, 9 output tokens |

The 0.664 vs 0.127 gap is the whole beyond-syllabus benchmark in one number.
Nearest-neighbour always returns something — five irrelevant chunks came back
for the cricket question — so the flagging instruction, not the retrieval, is
what prevents fabrication.

## M6 — scorecard

Full eval, 33 questions (29 in-corpus, 4 beyond-syllabus), judge = gemma3:12b.

| | chunk 2000 | chunk 800 |
|---|---|---|
| recall@5 | 29/29 (100%) | 29/29 (100%) |
| recall@1 | 28/29 (97%) | 27/29 (93%) |
| mean top score, in-corpus | 0.602 | 0.599 |
| mean top score, beyond-syllabus | 0.235 | 0.259 |
| separation | +0.367 | +0.340 |
| judge mean /5 | 4.52 | 4.73 |
| grounded | 33/33 (100%) | 32/33 (97%) |

Runtime: 182s for 33 questions (5.5s each), dominated by generation.

**A defect found by running the pipeline, not by reading it.** "What is an empty
set?" was retrieved perfectly (0.708, correct section, English source) and then
answered in Hindi, reproducibly. The system prompt said "if the question is in
Hindi or Hinglish, answer in the same language", and the model generalised that
to the Indian subject matter rather than the actual question language. Rewriting
the rule to name the question as the authority fixed it in both directions.
Grounding rose to 33/33 and the judge mean moved 4.73 -> 4.52; the chunk-800
column below predates that fix and is not directly comparable on generation.

**The one-variable experiment produced a null result at first, and the null was
the finding.** Changing chunk size moved recall@5 by zero, because recall@5 is
saturated: with only 13 files, five guesses nearly always contain the right one.
The metric could not see the change. Tightening to recall@1 made the difference
visible (97% vs 93%), and 2000 also holds a wider in/out separation. A saturated
metric looks like a stable system and is in fact a blind one.

Weakest scores cluster on class 2 mathematics, where the source is activity-based
worksheet prose that does not contain a clean expository answer to retrieve.
