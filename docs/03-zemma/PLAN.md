# zemma — eduZ GEMMA, on-device student companions

Codename for the eduZ-tuned Gemma 4 line that ships inside the Android app.
`zemma-e2b-v1` is the first build.

## What this is

Gemma 4 E2B, fine-tuned so that it *is* an eduZ companion out of the box —
identity, teaching posture and safety rules baked into the weights — exported to
`.litertlm` and running fully offline on a mid-range Indian Android phone.

Not a chatbot wrapper. The model itself answers to the brand.

---

## Three decisions, and why

### 1. We do not strip general knowledge out of the model

The original ask was to remove non-K12 knowledge to shrink the model. This is
not possible and would be harmful if it were.

Knowledge in a dense transformer is distributed, not filed. There is no region
holding "cricket" that can be excised without touching the region holding
"projectile motion" — they are the same weights. Structured pruning removes
whole heads or neurons and degrades *everything*, then needs the original
pre-training corpus (which Google does not publish) to recover.

It also fights the product. `education-core.md` §S1 requires guided-solve that
explains rather than answer-dumps, and the site promises Eddy will "explain this
three different ways". Explaining fractions through a cricket score *requires*
the cricket knowledge. Stripping world knowledge would make the companions
duller, not safer.

**Size comes from elsewhere:** E2B over E4B (2.3B vs 4.5B effective), int4
quantisation, and keeping syllabus facts in retrieval rather than weights.
Fine-tuning changes behaviour; it does not change size.

### 2. One merged model, personas by system prompt

Runtime LoRA sidecars do not currently work on the published Gemma 4 litert-lm
builds. LiteRT-LM issue #3173 (Aug 2026, unanswered) reports a correctly
converted rank-16 adapter producing byte-identical output across 54 test pairs —
the exported graph appears to lack LoRA injection points.

So the shipping artifact is a **merged** model. Which means 9 personas cannot be
9 adapters.

Instead: one adapter teaches the **shared eduZ layer** — the identity, the
guided-solve posture, the never-shame register, the Hindi/Hinglish handling.
Individual character voice is a **system prompt** selected at runtime. That is
also the better design: Lulu's quiz style can be retuned by editing a string,
not by a training run.

### 3. Gemma 4 E2B is the base

E2B is 2.3B effective parameters and runs in under 1.5GB, which matches the
"chapter packs <= small-MB for Jio-era low-end Android phones" constraint in
education-core.md §S1.

There is also a hard local reason: merging an adapter requires the full model in
system RAM at bf16. This box has 16GB. E2B (~5GB) fits comfortably; E4B (~16GB)
would not reliably.

---

## The corpus we actually have

Pulled from B2 `zsm-ncert/converted-mds` — 10,411 files, 879MB.

| | |
|---|---|
| Usable markdown files | 10,364 of 10,368 (100%) |
| Total text | **496M characters** |
| Languages | **23** |
| Grades | class 1–12 |

Top languages: english 2,761 files (132M chars), hindi 1,114 (59M), urdu 790,
gujarati 539, bengali 526, marathi 481, sanskrit 436, punjabi 398.

Also on B2: `zsm-ncert/ncert-pdfs` — 9,181 PDFs, 99.5GB, not yet pulled. These
are the raw source; the markdowns are already converted, so the PDFs are only
needed if we find conversion defects.

**Two data caveats found in the census:**
- 103 files land under a "class 13" that should not exist — a directory naming
  artifact worth tracing before these are used.
- 4 files yield no text at all.

The dominant subject is `fine_arts` (2,114 files), ahead of `mathematics`
(2,002). That is a corpus-shape fact worth knowing before assuming coverage.

---

## The nine student companions

From eduz-ai.com. These nine are student-facing and belong on-device. The other
19 (Arlo, Lala, Maddy, Pipi, Tata, Bibi, Nono, Kuber, Vidya, Saraswati, Naya,
Shanti, Surya, Smriti, Drishti, Chetna, Karma, Coco, Dodo) are teacher, school
and ERP roles — those are server-side and are **out of scope for zemma**.

| Companion | Role | Grades |
|---|---|---|
| Arli 🎈 | Plays-and-teaches — songs, stories, micro-games | 1–5 |
| Eddy 🧸 | The patient study buddy who never sleeps | 6–10 |
| Lulu 🦄 | Quick-fire quizmaster | 6–10 |
| Mina 🌙 | Calm focus coach, pomodoro | 8–12 |
| Nina 🌸 | Bright encourager | 3–8 |
| Riko 🕵️ | Curious detective, asks why | 6–10 |
| Koko 🐯 | Hindi-medium warrior, Hinglish-fluent | 6–12 |
| Yumi 📔 | Boards mode, answer schemes | 10, 12 |
| Lara 🚀 | JEE/NEET grinder — concept, trick, speed | 11–12 |

Site inconsistencies to fix on the web side: the header says 27 companions but
28 are listed; the student section says "eight personalities" but lists nine.

## The shared eduZ layer (what actually gets trained)

These are product rules from `education-core.md`, not invented voice:

1. **Guided-solve, never answer-dump.** Stepwise scaffolding for minors (§S3).
2. **Encouragement register, never shaming.** No guilt language on missed days
   (§S4).
3. **Effort, not rank.** Rank-shaming is explicitly prohibited (§S6).
4. **Hindi-first capable, Hinglish-fluent.** Answer in the language asked.
5. **Age-appropriate always.** Audience is 8–16.
6. **Identity:** answers as an eduZ companion, by name.
7. **Honest limits:** says when it does not know rather than fabricating —
   carried over from the M5 beyond-syllabus work.

---

## Pipeline

```
NCERT markdown (496M chars, 23 langs)
  -> persona dialogue generation (gemma3:12b, local)
  -> human review sample
  -> QLoRA on Gemma 4 E2B (A6000)
  -> merge adapter into base
  -> patch chat template
  -> quantise int4 + export .litertlm (ai-edge-torch)
  -> zemma Android app (AI Edge Gallery fork, rebranded)
  -> test build APK
```

## Known risks

| Risk | Status |
|---|---|
| LoRA sidecar unsupported | Confirmed. Mitigated by merging. |
| Merge step RAM on 16GB box | E2B fits; E4B likely does not. |
| `ai-edge-torch` on Python 3.14 | **Retired.** `ai-edge-torch` is deprecated and renamed `litert-torch`; it installs and imports cleanly on 3.14. `litert-torch export_hf` converts a HF model to tflite, `litert-lm-builder` packages the `.litertlm`. |
| Chat template mismatch after merge | Known failure mode in the field; must be verified on-device, not just locally. |
| Persona bleed across 9 characters | Needs a per-persona eval, not one aggregate score. |
| Judge bias (same family generating and grading) | Carried over from M7. Needs human spot-check. |

## Success criteria for zemma-e2b-v1

1. Answers "who are you" as the selected eduZ companion, in character.
2. Beats base Gemma 4 E2B on a per-persona eval, scored separately.
3. Holds the seven shared-layer rules under adversarial prompting
   (asking it to just give the answer, to shame, to rank).
4. Runs offline on a mid-range Android phone at usable speed.
5. Installable APK the team can hold.
