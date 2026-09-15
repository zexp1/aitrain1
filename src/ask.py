"""M1 — ask a question at a grade level, straight to Ollama.

    python src/ask.py --grade 7 "Why is the sky blue?"
"""
import argparse
import sys

import ollama

GRADE_STYLE = {
    range(1, 6): "Use very simple words and short sentences. One idea at a time. "
                 "Use an everyday example a child in India would recognise.",
    range(6, 9): "Use clear school-level language. Define any technical term the "
                 "first time you use it. Keep to about 120 words.",
    range(9, 13): "Use precise CBSE board vocabulary. Structure the answer the way "
                  "an exam answer is structured, with numbered points where the "
                  "question implies steps.",
}


def system_prompt(grade):
    style = next((v for r, v in GRADE_STYLE.items() if grade in r), None)
    if style is None:
        sys.exit(f"grade must be 1-12, got {grade}")
    return (
        f"You are a teacher for an Indian CBSE student in class {grade}. "
        f"{style} "
        "The reader is a minor: keep everything age-appropriate. "
        "If the question is in Hindi or Hinglish, answer in the same language."
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("question")
    p.add_argument("--grade", type=int, default=7)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--model", default=ollama.GEN_MODEL)
    p.add_argument("--json", action="store_true",
                   help="force strict JSON output")
    args = p.parse_args()

    fmt = None
    question = args.question
    if args.json:
        fmt = "json"
        question += ('\n\nReturn JSON only, with keys: "answer" (string), '
                     '"key_terms" (array of strings).')

    text, stats = ollama.generate(
        question, system=system_prompt(args.grade), model=args.model,
        temperature=args.temperature, fmt=fmt)

    print(text.strip())
    print(f"\n[{stats['prompt_tokens']} prompt tok -> "
          f"{stats['output_tokens']} out tok @ "
          f"{stats['tokens_per_sec']:.1f} tok/s, "
          f"{stats['total_sec']:.1f}s, grade={args.grade}, "
          f"temp={args.temperature}]")


if __name__ == "__main__":
    main()
