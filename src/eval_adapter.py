"""M7 — score the adapter against the frozen base. A number, either way.

The adapter was trained on answer FORMAT, so format is what is measured: does a
3-mark question get three numbered points, does a 1-mark question get one
sentence, is the register right. Judged by gemma3:12b, which never saw the
training run.

Held-out split only — the same 10% the trainer evaluated on, never the 90% it
learned from.

    python src/eval_adapter.py --n 40
"""
import argparse
import json

import ollama

JUDGE = """You grade CBSE exam answer FORMAT, not factual accuracy.

The question is worth {marks} mark(s). Correct format for that value:
  1 mark  = one precise sentence, no preamble
  2 marks = two short numbered points
  3 marks = three numbered points, one sentence each
  5 marks = a definition line, four numbered points, a conclusion line

Return JSON only: {{"format_score": 0-5, "reason": "one line"}}
5 = exactly the expected shape and exam register.
0 = ignores the expected shape entirely (e.g. a rambling paragraph for 3 marks).
Ignore whether the facts are right."""


def load_model(base, adapter=None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    quant = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForCausalLM.from_pretrained(
        base, quantization_config=quant, dtype=torch.bfloat16,
        device_map={"": 0}, low_cpu_mem_usage=True)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model, tok


def generate(model, tok, instruction, question, max_new=300):
    import torch
    msgs = [{"role": "user", "content": f"{instruction}\n\n{question}"}]
    text = tok.apply_chat_template(msgs, tokenize=False,
                                   add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
    return tok.decode(out[0][ids["input_ids"].shape[1]:],
                      skip_special_tokens=True).strip()


def score(item, answer_text):
    raw, _ = ollama.generate(
        f"QUESTION: {item['input']}\n\nANSWER:\n{answer_text}",
        system=JUDGE.format(marks=item["marks"]), temperature=0.0, fmt="json")
    try:
        return int(json.loads(raw).get("format_score", 0))
    except (json.JSONDecodeError, ValueError, TypeError):
        return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="unsloth/gemma-3-1b-it")
    p.add_argument("--adapter", default="adapters/cbse-format")
    p.add_argument("--data", default="eval/sft_format.jsonl")
    p.add_argument("--n", type=int, default=40)
    args = p.parse_args()

    with open(args.data, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    # The trainer used train_test_split(test_size=0.1, seed=7) on this file in
    # order, so the tail is the held-out portion.
    held_out = rows[int(len(rows) * 0.9):][:args.n]
    print(f"scoring {len(held_out)} held-out examples\n")

    results = {}
    for label, adapter in (("base", None), ("adapter", args.adapter)):
        model, tok = load_model(args.base, adapter)
        scores = []
        for i, item in enumerate(held_out, 1):
            text = generate(model, tok, item["instruction"], item["input"])
            scores.append(score(item, text))
            print(f"\r  {label}: {i}/{len(held_out)}", end="", flush=True)
        results[label] = scores
        print(f"   mean {sum(scores) / len(scores):.2f}/5")
        del model
        import torch, gc
        gc.collect(); torch.cuda.empty_cache()

    b = sum(results["base"]) / len(results["base"])
    a = sum(results["adapter"]) / len(results["adapter"])
    print(f"\n=== M7 FORMAT SCORECARD ===")
    print(f"base    {b:.2f}/5")
    print(f"adapter {a:.2f}/5")
    print(f"delta   {a - b:+.2f}")
    for marks in sorted({r["marks"] for r in held_out}):
        idx = [i for i, r in enumerate(held_out) if r["marks"] == marks]
        bm = sum(results["base"][i] for i in idx) / len(idx)
        am = sum(results["adapter"][i] for i in idx) / len(idx)
        print(f"  {marks}-mark (n={len(idx)}): base {bm:.2f} -> "
              f"adapter {am:.2f}  ({am - bm:+.2f})")
    print("\n" + ("adapter beats base." if a > b else
                  "adapter does NOT beat base — diagnose, do not ship."))


if __name__ == "__main__":
    main()
