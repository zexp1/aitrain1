"""M7 — QLoRA adapter for CBSE answer FORMAT.

What LoRA modifies: the base weights stay frozen. We insert small low-rank
matrices next to the attention projections and train only those — a few million
parameters instead of billions. QLoRA additionally holds the frozen base in
4-bit, so the memory cost is the 4-bit base plus tiny adapters.

Why this fits: a 1B base in 4-bit is well under 1GB of VRAM; even with
activations and optimiser state for the adapters, an A6000 is not stressed. The
binding constraint here is the 16GB of system RAM, which is why the dataset is
pre-tokenised to disk and streamed rather than held as Python objects.

Requires Hugging Face access to the gated Gemma repo:
    huggingface-cli login          (after accepting the licence on the model page)

    python src/train_qlora.py --base google/gemma-3-1b-it
"""
import argparse
import json
import os


def build_dataset(path, tokenizer, max_len, cache_dir):
    from datasets import Dataset

    def rows():
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)

    def to_text(r):
        # Chat template, so the adapter learns format in the same wrapper the
        # model is served with. A mismatch here silently wrecks the fine-tune.
        msgs = [
            {"role": "user",
             "content": f"{r['instruction']}\n\n{r['input']}"},
            {"role": "assistant", "content": r["output"]},
        ]
        return {"text": tokenizer.apply_chat_template(msgs, tokenize=False)}

    ds = Dataset.from_list([to_text(r) for r in rows()])

    def tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_len)

    # Tokenise once, write Arrow shards to disk, then memory-map them. This is
    # the step that keeps a 16GB box alive on a real-sized dataset.
    ds = ds.map(tok, batched=True, remove_columns=["text"],
                cache_file_name=os.path.join(cache_dir, "tokenised.arrow"))
    return ds


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="google/gemma-3-1b-it")
    p.add_argument("--data", default="eval/sft_format.jsonl")
    p.add_argument("--out", default="adapters/cbse-format")
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--max-len", type=int, default=1024)
    p.add_argument("--batch", type=int, default=4)
    args = p.parse_args()

    import torch
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig, DataCollatorForLanguageModeling,
                              Trainer, TrainingArguments)
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    os.makedirs(args.out, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ds = build_dataset(args.data, tokenizer, args.max_len, args.out)
    split = ds.train_test_split(test_size=0.1, seed=7)

    # bf16 compute: the A6000 is Ampere, so bf16 and INT8 are native. FP8 is not
    # supported on this card and must never be requested here.
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.base, quantization_config=quant, torch_dtype=torch.bfloat16,
        device_map={"": 0}, low_cpu_mem_usage=True)
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False

    lora = LoraConfig(
        r=args.rank, lora_alpha=args.rank * 2, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=args.out,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            bf16=True,
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            # Held-out loss rising while training loss falls is overfitting;
            # the M6 scorecard is the check that actually matters.
            report_to=[],
            dataloader_num_workers=2,
        ),
        train_dataset=split["train"],
        eval_dataset=split["test"],
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    trainer.train()
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"\nadapter saved to {args.out}")
    print("Next: score it against the base on eval/golden.jsonl (M6). "
          "If it does not beat the base, say so and diagnose.")


if __name__ == "__main__":
    main()
