"""Thin wrapper over the Ollama HTTP API. No client library — this is the whole
surface we need, and seeing the raw request/response is the point.
"""
import httpx

HOST = "http://localhost:11434"
GEN_MODEL = "gemma3:12b"
EMBED_MODEL = "embeddinggemma"

# Long context costs VRAM linearly via the KV cache (see docs/02-runs/M0).
# 8k is enough to hold retrieved chunks plus an answer.
NUM_CTX = 8192


def generate(prompt, system=None, model=GEN_MODEL, temperature=0.0,
             top_p=0.9, num_ctx=NUM_CTX, fmt=None, timeout=600):
    """One-shot completion. Returns (text, stats_dict)."""
    body = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature, "top_p": top_p,
                    "num_ctx": num_ctx},
    }
    if system:
        body["system"] = system
    if fmt:
        body["format"] = fmt

    r = httpx.post(f"{HOST}/api/generate", json=body, timeout=timeout)
    r.raise_for_status()
    d = r.json()

    eval_count = d.get("eval_count", 0)
    eval_ns = d.get("eval_duration", 1) or 1
    stats = {
        "prompt_tokens": d.get("prompt_eval_count", 0),
        "output_tokens": eval_count,
        "tokens_per_sec": eval_count / (eval_ns / 1e9),
        "total_sec": d.get("total_duration", 0) / 1e9,
    }
    return d["response"], stats


def embed(texts, model=EMBED_MODEL, timeout=600, kind="document", title=None):
    """Embed a list of strings. Returns a list of float vectors.

    EmbeddingGemma was trained with task prefixes and is measurably worse
    without them: a question and the passage answering it are not the same kind
    of text, and the prefix tells the model which role it is encoding.
    """
    if kind == "query":
        texts = [f"task: search result | query: {t}" for t in texts]
    else:
        texts = [f"title: {title or 'none'} | text: {t}" for t in texts]

    r = httpx.post(f"{HOST}/api/embed",
                   json={"model": model, "input": texts}, timeout=timeout)
    r.raise_for_status()
    return r.json()["embeddings"]


def count_tokens(text, model=GEN_MODEL):
    """Token count for a string, straight from the model's own tokenizer.

    Asks for zero output tokens so only the prompt is tokenised — cheap enough
    to call in a loop.
    """
    r = httpx.post(f"{HOST}/api/generate",
                   json={"model": model, "prompt": text, "stream": False,
                         "options": {"num_predict": 0}}, timeout=600)
    r.raise_for_status()
    return r.json().get("prompt_eval_count", 0)
