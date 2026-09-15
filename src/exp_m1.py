"""M1 experiments: token cost across languages, and temperature's effect.

    python src/exp_m1.py
"""
import ollama

PAIRS = [
    ("English", "Plants make their own food using sunlight, water and carbon dioxide."),
    ("Hindi", "पौधे सूर्य के प्रकाश, पानी और कार्बन डाइऑक्साइड का उपयोग करके अपना भोजन स्वयं बनाते हैं।"),
    ("Hinglish", "Paudhe sunlight, paani aur carbon dioxide use karke apna khana khud banate hain."),
]

QUESTION = "In one sentence, why do leaves look green?"


def token_costs():
    print("=== Token cost of the same sentence in three languages ===")
    base = None
    for lang, text in PAIRS:
        n = ollama.count_tokens(text)
        if base is None:
            base = n
        print(f"{lang:9s} {len(text):4d} chars  {n:4d} tokens  "
              f"{n / base:.2f}x English")
    print("\nWhy it matters: you pay per token. Hindi costs more tokens for the\n"
          "same meaning, so the same answer is more expensive and fills the\n"
          "context window faster.\n")


def temperature_sweep():
    for temp in (0.0, 1.0):
        print(f"=== Same question, 5 times, temperature={temp} ===")
        seen = set()
        for i in range(5):
            text, _ = ollama.generate(QUESTION, temperature=temp)
            text = " ".join(text.split())
            seen.add(text)
            print(f"  {i + 1}. {text[:150]}")
        print(f"  -> {len(seen)} distinct answers out of 5\n")


if __name__ == "__main__":
    token_costs()
    temperature_sweep()
