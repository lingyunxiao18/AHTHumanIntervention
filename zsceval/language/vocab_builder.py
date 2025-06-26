# vocab_builder.py

import json
from typing import List, Set

# adjust this import if your MAX_LEN lives elsewhere
from language_conditioned_policy import MAX_LEN  

def build_vocab(sample_texts: List[str]) -> dict:
    """
    Build a whitespace-based vocab mapping from a list of sample strings.
    Reserves 0:<pad>, 1:<unk>, then assigns indices to each unique token.
    """
    unique_tokens: Set[str] = set()
    for text in sample_texts:
        for tok in text.lower().split():
            unique_tokens.add(tok)
    vocab = {"<pad>": 0, "<unk>": 1}
    for idx, tok in enumerate(sorted(unique_tokens), start=2):
        vocab[tok] = idx
    return vocab

if __name__ == "__main__":
    # === Example samples ===
    # (Replace these with a few representative prompts and commands.)
    sample_prompts = [
        "=== Environment ===\nKitchen layout: simple_layout\nPot 0 at (2,3), cooking nothing, time left 0s\nAgent0 at (1,1), holding nothing",
        "=== Agents ===\nAgent1 at (4,0), holding onion",
        # … add 3–5 more variations …
    ]
    sample_commands = [
        "pick up the onion",
        "serve the soup",
        "go to pot 1",
        # … add all your 3×4 matrix commands …
    ]

    all_samples = sample_prompts + sample_commands
    vocab = build_vocab(all_samples)

    # Save to JSON so you can load it in your agent module
    with open("vocab.json", "w") as f:
        json.dump(vocab, f, indent=2)

    print(f"✅ Built vocab with {len(vocab)} tokens (including <pad> & <unk>).")
    print("Saved to vocab.json; you can now load this in your code like:")
    print("  with open('vocab.json') as f: VOCAB = json.load(f)")
