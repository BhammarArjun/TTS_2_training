#!/usr/bin/env python3
"""
Extend F5-TTS pretrained model embeddings for new vocabulary characters.

Standalone script — does NOT import from finetune_gradio.py (which drags in
gradio, transformers, torchvision, and 20+ other heavy dependencies).

What it does:
  1. Loads the pretrained F5TTS_v1_Base checkpoint (2545 vocab entries)
  2. Compares with your custom vocab.txt to find missing characters
  3. Expands the text embedding layer with random init for new chars
  4. Saves extended checkpoint + extended vocab.txt

Usage:
  python extend_embeddings.py

Output:
  ckpts/f5_gu_hi_extended/pretrained_model_1250000.safetensors
  ckpts/f5_gu_hi_extended/vocab.txt
"""

import os
import random
import torch
from safetensors.torch import load_file, save_file


def expand_model_embeddings(ckpt_path, new_ckpt_path, num_new_tokens=42):
    """Expand the text embedding layer in an F5-TTS checkpoint."""
    # Deterministic init for reproducibility
    seed = 666
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Load checkpoint
    if ckpt_path.endswith(".safetensors"):
        ckpt = load_file(ckpt_path, device="cpu")
        ckpt = {"ema_model_state_dict": ckpt}
    elif ckpt_path.endswith(".pt"):
        ckpt = torch.load(ckpt_path, map_location="cpu")
    else:
        raise ValueError(f"Unsupported checkpoint format: {ckpt_path}")

    # Find and expand embedding
    ema_sd = ckpt.get("ema_model_state_dict", {})
    embed_key = "ema_model.transformer.text_embed.text_embed.weight"

    if embed_key not in ema_sd:
        raise KeyError(f"Embedding key '{embed_key}' not found in checkpoint. "
                       f"Available keys: {[k for k in ema_sd.keys() if 'embed' in k]}")

    old_embed = ema_sd[embed_key]
    vocab_old = old_embed.size(0)
    embed_dim = old_embed.size(1)
    vocab_new = vocab_old + num_new_tokens

    print(f"  Old embedding: [{vocab_old}, {embed_dim}]")
    print(f"  Adding {num_new_tokens} new tokens")
    print(f"  New embedding: [{vocab_new}, {embed_dim}]")

    # Create expanded embedding
    new_embed = torch.zeros((vocab_new, embed_dim))
    new_embed[:vocab_old] = old_embed
    new_embed[vocab_old:] = torch.randn((num_new_tokens, embed_dim))
    ema_sd[embed_key] = new_embed

    # Save
    os.makedirs(os.path.dirname(new_ckpt_path), exist_ok=True)

    if new_ckpt_path.endswith(".safetensors"):
        save_file(ema_sd, new_ckpt_path)
    elif new_ckpt_path.endswith(".pt"):
        torch.save(ckpt, new_ckpt_path)

    return vocab_new


def main():
    # ── Paths — edit these to match your setup ──
    f5_root = os.path.expanduser("~/F5-TTS")

    original_ckpt = os.path.join(f5_root, "ckpts/F5TTS_v1_Base/model_1250000.safetensors")
    original_vocab_path = os.path.join(f5_root, "ckpts/F5TTS_v1_Base/vocab.txt")
    new_vocab_path = os.path.join(f5_root, "data/f5-gu_custom/vocab.txt")  # your generated vocab

    output_dir = os.path.join(f5_root, "ckpts/f5_gu_hi_extended")
    output_ckpt = os.path.join(output_dir, "pretrained_model_1250000.safetensors")
    output_vocab = os.path.join(output_dir, "vocab.txt")

    # ── Verify files exist ──
    for path, name in [(original_ckpt, "Pretrained checkpoint"),
                       (original_vocab_path, "Pretrained vocab"),
                       (new_vocab_path, "Your vocab")]:
        if not os.path.exists(path):
            print(f"ERROR: {name} not found: {path}")
            print("  Download pretrained model first:")
            print("  python -c \"from huggingface_hub import hf_hub_download; "
                  "hf_hub_download('SWivid/F5-TTS', 'F5TTS_v1_Base/model_1250000.safetensors', local_dir='ckpts'); "
                  "hf_hub_download('SWivid/F5-TTS', 'F5TTS_v1_Base/vocab.txt', local_dir='ckpts')\"")
            return

    # ── Load vocabs ──
    with open(original_vocab_path, 'r', encoding='utf-8') as f:
        original_chars = set(f.read().strip().split('\n'))

    with open(new_vocab_path, 'r', encoding='utf-8') as f:
        new_chars = set(f.read().strip().split('\n'))

    missing = new_chars - original_chars

    print(f"Original vocab size: {len(original_chars)}")
    print(f"Your vocab size:     {len(new_chars)}")
    print(f"Missing characters:  {len(missing)}")

    if not missing:
        print("\n✅ All characters already in vocab — no extension needed!")
        return

    print(f"\nMissing chars sample: {sorted(missing)[:20]}")

    # ── Extend model ──
    print(f"\nExtending model embeddings...")
    new_size = expand_model_embeddings(original_ckpt, output_ckpt, num_new_tokens=len(missing))
    print(f"✅ Extended model saved: {output_ckpt}")

    # ── Create extended vocab (original order + new chars appended) ──
    with open(original_vocab_path, 'r', encoding='utf-8') as f:
        original_vocab_list = [line.rstrip('\n') for line in f]

    extended_vocab = original_vocab_list + sorted(missing)

    with open(output_vocab, 'w', encoding='utf-8') as f:
        for char in extended_vocab:
            f.write(char + "\n")

    print(f"✅ Extended vocab saved: {output_vocab} ({len(extended_vocab)} chars)")
    print(f"\nNew embedding size: {new_size}")
    print(f"Use these for training:")
    print(f"  --pretrain \"{output_ckpt}\"")
    print(f"  --tokenizer_path \"{output_vocab}\"")


if __name__ == "__main__":
    main()
