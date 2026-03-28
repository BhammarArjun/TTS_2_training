#!/usr/bin/env python3
"""
Generate vocab.txt for F5-TTS from metadata files.

- Extracts all unique characters from Gujarati + Hindi metadata
- Ensures space is at index 0 (required by F5-TTS tokenizer)
- Writes identical vocab to each language folder + a combined folder

Usage:
  python generate_vocab.py
"""

import os

# Must match FOLDER_SUFFIX in download_and_prepare_f5.py
FOLDER_SUFFIX = "_custom"
BASE_DIR = os.path.expanduser("~/F5-TTS/data")
LANGUAGES = ["gu", "hi"]


def extract_vocab(metadata_path):
    """Extract all unique characters from metadata (skipping header)."""
    chars = set()
    with open(metadata_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i == 0 and line.startswith("audio_file|"):
                continue  # skip header
            parts = line.strip().split("|", 1)
            if len(parts) == 2:
                text = parts[1]
                chars.update(text)
    return chars


def main():
    all_chars = set()

    for lang in LANGUAGES:
        meta_path = os.path.join(BASE_DIR, f"f5-{lang}{FOLDER_SUFFIX}", "metadata.csv")
        if os.path.exists(meta_path):
            chars = extract_vocab(meta_path)
            all_chars.update(chars)
            print(f"{lang.upper()}: {len(chars)} unique characters")
        else:
            print(f"{lang.upper()}: metadata.csv not found at {meta_path}, skipping")

    if not all_chars:
        print("ERROR: No characters found. Check that metadata.csv files exist.")
        return

    # Space MUST be at index 0 — F5-TTS uses idx 0 for unknown characters
    vocab_list = sorted(all_chars)
    if " " in vocab_list:
        vocab_list.remove(" ")
    vocab_list = [" "] + vocab_list

    print(f"\nTotal unique characters: {len(vocab_list)}")
    print(f"First 20: {vocab_list[:20]}")

    # Write to each language folder + combined
    targets = LANGUAGES + ["combined"]
    for target in targets:
        vocab_dir = os.path.join(BASE_DIR, f"f5-{target}{FOLDER_SUFFIX}")
        os.makedirs(vocab_dir, exist_ok=True)
        vocab_path = os.path.join(vocab_dir, "vocab.txt")
        with open(vocab_path, 'w', encoding='utf-8') as f:
            for char in vocab_list:
                f.write(char + "\n")
        print(f"Wrote {vocab_path} ({len(vocab_list)} entries)")


if __name__ == "__main__":
    main()
