#!/usr/bin/env python3
"""
F5-TTS Data Preparation — Download from HuggingFace, filter, and save for training.

Filters:
  1. Duration: 5s ≤ duration ≤ 20s
  2. Character density (CPS): 4 ≤ chars/sec ≤ 25
  3. Text cleaning: remove >, |, and other problematic characters
  4. Minimum text length: 10 characters

Output:
  data/f5-{lang}_custom/
    wavs/         (24kHz mono WAV files)
    metadata.csv  (audio_file|text with header + absolute paths)

Usage:
  python download_and_prepare_f5.py
"""

import os
import io
import json
import soundfile as sf
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from datasets import load_dataset

# ============================================================
# CONFIGURATION — edit these to match your setup
# ============================================================
HF_REPO = "Arjun4707/gu-hi-tts"
CACHE_DIR = os.path.expanduser("~/hf_cache_f5")
OUTPUT_BASE = os.path.expanduser("~/F5-TTS")
ANALYSIS_DIR = os.path.join(OUTPUT_BASE, "analysis_f5")

# Filtering thresholds
MIN_DURATION = 5.0    # seconds
MAX_DURATION = 20.0   # seconds
MIN_CPS = 4.0         # minimum characters per second
MAX_CPS = 25.0        # maximum characters per second
MIN_TEXT_LEN = 10     # minimum characters in text

# Train/eval split
EVAL_RATIO = 0.05
SEED = 42

# Folder suffix — must match --tokenizer used during training
FOLDER_SUFFIX = "_custom"

# Checkpoint for resume
CHECKPOINT_FILE = os.path.join(OUTPUT_BASE, "f5_data_prep_checkpoint.json")


def clean_text(text):
    """Clean text: remove problematic characters."""
    if not isinstance(text, str):
        return ""
    text = text.replace(">", "")
    text = text.replace("|", " ")
    text = text.replace("\n", " ")
    text = text.replace("\r", " ")
    text = text.replace("\t", " ")
    while "  " in text:
        text = text.replace("  ", " ")
    return text.strip()


def compute_cps(text, duration):
    """Compute characters per second."""
    if duration <= 0:
        return 0
    return len(text) / duration


def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE) as f:
            return json.load(f)
    return {"phase": "not_started", "processed_rows": 0}


def save_checkpoint(data):
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(data, f, indent=2)


def run_analysis(records, lang):
    """Run duration and character length analysis for a language."""
    if not records:
        return {}

    durations = [r['duration'] for r in records]
    char_lens = [len(r['text']) for r in records]
    cps_values = [r['cps'] for r in records]

    stats = {
        "language": lang,
        "total_clips": len(records),
        "total_hours": sum(durations) / 3600,
        "duration": {
            "min": float(min(durations)),
            "max": float(max(durations)),
            "mean": float(np.mean(durations)),
            "median": float(np.median(durations)),
            "p95": float(np.percentile(durations, 95)),
            "p99": float(np.percentile(durations, 99)),
        },
        "char_length": {
            "min": int(min(char_lens)),
            "max": int(max(char_lens)),
            "mean": float(np.mean(char_lens)),
            "median": float(np.median(char_lens)),
            "p95": float(np.percentile(char_lens, 95)),
            "p99": float(np.percentile(char_lens, 99)),
        },
        "cps": {
            "min": float(min(cps_values)),
            "max": float(max(cps_values)),
            "mean": float(np.mean(cps_values)),
            "median": float(np.median(cps_values)),
        }
    }

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"F5-TTS Data Analysis — {lang.upper()}", fontsize=14)

    axes[0].hist(durations, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0].set_title(f"Duration Distribution (n={len(records)})")
    axes[0].set_xlabel("Duration (seconds)")
    axes[0].axvline(np.mean(durations), color='red', linestyle='--', label=f"mean={np.mean(durations):.1f}s")
    axes[0].legend()

    axes[1].hist(char_lens, bins=50, color='coral', edgecolor='black', alpha=0.7)
    axes[1].set_title("Character Length Distribution")
    axes[1].set_xlabel("Characters")
    axes[1].axvline(np.mean(char_lens), color='red', linestyle='--', label=f"mean={np.mean(char_lens):.0f}")
    axes[1].legend()

    axes[2].hist(cps_values, bins=50, color='mediumseagreen', edgecolor='black', alpha=0.7)
    axes[2].set_title("Characters Per Second (CPS)")
    axes[2].set_xlabel("CPS")
    axes[2].axvline(np.mean(cps_values), color='red', linestyle='--', label=f"mean={np.mean(cps_values):.1f}")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(ANALYSIS_DIR, f"analysis_f5_{lang}.png"), dpi=150)
    plt.close()

    return stats


def main():
    os.makedirs(ANALYSIS_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    # ── Step 1: Load dataset (cached) ──
    print("=" * 60)
    print("STEP 1: Loading dataset from HuggingFace (with caching)")
    print("=" * 60)

    ds = load_dataset(HF_REPO, split="train", cache_dir=CACHE_DIR)
    total_rows = len(ds)
    print(f"Total rows in dataset: {total_rows}")

    # ── Step 2: Filter & clean ──
    print("\n" + "=" * 60)
    print("STEP 2: Filtering and cleaning data")
    print("=" * 60)

    lang_records = {"gu": [], "hi": [], "en": []}
    filter_stats = {
        "total": 0, "passed": 0,
        "rejected_duration_short": 0, "rejected_duration_long": 0,
        "rejected_cps_low": 0, "rejected_cps_high": 0,
        "rejected_text_short": 0, "rejected_empty_text": 0,
    }

    for i in range(total_rows):
        row = ds[i]
        filter_stats["total"] += 1

        if i % 5000 == 0:
            print(f"  Processing row {i}/{total_rows}...")

        lang = row.get("language", "unknown")
        if lang not in lang_records:
            continue

        duration = row.get("duration_sec", 0)
        if duration < MIN_DURATION:
            filter_stats["rejected_duration_short"] += 1
            continue
        if duration > MAX_DURATION:
            filter_stats["rejected_duration_long"] += 1
            continue

        text = clean_text(row.get("text", ""))
        if not text:
            filter_stats["rejected_empty_text"] += 1
            continue
        if len(text) < MIN_TEXT_LEN:
            filter_stats["rejected_text_short"] += 1
            continue

        cps = compute_cps(text, duration)
        if cps < MIN_CPS:
            filter_stats["rejected_cps_low"] += 1
            continue
        if cps > MAX_CPS:
            filter_stats["rejected_cps_high"] += 1
            continue

        filter_stats["passed"] += 1
        lang_records[lang].append({
            "id": row["id"],
            "text": text,
            "duration": duration,
            "cps": cps,
            "audio_bytes": row["audio"]["bytes"],
            "sr": row["audio"]["sampling_rate"],
        })

    # Print filter summary
    print("\n┌─────────────────────────────────────────────┐")
    print("│          FILTERING SUMMARY                  │")
    print("├─────────────────────────────────────────────┤")
    for key, val in filter_stats.items():
        print(f"│ {key:<35} {val:>6}  │")
    print("└─────────────────────────────────────────────┘")

    for lang in ["gu", "hi", "en"]:
        count = len(lang_records[lang])
        hours = sum(r['duration'] for r in lang_records[lang]) / 3600
        print(f"  {lang.upper()}: {count} clips, {hours:.1f} hours")

    # ── Step 3: Analysis ──
    print("\n" + "=" * 60)
    print("STEP 3: Running analysis")
    print("=" * 60)

    all_stats = {}
    for lang in ["gu", "hi"]:
        if lang_records[lang]:
            stats = run_analysis(lang_records[lang], lang)
            all_stats[lang] = stats
            print(f"  {lang.upper()}: {stats['total_clips']} clips, "
                  f"{stats['total_hours']:.1f} hrs, "
                  f"mean duration={stats['duration']['mean']:.1f}s, "
                  f"mean CPS={stats['cps']['mean']:.1f}")

    with open(os.path.join(ANALYSIS_DIR, "f5_data_analysis.json"), 'w') as f:
        json.dump(all_stats, f, indent=2)
    with open(os.path.join(ANALYSIS_DIR, "f5_filter_stats.json"), 'w') as f:
        json.dump(filter_stats, f, indent=2)

    # ── Step 4: Save WAVs and metadata ──
    print("\n" + "=" * 60)
    print("STEP 4: Saving WAVs and metadata.csv for F5-TTS")
    print("=" * 60)

    for lang in ["gu", "hi"]:
        records = lang_records[lang]
        if not records:
            print(f"  Skipping {lang} — no records")
            continue

        np.random.seed(SEED)
        indices = np.random.permutation(len(records))
        eval_count = int(len(records) * EVAL_RATIO)
        eval_indices = set(indices[:eval_count])

        lang_dir = os.path.join(OUTPUT_BASE, f"data/f5-{lang}{FOLDER_SUFFIX}")
        wavs_dir = os.path.join(lang_dir, "wavs")
        os.makedirs(wavs_dir, exist_ok=True)

        train_lines = []
        eval_lines = []

        for j, rec in enumerate(records):
            if j % 2000 == 0:
                print(f"  [{lang.upper()}] Writing WAV {j}/{len(records)}...")

            wav_filename = f"{rec['id']}.wav"
            wav_path = os.path.join(wavs_dir, wav_filename)

            if not os.path.exists(wav_path):
                audio_array, _ = sf.read(io.BytesIO(rec['audio_bytes']), dtype="float32")
                sf.write(wav_path, audio_array, rec['sr'], subtype="PCM_16")

            wav_abs_path = os.path.abspath(wav_path)
            line = f"{wav_abs_path}|{rec['text']}"
            if j in eval_indices:
                eval_lines.append(line)
            else:
                train_lines.append(line)

        # F5-TTS prepare_csv_wavs.py expects: header + absolute paths
        with open(os.path.join(lang_dir, "metadata.csv"), 'w', encoding='utf-8') as f:
            f.write("audio_file|text\n")
            for line in train_lines + eval_lines:
                f.write(line + "\n")

        with open(os.path.join(lang_dir, "metadata_eval.csv"), 'w', encoding='utf-8') as f:
            f.write("audio_file|text\n")
            for line in eval_lines:
                f.write(line + "\n")

        print(f"  [{lang.upper()}] Done: {len(train_lines)} train + {len(eval_lines)} eval clips")

    save_checkpoint({"phase": "complete", "filter_stats": filter_stats})
    print("\n✅ Data preparation complete!")
    print(f"   Analysis plots: {ANALYSIS_DIR}/")
    for lang in ["gu", "hi"]:
        print(f"   {lang.upper()} data: {OUTPUT_BASE}/data/f5-{lang}{FOLDER_SUFFIX}/")


if __name__ == "__main__":
    main()
