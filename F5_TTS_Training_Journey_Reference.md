# F5-TTS Fine-Tuning for Gujarati — Complete Training Journal

**Project:** Fine-tuning F5-TTS v1 Base for Gujarati (gu) TTS
**Machine:** Lightning.ai L4 GPU Studio (24 GB VRAM, 31 GB RAM, 8 CPUs)
**Date:** March 22, 2026
**Repo:** `https://github.com/SWivid/F5-TTS.git` (official)
**Dataset:** `Arjun4707/gu-hi-tts` (private HuggingFace repo, ~65,700 rows)
**Trained Model:** `Arjun4707/F5-TTS-Gujarati` (private HuggingFace repo)

---

## Table of Contents

1. [Environment Setup](#1-environment-setup)
2. [Data Download & Preparation](#2-data-download--preparation)
3. [Vocabulary & Embedding Extension](#3-vocabulary--embedding-extension)
4. [Training — Issues & Solutions](#4-training--issues--solutions)
5. [Final Working Training Command](#5-final-working-training-command)
6. [Inference](#6-inference)
7. [Key Lessons for Future Projects](#7-key-lessons-for-future-projects)
8. [Dataset Structure Reference](#8-dataset-structure-reference)
9. [Hyperparameter Reference](#9-hyperparameter-reference)
10. [Key Differences: F5-TTS vs XTTS v2](#10-key-differences-f5-tts-vs-xtts-v2)

---

## 1. Environment Setup

### 1.1 Python Version

F5-TTS supports Python >= 3.10. Lightning.ai comes with 3.12. **No Conda downgrade needed** (unlike XTTS which required 3.10).

```bash
python --version  # 3.12.x — works fine
```

### 1.2 Clone Repo & Install Dependencies

```bash
cd ~
git clone https://github.com/SWivid/F5-TTS.git
cd F5-TTS
pip install -e .
pip install datasets soundfile matplotlib pandas psutil tensorboard
```

### 1.3 FFmpeg & HuggingFace Login

```bash
conda install ffmpeg -y
huggingface-cli login
# Paste HF token when prompted
```

### 1.4 Configure Accelerate

```bash
accelerate config
```

Answers for L4 single GPU:
- Machine type: No distributed training
- CPU only: NO
- torch dynamo: NO
- DeepSpeed: NO
- GPU(s): all
- NUMA efficiency: NO
- Mixed precision: **bf16**

### 1.5 Dependency Fixes (Lightning.ai Specific)

The `pip install -e .` and subsequent installs can cause cascading version conflicts:

```bash
# Fix numpy 2.x breaking scipy/sklearn
pip install "numpy<2" scipy scikit-learn --force-reinstall

# Fix torchvision::nms operator error
pip install torchvision --force-reinstall

# Fix pkg_resources missing (setuptools 82+ removed it)
pip install "setuptools<81"
```

**Order matters.** The `--force-reinstall` on one package can break others. Run all three, then verify:

```bash
python -c "import numpy; print('numpy', numpy.__version__)"
python -c "import scipy; print('scipy OK')"
python -c "import transformers; print('transformers OK')"
python -c "import torchvision; print('torchvision OK')"
python -c "from accelerate import Accelerator; print('accelerate OK')"
```

---

## 2. Data Download & Preparation

### 2.1 Dataset Schema (HuggingFace Parquet)

| Column         | Type                                          | Description                    |
|----------------|-----------------------------------------------|--------------------------------|
| `id`           | string                                        | Unique clip ID                 |
| `audio`        | struct{bytes: binary, sampling_rate: int32}   | Inline WAV bytes @ 24kHz      |
| `text`         | string                                        | Transcript                     |
| `language`     | string                                        | `gu` / `hi` / `en`            |
| `duration_sec` | float32                                       | Clip duration in seconds       |

Audio: 24kHz, mono, PCM-16 — **F5-TTS native format, no resampling needed.**

### 2.2 Data Preparation Script

Created `download_and_prepare_f5.py` which:

1. Downloads from HuggingFace with **local disk caching** (`hf_cache_f5/` directory)
2. **Filters by duration:** 5s <= duration <= 20s
3. **Filters by CPS (Characters Per Second):** 4 <= CPS <= 25 (removes noisy/silent clips)
4. **Cleans text:** removes `>` and `|` characters
5. **Runs analysis:** duration/character length/CPS per language, generates plots
6. Splits per language (95% train / 5% eval)
7. Saves to F5-TTS expected format with **absolute paths** and **header line**

### 2.3 CSV Format (F5-TTS Expected for prepare_csv_wavs.py)

```
audio_file|text
/absolute/path/to/wavs/clip001.wav|ગુજરાતી ટેક્સ્ટ અહીં
```

**Critical differences from XTTS:**
- Header line `audio_file|text` is **required**
- Paths must be **absolute** (not relative)
- Only 2 columns (no speaker_name column)
- No quoting

### 2.4 CPS (Characters Per Second) Filtering

Novel filtering approach not used in XTTS training:

| CPS Range | Meaning | Action |
|-----------|---------|--------|
| < 4 | Too few characters for duration — noise/silence | Reject |
| 4-25 | Normal speech density | Keep |
| > 25 | Too many characters for duration — bad transcription | Reject |

### 2.5 Data Sizes (After Filtering)

| Language | Clips | Hours | Disk Size |
|----------|-------|-------|-----------|
| Gujarati | ~36,155 | ~40+ hrs | ~19 GB |
| Hindi    | ~11,000 | ~12 hrs  | ~5.4 GB |
| HF Cache |         |          | ~50 GB  |

### 2.6 Converting to Arrow Format

F5-TTS training requires `raw.arrow` + `duration.json`. Used `prepare_csv_wavs.py`:

```bash
python src/f5_tts/train/datasets/prepare_csv_wavs.py \
    data/f5-gu_custom/metadata.csv \
    data/f5-gu_custom \
    --pretrain
```

### ISSUE: Pinyin Conversion Corrupts Indic Text

**Problem:** `prepare_csv_wavs.py` runs `convert_char_to_pinyin()` on ALL text by default. This is designed for Chinese and **destroys Devanagari/Gujarati characters.**

**Fix:** Patch `batch_convert_texts()` in `prepare_csv_wavs.py`:

```python
# ORIGINAL (mangles non-Chinese text):
def batch_convert_texts(texts, polyphone, batch_size=BATCH_SIZE):
    converted_texts = []
    for i in tqdm(...):
        batch = texts[i : i + batch_size]
        converted_batch = convert_char_to_pinyin(batch, polyphone=polyphone)
        converted_texts.extend(converted_batch)
    return converted_texts

# PATCHED (pass-through for Indic languages):
def batch_convert_texts(texts, polyphone, batch_size=BATCH_SIZE):
    print("Using raw character text (pinyin conversion SKIPPED for Indic languages)")
    return texts
```

### ISSUE: `--pretrain` Flag Naming Confusion

**Problem:** The flag `--pretrain` sounds like "use pretrained model" but actually means the opposite.

**Reality:**
- `--pretrain` = write fresh vocab from YOUR data (what we want)
- Without flag (default) = copy pretrained ZH+EN vocab (wrong for new languages)

### ISSUE: Folder Naming Must Match

The `download_and_prepare_f5.py` script outputs to `data/f5-gu_char/` but training expects `data/f5-gu_custom/` (because `{dataset_name}_{tokenizer}`).

**Fix:** Rename the folder AND update paths inside metadata.csv:

```bash
mv data/f5-gu_char data/f5-gu_custom
sed -i 's|f5-gu_char|f5-gu_custom|g' data/f5-gu_custom/metadata.csv
```

---

## 3. Vocabulary & Embedding Extension

### 3.1 How F5-TTS Vocab Works

- `vocab.txt` = one character per line
- Character at line N = embedding index N
- **Space MUST be at index 0** (line 1)
- Pretrained model has ~2545 characters (Chinese pinyin + English + symbols)
- Gujarati/Hindi characters are NOT in the pretrained vocab

### 3.2 Tokenizer Types

| Tokenizer | Usage | Path Resolution |
|-----------|-------|----------------|
| `pinyin` | Chinese + English (default) | `data/{dataset_name}_pinyin/vocab.txt` |
| `char` | Character-level | `data/{dataset_name}_char/vocab.txt` |
| `custom` | Direct file path | `--tokenizer_path` argument directly |

**For Indic languages: use `custom`** — it's the only one that doesn't append suffixes or run pinyin conversion internally.

### 3.3 Generate Combined Vocab

Extract characters from both languages, sort, ensure space at index 0, write to vocab files.

The vocab used for training must be the **extended** one (original 2545 + new Indic chars), NOT the 170-char dataset-only vocab. The model's embedding layer must match.

### 3.4 Extend Pretrained Model Embeddings

**Cannot import `expand_model_embeddings` from `finetune_gradio.py`** — it pulls in gradio, transformers, torchvision and causes import crashes.

**Solution:** Standalone script using only `torch` + `safetensors`:

```python
from safetensors.torch import load_file, save_file

# Load pretrained checkpoint
ckpt = load_file("ckpts/F5TTS_v1_Base/model_1250000.safetensors", device="cpu")
ckpt = {"ema_model_state_dict": ckpt}

# Find embedding layer
embed_key = "ema_model.transformer.text_embed.text_embed.weight"
old_embed = ckpt["ema_model_state_dict"][embed_key]  # [2546, 512]

# Extend with random init for new chars
new_embed = torch.zeros((2546 + num_new_tokens, 512))
new_embed[:2546] = old_embed
new_embed[2546:] = torch.randn((num_new_tokens, 512))
ckpt["ema_model_state_dict"][embed_key] = new_embed

# Save extended vocab = original vocab lines + new chars appended
```

**Result:** Extended checkpoint at `ckpts/f5_gu_hi_extended/pretrained_model_1250000.safetensors` with vocab size 2645 (2545 original + 100 new Gujarati/Hindi chars).

### 3.5 Extended Vocab.txt Structure

```
 (space — index 0, MUST be first)
... (original 2545 chars from pretrained model)
અ (new Gujarati char — index 2546)
આ (new Gujarati char — index 2547)
... etc
```

**The vocab used for training (`--tokenizer_path`) must be this extended one, NOT the 170-char dataset-only vocab.**

---

## 4. Training — Issues & Solutions

### ISSUE 1: `--exp_name` Must Be a Valid Model Name

**Error:** `unrecognized arguments` or model config not found.

**Cause:** `--exp_name` is NOT a project name. It must be one of:
- `F5TTS_v1_Base` (recommended — latest model)
- `F5TTS_Base` (older v0)
- `E2TTS_Base`

**Fix:** Always use `--exp_name F5TTS_v1_Base`.

### ISSUE 2: `--last_per_steps` vs `--last_per_updates`

**Error:** `unrecognized arguments: --last_per_steps`

**Fix:** The correct argument is `--last_per_updates` (not `--last_per_steps`).

### ISSUE 3: torchvision::nms Operator Error

**Error:**
```
RuntimeError: operator torchvision::nms does not exist
```

**Cause:** `--log_samples` triggers import of inference code -> transformers -> torchvision, and torchvision is incompatible with the installed PyTorch version.

**Fix:**
```bash
pip install torchvision --force-reinstall
```

### ISSUE 4: NumPy 2.x Breaking Everything

**Error:**
```
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.4.3
```

**Cause:** `pip install torchvision --force-reinstall` pulled in numpy 2.4 which broke scipy/sklearn.

**Fix:**
```bash
pip install "numpy<2" scipy scikit-learn torchvision transformers --force-reinstall
```

### ISSUE 5: Embedding Size Mismatch

**Error:**
```
size mismatch for ema_model.transformer.text_embed.text_embed.weight:
copying a param with shape torch.Size([2546, 512]) from checkpoint,
the shape in current model is torch.Size([2646, 512]).
```

**Cause:** A previous failed run (without `--pretrain`) copied the original non-extended model into `ckpts/f5-gu/`. The code checks "does checkpoint exist?" — if yes, it skips copying the new one.

**Fix:**
```bash
rm -rf ckpts/f5-gu/
# Then re-run training — it will copy the extended checkpoint fresh
```

### ISSUE 6: DataLoader Worker Warning

**Warning:**
```
This DataLoader will create 16 worker processes in total.
Our suggested max number of worker in current system is 8
```

**This is harmless.** Training runs fine. To silence it, modify Trainer code to set `num_workers=8`.

### ISSUE 7: Subprocess Unicode Corruption for Inference

**Problem:** Running `f5-tts_infer-cli` via subprocess mangles Gujarati characters.

**Fix:** Use the Python API directly instead of CLI:
```python
from f5_tts.api import F5TTS
tts = F5TTS(model="F5TTS_v1_Base", ckpt_file=..., vocab_file=...)
tts.infer(ref_file=..., ref_text=..., gen_text=..., file_wave=...)
```

---

## 5. Final Working Training Command

### 5.1 All Patches Applied

| File | Change |
|------|--------|
| `src/f5_tts/train/datasets/prepare_csv_wavs.py` | `batch_convert_texts()` -> pass-through (skip pinyin) |

### 5.2 Training Command (L4 GPU)

```bash
accelerate launch --mixed_precision=bf16 \
    src/f5_tts/train/finetune_cli.py \
    --exp_name F5TTS_v1_Base \
    --dataset_name "f5-gu" \
    --learning_rate 1e-5 \
    --batch_size_per_gpu 3200 \
    --batch_size_type frame \
    --max_samples 64 \
    --grad_accumulation_steps 2 \
    --max_grad_norm 1.0 \
    --epochs 100 \
    --num_warmup_updates 1000 \
    --save_per_updates 5000 \
    --last_per_updates 1000 \
    --finetune \
    --pretrain "ckpts/f5_gu_hi_extended/pretrained_model_1250000.safetensors" \
    --tokenizer custom \
    --tokenizer_path "ckpts/f5_gu_hi_extended/vocab.txt" \
    --log_samples \
    --logger tensorboard
```

### 5.3 Resource Usage

| Metric | Value |
|--------|-------|
| GPU VRAM Used | ~18-20 GB |
| GPU VRAM Total | 24 GB |
| System RAM | ~4-6 GB |
| Batch Size | 3200 frames |
| Effective Batch | 3200 x 2 (grad_accum) = 6400 frames |
| Steps per Epoch | ~7,512 |
| Step Time | ~0.47s (~2.1 updates/sec) |
| Time per Epoch | ~60 minutes |
| Training Stopped | Epoch 21 (~150K steps, ~21 hours) |

### 5.4 Training Metrics

| Step | Loss | Notes |
|------|------|-------|
| 36 | 0.516 | Starting |
| 1000 | 0.634 | Warmup phase (slight bump normal) |
| 1900 | 0.469 | Dropping |
| ~2860 | 0.4-0.9 | Noisy per-batch (normal for flow matching) |
| 5000 | ~0.5 | First checkpoint + audio samples |
| 6415 | 0.495 | Steady |
| ~150K | ~0.4-0.5 | Stopped here — good single-speaker quality |

**Note:** F5-TTS loss is inherently noisier than XTTS because of random timestep sampling per batch in flow matching training. Judge quality by listening to samples, not loss numbers.

### 5.5 Saved Checkpoints

```
ckpts/f5-gu/
├── pretrained_model_1250000.safetensors  (extended base — auto-copied)
├── model_25000.pt through model_150000.pt (every 5K steps)
├── model_last.pt (every 1K steps, overwritten)
└── samples/ (audio generated at each 5K checkpoint)
```

---

## 6. Inference

### 6.1 Files Needed

Only two files required:
- `model_150000.pt` — trained checkpoint
- `vocab.txt` — extended vocabulary (2645 chars)

Both uploaded to `Arjun4707/F5-TTS-Gujarati` (private HuggingFace repo).

### 6.2 Model Config

```json
{"dim": 1024, "depth": 22, "heads": 16, "ff_mult": 2, "text_dim": 512, "conv_layers": 4}
```

### 6.3 Python API (Recommended)

```python
from f5_tts.api import F5TTS

tts = F5TTS(
    model="F5TTS_v1_Base",
    ckpt_file="model_150000.pt",
    vocab_file="vocab.txt",
)

tts.infer(
    ref_file="reference_audio.wav",       # 5-10s clean single-speaker
    ref_text="exact transcript of ref audio",
    gen_text="જે ટેક્સ્ટ જનરેટ કરવો છે તે અહીં લખો",
    file_wave="output.wav",
    speed=0.8,      # 1.0 default, lower = slower
    nfe_step=64,    # 32 default, higher = better quality but slower
)
```

### 6.4 Inference Parameters

| Parameter | Default | Recommended | Notes |
|-----------|---------|-------------|-------|
| `speed` | 1.0 | 0.8 | Model output slightly fast, 0.8 feels natural |
| `nfe_step` | 32 | 64 | More denoising steps = better quality, ~2x slower |
| `remove_silence` | False | False | Optional silence trimming |
| `seed` | None | -1 for random | Set for reproducibility |

### 6.5 Cross-Lingual Voice Cloning

English reference audio + Gujarati generation text works:

```python
tts.infer(
    ref_file="english_speaker.wav",
    ref_text="The exact English words spoken in that clip",
    gen_text="ગુજરાતી ટેક્સ્ટ અહીં",
    file_wave="cross_lingual.wav",
    speed=0.8,
)
```

### 6.6 CLI Inference (Avoid for Indic Scripts)

```bash
f5-tts_infer-cli --model F5TTS_v1_Base \
    --ckpt_file model_150000.pt \
    --vocab_file vocab.txt \
    --ref_audio ref.wav --ref_text "..." --gen_text "..."
```

**Warning: CLI has Unicode issues with Indic scripts via subprocess.** Use Python API instead.

---

## 7. Key Lessons for Future Projects

### 7.1 Environment & Dependencies

1. **F5-TTS works with Python 3.10-3.12.** No Conda downgrade needed (unlike XTTS).
2. **Pin `numpy<2`** on Lightning.ai — scipy/sklearn break with numpy 2.x.
3. **Pin `setuptools<81`** — tensorboard needs `pkg_resources` which was removed in 82+.
4. **Reinstall torchvision** if you see `torchvision::nms` errors.
5. **Fix all dependency issues at once** — cascading reinstalls break other packages.

### 7.2 Data Preparation

1. **No resampling needed** — F5-TTS is 24kHz native, matching the dataset.
2. **CPS filtering is essential** — characters-per-second catches noisy clips that duration filtering misses.
3. **Speaker diarization BEFORE training** — multi-speaker clips cause stopping/blabbering. Use pyannote-audio to filter. Quality >> quantity.
4. **Patch pinyin conversion** — `prepare_csv_wavs.py` runs Chinese pinyin on ALL text. Must bypass for Indic languages.
5. **Absolute paths in CSV** — F5-TTS expects absolute paths (XTTS used relative).
6. **Header line required** — `audio_file|text` as first line.
7. **`--pretrain` flag means "generate fresh vocab"** — confusing name, but needed for new languages.

### 7.3 Folder Naming & Path Gotchas

1. **`dataset.py` appends `_{tokenizer}` to dataset_name.** If `--dataset_name f5-gu --tokenizer custom`, it looks for `data/f5-gu_custom/`. Get this wrong = FileNotFoundError.
2. **`--exp_name` is NOT a project name.** Must be `F5TTS_v1_Base`, `F5TTS_Base`, or `E2TTS_Base`. Controls model architecture.
3. **Checkpoints save to `ckpts/{dataset_name}/`** — not `ckpts/{exp_name}/`.
4. **Stale checkpoints block new ones.** If a failed run left a checkpoint in `ckpts/{dataset_name}/`, the code skips copying your extended model. Delete the folder and retry.
5. **When renaming folders, update CSV contents too.** Absolute paths inside `metadata.csv` still point to the old name.

### 7.4 Training Configuration

1. **`batch_size_per_gpu = 3200` for L4 (24GB).** Source code comments confirm this. Can push to ~9000 once stable.
2. **Frame-based batching is fundamentally different from XTTS.** 3200 = total mel frames per batch, not clip count.
3. **Loss is noisy — that's normal.** Flow matching randomly samples timesteps, so per-batch loss varies 0.4-0.9. Judge by audio quality, not loss numbers.
4. **21 epochs (~150K steps) was enough** for good single-speaker Gujarati with 36K clips.
5. **`--log_samples` generates audio at checkpoint saves** — the real quality metric.
6. **Resume by re-running the same command** — auto-loads from last checkpoint.

### 7.5 Vocab & Embeddings

1. **Use extended vocab for training, not dataset-only vocab.** 170-char Gujarati vocab causes embedding size mismatch. Must use 2645-char extended vocab.
2. **Space at index 0** — F5-TTS convention for padding/unknown.
3. **Extended vocab = original pretrained chars + new chars appended.** Order matters.
4. **Don't import from `finetune_gradio.py`** — it drags in the entire dependency tree. Copy the function standalone.

### 7.6 Multi-Speaker Data Issue

**Problem discovered:** Podcast data has clips where speaker A is talking and speaker B starts. Model gets confused — reference voice is speaker A but text corresponds to speaker B.

**Symptoms:** Good quality on single-speaker clips. Stopping/blabbering on multi-speaker clips.

**Solution for v2:** Run speaker diarization (pyannote-audio) to identify and remove multi-speaker clips before training. Quality >> quantity.

### 7.7 How F5-TTS Training Works Internally

1. WAV -> mel spectrogram (24kHz / 256 hop = 93.75 frames/sec)
2. Text -> character-level tokenization via vocab.txt
3. Same audio clip is split: random portion kept as "reference", rest masked with noise
4. Model learns to predict denoising direction for a **single random timestep** per sample
5. Full text always provided (no splitting) — model learns alignment internally
6. At inference: 32 or 64 iterative denoising steps from pure noise -> clean mel -> Vocos vocoder -> audio
7. Training = 1 timestep per sample (fast). Inference = 32-64 steps (slower).

---

## 8. Dataset Structure Reference

### F5-TTS Expected Directory Layout

```
~/F5-TTS/
├── data/
│   ├── f5-gu_custom/
│   │   ├── wavs/               (WAV files at 24kHz mono)
│   │   ├── metadata.csv        (audio_file|text, header + absolute paths)
│   │   ├── vocab.txt           (generated by prepare_csv_wavs.py --pretrain)
│   │   ├── raw.arrow           (generated by prepare_csv_wavs.py)
│   │   └── duration.json       (generated by prepare_csv_wavs.py)
│   └── Emilia_ZH_EN_pinyin/    (pretrained vocab — comes with repo)
│       └── vocab.txt
├── ckpts/
│   ├── F5TTS_v1_Base/
│   │   ├── model_1250000.safetensors  (auto-downloaded pretrained)
│   │   └── vocab.txt                  (original ZH+EN vocab)
│   ├── f5_gu_hi_extended/
│   │   ├── pretrained_model_1250000.safetensors  (extended embeddings)
│   │   └── vocab.txt                             (extended: 2645 chars)
│   └── f5-gu/                   (training outputs)
│       ├── pretrained_model_1250000.safetensors  (auto-copied by finetune_cli)
│       ├── model_25000.pt ... model_150000.pt
│       ├── model_last.pt
│       └── samples/
├── analysis_f5/
│   ├── f5_data_analysis.json
│   └── analysis_f5_gu.png
├── hf_cache_f5/                 (HuggingFace download cache — ~50GB)
├── download_and_prepare_f5.py
├── generate_vocab.py
├── extend_embeddings.py
└── run_inference.py
```

---

## 9. Hyperparameter Reference

### L4 GPU (24 GB VRAM) — Settings Used

| Parameter | Value | Notes |
|-----------|-------|-------|
| batch_size_per_gpu | 3200 | Conservative start; can push to ~9000 |
| batch_size_type | frame | Packs clips until frame budget reached |
| max_samples | 64 | Max clips per batch |
| grad_accumulation_steps | 2 | Effective batch = 6400 frames |
| max_grad_norm | 1.0 | Standard gradient clipping |
| learning_rate | 1e-5 | Recommended for fine-tuning |
| num_warmup_updates | 1000 | Warmup steps |
| epochs | 100 (stopped at 21) | Monitor samples, stop when quality plateaus |
| save_per_updates | 5000 | Checkpoint + sample audio every 5K steps |
| last_per_updates | 1000 | Resume checkpoint every 1K steps |
| mixed_precision | bf16 | L4 supports bf16 well |
| tokenizer | custom | Required for Indic languages |

### Batch Size Guidance from Source Code

```
batch_size_per_gpu = 1000   for 8GB GPU
batch_size_per_gpu = 1600   for 12GB GPU
batch_size_per_gpu = 2000   for 16GB GPU
batch_size_per_gpu = 3200   for 24GB GPU
```

Auto-calculator formula: `batch_size = 38400 * (gpu_memory_gb - 5) / 75`
For L4 (24GB): `38400 * 19 / 75 = 9728` (upper limit)

### Community Training References

| Language | Data | Steps | GPU | Notes |
|----------|------|-------|-----|-------|
| Finnish | ~4 days speech | ~200 epochs | 4070 Ti Super 16GB | v1 Base, bf16 |
| Korean | ~12 hrs single speaker | 1000 epochs | RTX 3080 10GB | Base, bf16 |
| Polish | 142 hrs | ~350K steps | RTX 4090 | v1 Base |
| Hindi (SPRINGLab) | IndicTTS+IndicVoices-R | 2.5M steps | - | Small config (151M) |
| **Gujarati (ours)** | **36K clips (~40 hrs)** | **150K steps** | **L4 24GB** | **v1 Base, 3200 frames, bf16** |

---

## 10. Key Differences: F5-TTS vs XTTS v2

| Aspect | XTTS v2 | F5-TTS |
|--------|---------|--------|
| Architecture | GPT autoregressive | Flow Matching (DiT) |
| Sample rate | 22,050 Hz | **24,000 Hz** |
| Python | 3.10 only | >= 3.10 |
| Tokenizer | BPE vocab | Character-level vocab.txt |
| Batch sizing | Fixed clip count | Frame-based (variable clips) |
| Training launcher | `python train_gpt_xtts.py` | `accelerate launch finetune_cli.py` |
| Loss behavior | Smooth, steadily decreasing | Noisy (random timestep sampling) |
| Data format | Relative paths, pipe-delimited | **Absolute paths**, Arrow format |
| Vocab extension | `extend_vocab_config.py` | `expand_model_embeddings()` standalone |
| Audio loading | Patched torchaudio -> soundfile | torchaudio (works as-is) |
| Training speed | ~0.2s/step | ~0.47s/step |
| Quality judge | Loss curves reliable | **Listen to samples** |

---

## Appendix: Complete Execution Order

```
1.  git clone https://github.com/SWivid/F5-TTS.git && cd F5-TTS
2.  pip install -e .
3.  pip install datasets soundfile matplotlib pandas psutil tensorboard
4.  pip install "numpy<2" scipy scikit-learn torchvision transformers --force-reinstall
5.  pip install "setuptools<81"
6.  conda install ffmpeg -y
7.  huggingface-cli login
8.  accelerate config                              # bf16, single GPU
9.  python download_and_prepare_f5.py              # Download + filter + save WAVs
10. mv data/f5-gu_char data/f5-gu_custom           # Fix folder naming
11. sed -i 's|f5-gu_char|f5-gu_custom|g' data/f5-gu_custom/metadata.csv
12. ** PATCH prepare_csv_wavs.py **                # Skip pinyin conversion
13. python src/f5_tts/train/datasets/prepare_csv_wavs.py data/f5-gu_custom/metadata.csv data/f5-gu_custom --pretrain
14. python generate_vocab.py                        # Create extended vocab
15. python extend_embeddings.py                     # Extend pretrained model
16. accelerate launch --mixed_precision=bf16 src/f5_tts/train/finetune_cli.py [full args above]
17. python run_inference.py                         # Test trained model
```
