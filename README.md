# TTS Training for Gujarati & Hindi

Training and fine-tuning Text-to-Speech models for Indian languages (Gujarati and Hindi) using XTTS v2 and F5-TTS.

## Models Trained

| Model | Language | Dataset | Steps | Quality | Checkpoint |
|-------|----------|---------|-------|---------|------------|
| **F5-TTS v1 Base** | Gujarati | ~36K clips (~40 hrs) | 150K | Good (single-speaker) | [Private HF Repo](https://huggingface.co/Arjun4707/F5-TTS-Gujarati) |
| **XTTS v2** | Gujarati + Hindi | ~51K clips (~55 hrs) | 5 epochs | Baseline | — |

## Repository Structure

```
├── README.md
├── F5_TTS_Training_Journey_Reference.md    # Complete F5-TTS training journal
├── XTTS_v2_Training_Journey_Reference.md   # Complete XTTS v2 training journal
├── download_and_prepare_f5.py              # Download HF dataset + smart filtering
├── generate_vocab.py                       # Create vocab.txt for F5-TTS
├── extend_embeddings.py                    # Extend pretrained model for new languages
└── run_inference.py                        # Generate speech from trained model
```

## Quick Start — F5-TTS Gujarati Inference

```python
from f5_tts.api import F5TTS

tts = F5TTS(
    model="F5TTS_v1_Base",
    ckpt_file="model_150000.pt",
    vocab_file="vocab.txt",
)

tts.infer(
    ref_file="reference_audio.wav",
    ref_text="exact transcript of reference audio",
    gen_text="તમે જે પણ ગુજરાતી ટેક્સ્ટ જનરેટ કરવા માંગો છો",
    file_wave="output.wav",
    speed=0.8,
    nfe_step=64,
)
```

## Training Pipeline

### Step 1: Data Preparation
```bash
python download_and_prepare_f5.py
```
Downloads from HuggingFace, applies smart filtering:
- **Duration filter:** 5–20 seconds
- **CPS filter:** 4–25 characters/second (removes noisy clips)
- **Text cleaning:** removes `>`, `|`, and other delimiter-breaking characters

### Step 2: Vocabulary
```bash
python generate_vocab.py
```
Extracts all unique Gujarati + Hindi characters, ensures space at index 0.

### Step 3: Extend Pretrained Model
```bash
python extend_embeddings.py
```
Adds new character embeddings to the pretrained F5TTS_v1_Base model (2545 → 2645 vocab entries).

### Step 4: Train
```bash
accelerate launch --mixed_precision=bf16 \
    src/f5_tts/train/finetune_cli.py \
    --exp_name F5TTS_v1_Base \
    --dataset_name "f5-gu" \
    --batch_size_per_gpu 3200 \
    --finetune \
    --pretrain "ckpts/f5_gu_hi_extended/pretrained_model_1250000.safetensors" \
    --tokenizer custom \
    --tokenizer_path "ckpts/f5_gu_hi_extended/vocab.txt" \
    --log_samples --logger tensorboard
```
See [F5_TTS_Training_Journey_Reference.md](F5_TTS_Training_Journey_Reference.md) for full command with all flags.

### Step 5: Inference
```bash
python run_inference.py
```

## Key Learnings

### Data Quality > Quantity
- Speaker diarization is essential for podcast/scraped data
- Multi-speaker clips cause stopping/blabbering in generated audio
- CPS filtering (chars-per-second) removes clips where audio doesn't match text

### F5-TTS Gotchas
- `--exp_name` must be `F5TTS_v1_Base` (model architecture selector, not project name)
- Folder path is `data/{dataset_name}_{tokenizer}/` — naming matters
- Patch `prepare_csv_wavs.py` to skip pinyin conversion for Indic languages
- Use Python API for inference (CLI garbles Unicode for non-Latin scripts)
- Extended vocab.txt must match extended checkpoint embedding size

### Lightning.ai Environment
- Pin `numpy<2` (scipy/sklearn break with numpy 2.x)
- Pin `setuptools<81` (pkg_resources removed in 82+)
- Force-reinstall torchvision if `torchvision::nms` error appears

## Dataset

Private HuggingFace dataset: `Arjun4707/gu-hi-tts`
- ~65,700 rows total (Gujarati ~40K, Hindi ~11K)
- 24kHz mono PCM-16 WAV
- After filtering: ~36K Gujarati clips (~40 hrs), ~11K Hindi clips (~12 hrs)

## Hardware

- **GPU:** NVIDIA L4 (24 GB VRAM) on Lightning.ai
- **RAM:** 31 GB
- **Training speed:** ~2.1 updates/sec
- **Time per epoch:** ~60 minutes
- **Total training:** ~21 hours (21 epochs, 150K steps)

## References

- [F5-TTS](https://github.com/SWivid/F5-TTS) — Official repo
- [XTTS v2](https://github.com/anhnh2002/XTTSv2-Finetuning-for-New-Languages) — Fine-tuning fork
- [SPRINGLab F5-Hindi](https://huggingface.co/SPRINGLab/F5-Hindi-24KHz) — Existing Hindi F5-TTS model by IIT Madras

## License

Scripts in this repo are MIT. Trained model weights inherit CC-BY-NC from the F5-TTS pretrained base (trained on Emilia dataset).

## Data provenance

The training dataset (`Arjun4707/gu-hi-tts`) was constructed by scraping audio from publicly available YouTube videos, followed by automatic transcription and audio preprocessing.

**This means:**
- The trained model weights and generated audio are for **non-commercial use only**
- Reference audio clips from the training data should not be redistributed


## Author

**Arjun** — [HuggingFace](https://huggingface.co/Arjun4707) | [GitHub](https://github.com/BhammarArjun)
