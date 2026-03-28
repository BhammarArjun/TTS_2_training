#!/usr/bin/env python3
"""
F5-TTS Inference — Generate Gujarati speech from text.

Uses Python API directly (not subprocess/CLI) to avoid Unicode garbling
of Gujarati/Hindi text through shell encoding.

Usage:
  python run_inference.py

Configuration:
  Edit MODEL_CKPT, VOCAB_FILE, REF_AUDIO, REF_TEXT, and SENTENCES below.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from f5_tts.api import F5TTS

# ============================================================
# CONFIGURATION — edit these to match your setup
# ============================================================
MODEL_CKPT = "ckpts/f5-gu/model_150000.pt"
VOCAB_FILE = "ckpts/f5_gu_hi_extended/vocab.txt"
OUTPUT_DIR = "inference_outputs"

# Reference audio — use a clean 5-10s single-speaker clip
REF_AUDIO = ""
REF_TEXT = ""

# Inference settings
SPEED = 0.8       # 0.7-1.0, lower = slower speech (default 1.0 feels fast for Gujarati)
NFE_STEP = 64     # 32 = default/fast, 64 = higher quality

# ============================================================
# SENTENCES — edit these to generate different text
# ============================================================
SENTENCES = [
    # Short
    "નમસ્તે, કેમ છો તમે?",
    "આ વાત સાંભળીને મને ખૂબ આનંદ થયો.",
    "દરેક માણસે પોતાના સપના માટે મહેનત કરવી જોઈએ.",

    # Medium
    "અમદાવાદની પોળમાં ફરવાની મજા કંઈક અલગ જ છે અને ત્યાંનું ખાવાનું તો અદ્ભુત હોય છે.",
    "જ્યારે હું નાનો હતો ત્યારે મારા દાદા મને દરરોજ સાંજે વાર્તા કહેતા અને હું ધ્યાનથી સાંભળતો.",
    "ક્રિકેટ એ ભારતનો સૌથી લોકપ્રિય ખેલ છે અને દરેક ગલીમાં બાળકો ક્રિકેટ રમતા જોવા મળે છે.",

    # Long
    "શિક્ષણ એ સૌથી શક્તિશાળી હથિયાર છે જેના દ્વારા તમે દુનિયા બદલી શકો છો અને સમાજમાં સકારાત્મક પરિવર્તન લાવી શકો છો.",
    "ગુજરાતનો ઇતિહાસ ખૂબ જ સમૃદ્ધ છે અને મહાત્મા ગાંધી, સરદાર પટેલ જેવા મહાન નેતાઓએ આ ધરતી પરથી દેશને દિશા આપી છે.",
    "આજકાલ સોશિયલ મીડિયાનો ઉપયોગ એટલો વધી ગયો છે કે લોકો એકબીજા સાથે રૂબરૂ મળવાને બદલે ફોન પર જ વાત કરે છે અને આ ચિંતાનો વિષય છે.",
    "મારું માનવું છે કે જો આપણે આપણી માતૃભાષા ગુજરાતીને જાળવી રાખવી હોય તો આપણે બાળકોને નાનપણથી જ ગુજરાતીમાં વાંચતા અને લખતા શીખવવું જોઈએ જેથી આવનારી પેઢીમાં ભાષા જીવંત રહે.",
]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading model: {MODEL_CKPT}")
    print(f"Vocab: {VOCAB_FILE}")
    print(f"Reference: {REF_AUDIO}")
    print(f"Speed: {SPEED}, NFE steps: {NFE_STEP}")
    print(f"Output: {OUTPUT_DIR}/")
    print()

    # Load model once (not per sentence)
    tts = F5TTS(model="F5TTS_v1_Base", ckpt_file=MODEL_CKPT, vocab_file=VOCAB_FILE)

    for i, text in enumerate(SENTENCES):
        label = "short" if i < 3 else "medium" if i < 6 else "long"
        outfile = os.path.join(OUTPUT_DIR, f"{i+1:02d}_{label}.wav")

        print(f"[{i+1}/{len(SENTENCES)}] {label.upper()}: {text[:60]}...")

        tts.infer(
            ref_file=REF_AUDIO,
            ref_text=REF_TEXT,
            gen_text=text,
            file_wave=outfile,
            speed=SPEED,
            nfe_step=NFE_STEP,
        )
        print(f"  ✅ Saved: {outfile}")

    print(f"\n🎉 Done! {len(SENTENCES)} files in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
