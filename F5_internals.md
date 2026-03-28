
# 1. TRAINING PIPELINE (end-to-end)

## 1.1 Dataset: what happens to audio + text

From `dataset.py`:

### Audio

* Raw waveform → resampled to 24 kHz
* Converted to **mel spectrogram**
* Shape becomes:

```
mel_spec: (n_mel, T)
```

Then in collate:

```
mel: (B, n_mel, T_max)
→ permuted later to (B, T, n_mel)
```

Citation:

* mel creation + return dict 
* padding + batching 

---

### Text

* Text is kept as **raw string**
* No padding here
* Tokenization happens inside model later

---

## 1.2 Trainer: what goes into model

From `trainer.py`:

```
mel_spec = batch["mel"].permute(0, 2, 1)   # (B, T, n_mel)
text_inputs = batch["text"]
mel_lengths = batch["mel_lengths"]
```

Then:

```
loss, cond, pred = model(
    mel_spec,
    text=text_inputs,
    lens=mel_lengths
)
```

Citation:

* forward call 

---

# 2. CORE: WHAT MODEL DOES DURING TRAINING

Now from `cfm.py` → this is the truth.

---

## 2.1 Inputs to model.forward

```
inp  = mel_spec (B, T, D)
text = raw text
lens = lengths
```

---

## 2.2 Convert text → tokens

```
text = list_str_to_tensor(text)
```

So:

```
text: (B, Nt)
```

No explicit alignment here yet.

---

## 2.3 Define ground truth and noise

```
x1 = inp            # REAL mel (target)
x0 = torch.randn_like(x1)   # NOISE
```

---

## 2.4 Sample random timestep t

```
t ~ Uniform(0,1)
```

---

## 2.5 Interpolate (THIS IS CRITICAL)

```
φ = (1 - t) * x0 + t * x1
flow = x1 - x0
```

Interpretation:

* φ = partially noised mel
* flow = direction from noise → real audio

---

## 2.6 VERY IMPORTANT: how conditioning audio is formed

```
rand_span_mask = mask_from_frac_lengths(...)
cond = torch.where(rand_span_mask, 0, x1)
```

This is the key:

### What this means:

* Some random span of the audio is **masked out**
* Remaining part is kept as condition

So:

```
cond = PARTIAL REAL AUDIO
target = FULL AUDIO
```

---

### This directly answers your question:

> is initial part used?

**NO — not specifically initial**

It is:

* **random spans**
* sometimes middle
* sometimes end
* sometimes large chunk

Not prefix-based.

---

## 2.7 Transformer input

```
pred = transformer(
    x = φ                  # noisy mel
    cond = cond            # partial real mel
    text = text            # tokenized text
    time = t
)
```

---

## 2.8 Loss

```
loss = MSE(pred, flow)
only on masked region
```

So model learns:

> given partial audio + text → reconstruct missing parts from noise

---

## 2.9 Classifier-free guidance training

Randomly:

```
drop_audio_cond
drop_text
```

So model also learns:

* text-only generation
* audio-only conditioning
* both

---

# 3. TRAINING SUMMARY (precise)

For each sample:

```
INPUT:
    text
    partial audio (random masked mel)
    noisy mel (φ)

TARGET:
    flow = full_audio - noise

MODEL LEARNS:
    how to move noisy mel → real mel
    using text + partial audio
```

---

# 4. INFERENCE PIPELINE (THIS IS WHAT YOU CARE ABOUT)

Now from `CFM.sample()`

---

## 4.1 Inputs

```
cond = reference audio (mel)
text = target text
duration = desired length
```

---

## 4.2 Preprocess reference audio

```
cond → mel → (B, T_ref, D)
```

---

## 4.3 Padding to full generation length

```
cond → padded to max_duration
cond_mask marks where reference exists
```

---

## 4.4 Noise initialization

```
y0 = random noise (duration, D)
```

---

## 4.5 ODE solving (core generation)

Loop:

```
for t in steps:
    pred = transformer(x, cond, text, t)
    update x using ODE
```

Citation:

* ODE + fn 

---

## 4.6 IMPORTANT: how reference audio is used

```
step_cond = where(cond_mask, cond, 0)
```

So:

* Reference audio is **fixed conditioning**
* Not modified
* Not extended autoregressively

---

## 4.7 Final output stitching

```
out = where(cond_mask, cond, generated)
```

This is critical:

> The reference audio is **copied directly into output**

---

# 5. FINAL ANSWER TO YOUR CORE DOUBT

## Q1: Is initial part of reference audio used?

**YES — but not in training specifically**

During inference:

* reference audio occupies **first part (or masked region)**
* copied directly into output

---

## Q2: Is full audio used as target?

**YES (during training)**

But:

* model only predicts **masked parts**
* rest is given as condition

---

## Q3: Is it prefix continuation like VALL-E?

**NO**

Instead:

* Training:

  * random infilling (not prefix)
* Inference:

  * behaves like prefix because cond is placed at beginning

---

# 6. Correct mental model

### Training:

```
[random holes in audio]
→ fill using text + surrounding audio
```

### Inference:

```
[reference audio] + [empty region]
→ fill empty region using text
```

---

# 7. Subtle but important insight

This line defines everything:

```
cond = torch.where(rand_span_mask, 0, x1)
```

This is:

> **span-infilling diffusion**, not continuation

---

# 8. Why it still works like voice cloning

Because model learns:

* speaker identity from visible audio spans
* text alignment from full sequence

So at inference:

* give prefix audio → model continues naturally


