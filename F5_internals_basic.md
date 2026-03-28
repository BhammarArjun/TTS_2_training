
# 0. Notation (keep this fixed)

```text
B = batch size
T = time frames (audio length)
D = mel channels (e.g. 100)
Nt = text token length
```

---

# 1. TRAINING: full tensor flow

## 1.1 Dataset output

From `dataset.py`:

```text
mel_spec: (D, T)
text: string
```

After batching:

```text
mel: (B, D, T_max)
→ permute →
mel: (B, T, D)
```



---

## 1.2 Inside model.forward()

### Step 1: inputs

```text
inp (mel): (B, T, D)
text: list[str]
```

---

### Step 2: text → tokens

```text
text → tensor: (B, Nt)
```

---

### Step 3: define noise and real

```text
x1 = real mel → (B, T, D)
x0 = noise     → (B, T, D)
```

---

### Step 4: sample time t

```text
t: (B,)
→ broadcast →
(B, 1, 1)
```

---

### Step 5: interpolate

```text
φ = (1 - t)*x0 + t*x1
```

Shape:

```text
φ: (B, T, D)
flow = x1 - x0 → (B, T, D)
```

---

### Step 6: create condition (IMPORTANT)

```text
rand_span_mask: (B, T)
```

Example:

```text
mask:
[1 1 1 0 0 0 1 1]
```

Then:

```text
cond = where(mask, 0, x1)
```

So:

```text
cond: (B, T, D)
```

Diagram:

```text
x1 (real audio):
[A A A B B B C C]

mask:
[1 1 1 0 0 0 1 1]

cond:
[0 0 0 B B B 0 0]
```

---

### Step 7: transformer input

```text
x     = φ      → (B, T, D)
cond  = cond   → (B, T, D)
text  = (B, Nt)
time  = (B,)
```

---

### Step 8: transformer output

```text
pred: (B, T, D)
```

---

### Step 9: loss

```text
loss = MSE(pred, flow) only on masked region
```

---

# 2. INFERENCE: full tensor flow

Now the confusing part.

---

## 2.1 Inputs

```text
Reference audio → mel → cond
Target text → tokens
```

Let’s pick numbers:

```text
T_ref = 100
Nt = 30
```

So:

```text
cond: (1, 100, D)
text: (1, 30)
```

---

# 3. WHERE DOES T_out = 300 COME FROM?

This is your confusion. Let’s fix it.

---

## 3.1 Code logic

```python
duration = max(text_len, ref_len) + 1
```



So default:

```text
duration ≈ 101
```

---

## 3.2 Then why did I say 300 earlier?

Because:

### YOU CAN OVERRIDE IT

From trainer:

```python
duration = ref_audio_len * 2
```



So:

```text
T_out = 100 * 2 = 200
```

Or any value you pass.

---

## 3.3 Final rule

```text
T_out = user_given_duration
        OR
        auto = max(text_len, ref_len)+1
```

---

# 4. Now actual inference tensors

Assume:

```text
T_ref = 100
T_out = 300   (user decided)
```

---

## 4.1 Pad condition

```text
cond: (1, 100, D)
→ pad →

cond: (1, 300, D)
```

---

## 4.2 Create mask

```text
cond_mask: (1, 300)

[1 1 1 ... 1 | 0 0 0 ... 0]
   100 ones      200 zeros
```

---

## 4.3 Initialize noise

```text
y0: (1, 300, D)
```

Diagram:

```text
y0:
[N N N N N N N N N ... N]   (300 frames)
```

---

## 4.4 Build step_cond

```python
step_cond = where(cond_mask, cond, 0)
```

So:

```text
step_cond:
[A A A ... A | 0 0 0 ... 0]
```

---

# 5. TRANSFORMER INPUT (EACH STEP)

At each ODE step:

```text
x        : (1, 300, D)   ← current noisy audio
cond     : (1, 300, D)   ← known part
text     : (1, Nt)
time     : (1,)
```

---

# 6. WHAT TRANSFORMER DOES (INTUITION + SHAPES)

Internally (simplified):

### Concatenation / conditioning happens like:

```text
Audio tokens: (T_out = 300)
Text tokens : (Nt = 30)
```

Attention:

```text
Each audio frame attends to:
    - all audio frames (self-attn)
    - all text tokens
```

---

## 6.1 Attention diagram

```text
Audio sequence (300 frames):
[A A A A A A A A A ...]

Text tokens (30):
[t t t t t t ...]

Each audio frame:
    attends to ALL text tokens
```

---

# 7. ODE UPDATE LOOP

Loop:

```text
x_0 → x_1 → x_2 → ... → x_T
```

Shape stays:

```text
(1, 300, D)
```

Only values change.

---

# 8. FINAL STEP (VERY IMPORTANT)

```python
out = where(cond_mask, cond, generated)
```



---

## 8.1 Final output

```text
out:
[A A A ... A | G G G ... G]
```

Where:

* A = EXACT reference audio
* G = generated continuation

---

# 9. FULL INFERENCE PIPELINE (LINE DIAGRAM)

```text
INPUT:

Reference audio:
[A A A A A] (100 frames)

Text:
[t t t t t t]

-----------------------------------

PAD:

[A A A A A | 0 0 0 0 0]   → (300)

-----------------------------------

INIT:

[N N N N N N N N N ...]   (300)

-----------------------------------

ITERATE:

Step 1:
[A A A A A | noisy noisy]

Step 10:
[A A A A A | rough speech]

Step 30:
[A A A A A | clear speech]

-----------------------------------

FINAL:

[A A A A A | generated speech]
```

---

# 10. KEY TAKEAWAYS

## 1. Duration

```text
T_out is NOT predicted
YOU decide it
```

---

## 2. Shapes stay constant

```text
(B, T_out, D) everywhere
```

---

## 3. Reference audio

```text
used as fixed conditioning
copied directly in output
```

---

## 4. Model behavior

```text
fills missing region using:
    - text
    - known audio
```

---

# 11. If one thing to remember

```text
Training:
    random holes → fill

Inference:
    tail is hole → fill
```


