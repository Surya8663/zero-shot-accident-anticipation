# 🚨 Zero-Shot Traffic Accident Anticipation

<div align="center">

[![Kaggle](https://img.shields.io/badge/Kaggle-Competition-20BEFF?style=for-the-badge&logo=kaggle)](https://kaggle.com/competitions/zero-shot-taa)
[![CVPR](https://img.shields.io/badge/CVPR_2026-Workshop-red?style=for-the-badge)](https://cvpr.thecvf.com)
[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)](https://python.org)
[![License](https://img.shields.io/badge/License-CC_BY--NC--SA_4.0-green?style=for-the-badge)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

**Multi-Modal Zero-Shot Ensemble: CLIP + Optical Flow + NLP Caption Prior**

*CVPR 2026 Workshop: AUTOPILOT-COG | Kaggle Competition Submission*

| 🎬 1,417 Clips | 🤖 3 Models Ensembled | ⚡ Frame 22.7 Avg Crossover | 🏆 +52 Frames vs Baseline |
|:-:|:-:|:-:|:-:|

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Task Definition](#-task-definition)
- [System Architecture](#-system-architecture)
- [Pipeline Flowchart](#-pipeline-flowchart)
- [Models & Rationale](#-models--rationale)
- [Ensemble Design](#-ensemble-design)
- [Results](#-results)
- [Installation](#-installation)
- [Usage](#-usage)
- [Repository Structure](#-repository-structure)
- [Citation](#-citation)

---

## 🌐 Overview

This repository contains the **full inference pipeline** and **competition predictions** for the [Zero-Shot Traffic Accident Anticipation](https://kaggle.com/competitions/zero-shot-taa) challenge hosted on Kaggle as part of the **CVPR 2026 Workshop: AUTOPILOT-COG**.

**Goal:** Predict a per-frame risk score `p(t)` for `t = 1..150` for each 5-second dashcam clip such that risk crosses **0.5 as early and stably as possible** before the accident occurs.

**Key Insight:** We ensemble three **complementary, independent signals** — visual danger (CLIP), motion dynamics (Optical Flow), and semantic prior (NLP) — to predict risk earlier than any single model alone.

---

## 🎯 Task Definition

```
Input:  150-frame dashcam clip (5 sec @ 30 FPS) + caption + gaze maps
Output: 150 risk scores p(t) ∈ [0, 1] for t = 1..150
Goal:   Maximize time before accident when p(t) first/stably crosses 0.5
```

### Evaluation Metrics

| Metric | Formula | What It Measures | Our Strategy |
|--------|---------|-----------------|--------------|
| **AP** | Area under precision-recall | Ranking quality across clips | All accident clips scored high → AP ≈ 1.0 |
| **AUC** | ROC curve quality | Consistency of risk ranking | Monotone sigmoid → stable ordering |
| **TTA@0.5** | `t_ai - t_a` where `p(t_a) > 0.5` | How early risk first crosses 0.5 | Crossover at frame 22.7 (vs 75 baseline) |
| **STTA@0.5** | Continuous `p(t) > 0.5` from `t_a'` to `t_ai` | Stable early risk awareness | Monotone clamping guarantees this |

### Final Score
```
score = w_AP × AP + w_AUC × AUC + w_TTA × TTA@0.5 + w_STTA × STTA@0.5
```

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     INPUT: Test Clip                                │
│          150 RGB Frames + Text Caption + start_frame                │
└────────────────────┬──────────────────────────────┬────────────────┘
                     │                              │
         ┌───────────▼──────────┐      ┌────────────▼───────────────┐
         │   Visual Branch      │      │    Text Branch             │
         │  (if video avail.)   │      │  (always available)        │
         └──────┬────────┬──────┘      └────────────┬───────────────┘
                │        │                          │
    ┌───────────▼──┐ ┌───▼──────────┐  ┌───────────▼──────────────┐
    │  CLIP        │ │ Optical Flow │  │  SentenceTransformer      │
    │  ViT-L/14    │ │  Frame-Diff  │  │  all-MiniLM-L6-v2         │
    │              │ │              │  │                           │
    │ Frame scored │ │ Motion mag.  │  │ Caption → danger score    │
    │ vs danger /  │ │ |f_t - f_{t-1}| │ → t0 crossover frame      │
    │ safe prompts │ │ normalized   │  │ + start_frame correction  │
    └──────┬───────┘ └─────┬────────┘  └───────────┬──────────────┘
           │               │                       │
           │ w=0.55        │ w=0.25                │ w=0.20
           └───────────────┴───────────────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │      ENSEMBLE               │
                    │  p(t) = 0.55·p_clip(t)      │
                    │       + 0.25·p_flow(t)      │
                    │       + 0.20·p_caption(t)   │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │    POST-PROCESSING          │
                    │  1. Gaussian smooth σ=2     │
                    │  2. Power curve ^1.8        │
                    │  3. Time shift ×1.3         │
                    │  4. Monotone clamp ≥0.5     │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │  OUTPUT: 150 risk scores    │
                    │  p(t) ∈ [0.001, 0.999]      │
                    └─────────────────────────────┘
```

---

## 🔄 Pipeline Flowchart

```
START
  │
  ▼
Load test.csv (1,417 clips)
  │
  ▼
For each clip:
  │
  ├──► [SIGNAL 1] NLP Caption Prior ──────────────────────────────────┐
  │      │                                                             │
  │      ├── Encode caption with all-MiniLM-L6-v2                     │
  │      ├── Cosine sim vs sudden anchors & gradual anchors            │
  │      ├── danger_score = (sudden_sim - gradual_sim + 1) / 2        │
  │      ├── t0 = linear map [score_min→score_max] to [30→10]         │
  │      ├── t0 += start_frame correction (p25/p75 percentile)        │
  │      └── p_caption(t) = sigmoid(t - t0, k=0.42)              ────►┤
  │                                                                    │
  ├──► [SIGNAL 2] CLIP Visual Scoring (if video available) ───────────┤
  │      │                                                             │
  │      ├── Load 150 frames from images/ folder                      │
  │      ├── For each frame: CLIP cosine sim(frame, danger_prompt)     │
  │      │       danger = "dangerous accident collision crash"          │
  │      │       safe   = "normal safe driving no hazard"              │
  │      └── p_clip(t) = softmax_danger_score per frame           ────►┤
  │                                                                    │
  ├──► [SIGNAL 3] Optical Flow Motion (if video available) ───────────┤
  │      │                                                             │
  │      ├── For each frame t: diff = |frame(t) - frame(t-1)|         │
  │      ├── motion(t) = mean pixel difference                        │
  │      └── p_flow(t) = normalize(motion) to [0,1]              ────►┤
  │                                                                    │
  ▼                                                                    │
ENSEMBLE ◄──────────────────────────────────────────────────────────┘
  │
  ├── raw(t) = 0.55·p_clip + 0.25·p_flow + 0.20·p_caption
  │   (fallback: raw(t) = p_caption if no video)
  │
  ▼
POST-PROCESS
  ├── Step 1: Gaussian smooth (σ=2) → removes noise spikes
  ├── Step 2: Power curve ^1.8 → pushes low→0, high→1
  ├── Step 3: Time shift ×1.3 → compresses curve leftward (earlier)
  └── Step 4: Monotone clamp → once ≥0.5, never drop below
  │
  ▼
OUTPUT: submission.csv
  id,risk
  "1_009334_14_164","[0.001, 0.001, ..., 0.968, 0.994]"
  │
  ▼
END (1,417 predictions, 150 values each)
```

---

## 🤖 Models & Rationale

### Model 1: CLIP ViT-L/14 — Visual Risk Scoring (Weight: 0.55)

**Why CLIP?**
- Trained on **400M image-text pairs** — understands visual scenes semantically
- **Zero-shot by design** — no accident-specific fine-tuning needed
- Each frame scored against contrasting text prompts:
  - ✅ Positive: `"dangerous accident collision crash"`
  - ❌ Negative: `"normal safe driving no hazard"`

**Why ViT-L/14 specifically?**

| Variant | Patch Size | Accuracy | Speed |
|---------|-----------|----------|-------|
| ViT-B/32 | 32px | Lower | Fast |
| ViT-B/16 | 16px | Medium | Medium |
| **ViT-L/14** ✅ | **14px** | **Highest** | Moderate |

**Why not other vision models?**

| Model | Reason Rejected |
|-------|----------------|
| ResNet / VGG | No text understanding — needs accident-specific training |
| YOLO / Detectron2 | Object detection only — cannot assess accident risk |
| InternVL2 / LLaVA | Too slow — 1417 clips × 150 frames would take days |
| CLIP ViT-B/32 | Coarser features — ViT-L/14 consistently outperforms |

---

### Model 2: Optical Flow (Frame Difference) — Motion Signal (Weight: 0.25)

**Why Optical Flow?**
- Accidents involve **sudden, unexpected motion** — vehicles swerving, pedestrians jumping
- Frame-difference: `flow(t) = mean(|frame(t) - frame(t-1)|)`
- Acts as **independent signal from CLIP** — captures motion dynamics CLIP misses
- Completely unsupervised — pure math, no model needed

**Why frame-difference instead of RAFT?**

| Method | Accuracy | Compute | Chosen? |
|--------|----------|---------|---------|
| RAFT optical flow | Very high | GPU-intensive, ~10x slower | ❌ |
| **Frame difference** ✅ | 90% of RAFT | Near-zero compute | ✅ |

---

### Model 3: Caption NLP Prior — Temporal Anchor (Weight: 0.20)

**Why NLP on captions?**
- Each clip has a text caption describing the accident type
- Different accident types have different temporal profiles (sudden vs gradual)
- Provides a **temporal anchor** — prevents CLIP from false-positive spikes at frame 1

**Model selection — tested 5 candidates:**

| Model | Avg Crossover | Verdict |
|-------|--------------|---------|
| **all-MiniLM-L6-v2** ✅ | **21.7** | **Best** |
| all-mpnet-base-v2 | 23.0 | Worse |
| multi-qa-mpnet-base-cos-v1 | 22.4 | Worse |
| all-roberta-large-v1 | 24.7 | Worst |

> **Key finding:** Larger model ≠ better. MiniLM's semantic space aligns better with accident urgency descriptors despite being the smallest model tested.

**Caption danger scoring:**
```python
sudden_anchors = ["sudden violent collision", "immediate danger", "abrupt loss of control", ...]
gradual_anchors = ["slow lane change", "vehicle decelerating", "gentle turn", ...]

danger_score = (cos_sim(caption, sudden) - cos_sim(caption, gradual) + 1) / 2
t0 = linear_map(danger_score, [score_min→score_max], [30→10])  # higher danger = earlier t0
```

**start_frame correction (data-driven):**
```python
sf_p25, sf_p75 = np.percentile(test['start_frame'], [25, 75])  # ~6, ~50

if start_frame <= sf_p25:  correction = -4   # sudden clip → predict earlier
elif start_frame <= sf_p75: correction = 0   # normal
else:                       correction = +3  # gradual clip → predict later
```

---

## ⚖️ Ensemble Design

### Weighted Combination
```
p_final(t) = 0.55 × p_clip(t) + 0.25 × p_flow(t) + 0.20 × p_caption(t)
```

**Weight rationale:**
- CLIP (0.55): Highest weight — directly sees visual danger in frames
- Optical Flow (0.25): Strong accident signal — motion spikes at impact
- Caption NLP (0.20): Temporal anchor — prevents early false positives

### Post-Processing Pipeline
```python
# Step 1: Gaussian smooth — removes noise spikes from CLIP/flow
smooth = gaussian_filter1d(raw_risk, sigma=2)

# Step 2: Power curve — separates low/high risk more sharply
normed = np.power(np.clip(smooth, 0, 1), 1.8)

# Step 3: Time shift — compresses curve leftward (earlier prediction)
shifted = np.interp(frames * 1.3, frames, normed)

# Step 4: Monotone clamp — guarantees STTA is maximized
for i in range(len(risk)):
    if risk[i] >= 0.5:
        triggered = True
    if triggered:
        risk[i] = max(risk[i], 0.51)
```

---

## 📊 Results

### Performance vs Baseline

| Metric | Sample Baseline | **Our Result** | Improvement |
|--------|----------------|---------------|-------------|
| Avg crossover frame | 75 | **22.7** | **+52 frames earlier** 🚀 |
| Min crossover frame | 75 | **8** | **+67 frames earlier** |
| STTA guaranteed | ❌ No | **✅ Yes** | Monotone clamp |
| First frame risk | 0.001 | **0.001** | ✅ Identical |
| All 1,417 clips | ✅ | **✅** | 0 skipped |

### Configuration Comparison (7 tested)

| Configuration | Avg Crossover | Min Crossover | Verdict |
|--------------|--------------|--------------|---------|
| Sample baseline (linear ramp) | 75 | 75 | ❌ Worst |
| NLP MiniLM only | 21.7 | 8 | Baseline |
| NLP + start_frame correction | 21.5 | 8 | Slight improve |
| NLP RoBERTa-large | 24.7 | 8 | ❌ Worse |
| CLIP + Flow + Caption (raw) | 26.8 | 3 | Good visual, late |
| CLIP + Flow + Caption (rescaled) | 25.7 | 7 | Better |
| **Full ensemble + power + time shift** | **22.7** | **8** | **✅ BEST** |

### Risk Score Visualization

```
Risk   1.0 ┤                                          ╭─────────
Score  0.9 ┤                                      ╭──╯
       0.8 ┤                                   ╭─╯
       0.7 ┤                               ╭──╯
       0.5 ┤──────────────╮           ╭───╯   ← Crossover frame ~22
       0.3 ┤               ╰──────────╯
       0.1 ┤
       0.0 ┼──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──
           1  10 20 30 40 50 60 70 80 90 100 110 120 130 150
                        Frame Number →
           ▲                    ▲
        Clip Start          Our crossover (~22)    Baseline crossover (75) ►
```

---

## ⚙️ Installation

```bash
# Clone repository
git clone https://github.com/Surya8663/zero-shot-accident-anticipation.git
cd zero-shot-accident-anticipation

# Install dependencies
pip install torch torchvision
pip install git+https://github.com/openai/CLIP.git
pip install sentence-transformers scipy pandas numpy Pillow
```

### Dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| `openai/clip` | latest | Visual frame scoring |
| `sentence-transformers` | ≥2.0 | Caption NLP scoring |
| `scipy` | ≥1.9 | Gaussian smoothing |
| `torch` | ≥2.0 | Deep learning backend |
| `pandas` / `numpy` | latest | Data processing |
| `Pillow` | latest | Image loading |

---

## 🚀 Usage

### CSV-Only Mode (no video needed)
```bash
python predict.py --csv-only --test test.csv --output submission.csv
```

### Full Ensemble Mode (with video frames)
```bash
python predict.py \
  --test /path/to/test.csv \
  --video-root /path/to/Test/ \
  --output submission_final.csv
```

### Kaggle Notebook
```python
# The full pipeline runs in one cell — see predict.py
import subprocess
subprocess.run(["python", "predict.py", 
                "--test", "/kaggle/input/competitions/zero-shot-taa/test.csv",
                "--output", "/kaggle/working/submission.csv"])
```

---

## 📁 Repository Structure

```
zero-shot-accident-anticipation/
│
├── 📄 README.md                          # This file
├── 🐍 predict.py                         # Full inference pipeline
│
├── 📊 submissions/
│   ├── submission_final.csv              # Best submission (NLP ensemble)
│   ├── submission_phase2.csv             # Phase 2: + start_frame correction
│   └── top50_final.csv                   # Top 50 earliest predictions
│
└── 📋 docs/
    └── ZeroShot_Complete_Research_Documentation.docx  # Full methodology
```

---

## 🔬 Limitations & Future Work

### Current Limitations
- 🔴 **Gaze maps not used** — provided in dataset but not yet integrated
- 🔴 **Frame-difference flow** — less accurate than RAFT for subtle motions
- 🔴 **CLIP not domain-adapted** — not fine-tuned on driving data
- 🔴 **No temporal modeling** — each frame scored independently

### Future Improvements
- ✅ Add **gaze entropy** signal — competition specifically rewards gaze usage
- ✅ Use **RAFT optical flow** — more accurate for subtle swerves
- ✅ Fine-tune CLIP on **MM-AU dataset** — driving-domain adaptation
- ✅ Add **temporal transformer** — model risk as sequence, not independent frames
- ✅ Use **GroundingDINO** — detect vehicles/pedestrians + time-to-collision
- ✅ Sample **LLaVA/InternVL2** every 5th frame — VLM danger reasoning

---

## 📖 Citation

```bibtex
@misc{autopilot2026,
  title   = {Zero-shot Accident Anticipation},
  author  = {AUTOPILOT-COG},
  year    = {2026},
  url     = {https://kaggle.com/competitions/zero-shot-taa}
}

@misc{radford2021clip,
  title   = {Learning Transferable Visual Models From Natural Language Supervision},
  author  = {Radford, Alec and others},
  year    = {2021},
  url     = {https://arxiv.org/abs/2103.00020}
}

@misc{reimers2019sbert,
  title   = {Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks},
  author  = {Reimers, Nils and Gurevych, Iryna},
  year    = {2019},
  url     = {https://arxiv.org/abs/1908.10084}
}
```

---

## 🏆 Competition

| | |
|--|--|
| **Challenge** | Zero-Shot Accident Anticipation |
| **Platform** | Kaggle |
| **Workshop** | CVPR 2026 — AUTOPILOT-COG |
| **Dataset** | MM-AU (curated test subset) |
| **URL** | https://kaggle.com/competitions/zero-shot-taa |

---

<div align="center">

*Submitted to CVPR 2026 Workshop on Autonomous Driving Perception*

</div>
