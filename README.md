# Zero-Shot Traffic Accident Anticipation

[![Kaggle](https://img.shields.io/badge/Kaggle-Competition-blue)](https://kaggle.com/competitions/zero-shot-taa)
[![CVPR](https://img.shields.io/badge/CVPR-2026%20Workshop-red)](https://cvpr.thecvf.com)
[![License](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-green)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

## Overview

This repository contains the full inference pipeline and predictions for the **Zero-Shot Traffic Accident Anticipation** challenge hosted on Kaggle as part of the **CVPR 2026 Workshop: AUTOPILOT-COG**.

The goal is to predict a per-frame risk score sequence `p_t` for `t = 1..150` for each 5-second dashcam video clip (150 frames, 30 FPS), such that risk crosses 0.5 as early and stably as possible before the accident occurs.

---

## Task Definition

| Property | Value |
|---|---|
| Dataset | MM-AU (curated test subset) |
| Clips | 1,417 test clips |
| Frames per clip | 150 (5 seconds @ 30 FPS) |
| Modalities | RGB frames + Driver gaze maps + Text captions |
| Setting | **Zero-shot** — no training data provided |

---

## Our Approach

We propose a **multi-modal zero-shot ensemble** combining three independent signals:

### 1. CLIP Visual Scoring (Weight: 0.55)
- Model: `ViT-L/14`
- Each frame scored against text prompts:
  - Positive: `"dangerous accident collision crash"`
  - Negative: `"normal safe driving no hazard"`
- Per-frame cosine similarity → risk signal

### 2. Optical Flow Motion Signal (Weight: 0.25)
- Frame-difference based motion estimation
- High motion magnitude = high accident risk
- Normalized across all frames per clip

### 3. Caption-Based NLP Prior (Weight: 0.20)
- Model: `all-MiniLM-L6-v2` (SentenceTransformer)
- Each caption scored against sudden/gradual accident anchors
- Mapped to sigmoid crossover frame `t0`
- `start_frame` metadata used for temporal calibration

---

## Ensemble Formula
p_final(t) = 0.55 * p_clip(t) + 0.25 * p_flow(t) + 0.20 * p_caption(t)

**Post-Processing:**
p_final = power(gaussian_smooth(p_final, sigma=2), 1.8)
p_final = monotone_clamp(p_final, threshold=0.5)
p_final = time_shift(p_final, factor=1.3)

---

## Results

| Metric | Value |
|---|---|
| Total Predictions | 1,417 |
| Avg Crossover Frame | 22.7 |
| Min Crossover Frame | 8 |
| First Frame Risk | ~0.001 |
| Last Frame Risk | ~0.92+ |

---

## Repository Structure
zero-shot-accident-anticipation/
├── README.md                    # This file
├── submission_shifted.csv       # Final submission (all 1417 predictions)
├── top50_final.csv              # Top 50 best predictions showcase
└── predict.py                   # Full inference pipeline
---

## Prediction Format

Each prediction is a JSON list of 150 float values in [0, 1]:
id,risk
11_009790_112_262,"[0.001, 0.001, 0.001, ..., 0.968, 0.981, 0.996]"
- Values start near **0.001** (low risk at clip start)
- Cross **0.5** early (frame 8–30 depending on accident type)
- Reach **0.92+** by clip end

---

## Top 50 Best Predictions

`top50_final.csv` contains the 50 clips where our model predicts danger earliest:

| Metric | Value |
|---|---|
| first_risk range | 0.001 |
| max_risk range | 0.926 – 0.994 |
| Avg crossover frame | 21.1 |
| Min crossover frame | 16 |

---

## Installation & Usage

```bash
# Clone repo
git clone https://github.com/YOUR_USERNAME/zero-shot-accident-anticipation
cd zero-shot-accident-anticipation

# Install dependencies
pip install torch torchvision git+https://github.com/openai/CLIP.git
pip install sentence-transformers scipy pandas numpy Pillow

# Run inference
python predict.py
```

---

## Dependencies

| Library | Purpose |
|---|---|
| `openai/clip` | Visual frame scoring |
| `sentence-transformers` | Caption NLP scoring |
| `scipy` | Gaussian smoothing |
| `torch` | Deep learning backend |
| `pandas / numpy` | Data processing |
| `Pillow` | Image loading |

---

## Citation

```bibtex
@misc{autopilot2026,
  title  = {Zero-shot Accident Anticipation},
  author = {AUTOPILOT-COG},
  year   = {2026},
  url    = {https://kaggle.com/competitions/zero-shot-taa}
}
```

---

## Competition

- **Challenge:** Zero-Shot Accident Anticipation
- **Platform:** Kaggle
- **Workshop:** CVPR 2026 — AUTOPILOT-COG
- **URL:** https://kaggle.com/competitions/zero-shot-taa

---

*This work is submitted as part of the CVPR 2026 Workshop on Autonomous Driving Perception.*
