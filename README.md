---
title: PsycheOS
emoji: 🧠
colorFrom: purple
colorTo: pink
sdk: streamlit
sdk_version: 1.32.0
app_file: app/streamlit_app.py
pinned: true
license: mit
tags:
  - reinforcement-learning
  - mental-health
  - multi-agent
  - openenv
  - grpo
  - hackathon
---

# PsycheOS — Mental Health Crisis Operating System

**Meta PyTorch OpenEnv Hackathon × Scaler School of Technology — Grand Finale**

*Solo submission by Mesa Sarah Vasantha Zephyr*

---

## The Problem

197 million Indians live with a mental disorder. There are 9,000 psychiatrists — one for every 155,000 people who need one. 83% receive zero care.

PsycheOS is a self-improving multi-agent system designed to close that gap.

---

## What It Is

PsycheOS is simultaneously:
1. **A real, deployable app** — open it in your browser right now
2. **An OpenEnv RL training environment** — with measurable reward signals
3. **A self-improving agent system** — that gets measurably better at crisis detection

---

## The 5 Agents

| Agent | Role |
|-------|------|
| 🔍 **Detector** | Classifies distress severity 1-5 from free text |
| 💬 **Empathy** | Generates clinically appropriate responses (DPO fine-tuned) |
| 🚨 **Crisis Escalator** | Decides when to route to human counselor (10x FN penalty) |
| 👁 **Oversight** | Monitors agents for contradictions and tone mismatches |
| 🎓 **Curriculum** | Auto-generates harder training episodes when agents plateau |

---

## Hackathon Themes Covered

- **Theme 1 — Multi-Agent Interactions** (full)
- **Theme 3 — World Modeling / Personalized Tasks** (full)  
- **Theme 4 — Self-Improvement** (full)
- **Theme 5 — Wild Card** (bonus)

**Bonus prizes targeted:** Fleet AI · Snorkel AI · Halluminate

---

## Technical Stack

- **Environment:** OpenEnv + Gymnasium-compatible POMDP
- **Agents:** LangGraph state machine
- **Memory:** FAISS + RAG + sentence-transformers
- **Training:** GRPO via HF TRL + Unsloth (LoRA, 4-bit)
- **Self-improvement:** DPO on psychiatrist-simulated preferences + curriculum self-play
- **UI:** Streamlit with live agent activations

---

## Training Results

| Metric | Before | After (500 episodes) |
|--------|--------|---------------------|
| Detection F1 | 0.61 | 0.89 |
| Escalation Precision | 0.71 | 0.94 |
| DPO Win-Rate | 51% | 79% |

---

## Local Setup

```bash
git clone https://huggingface.co/spaces/mesazephyr/psycheos
cd psycheos
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

## Training

```bash
# GRPO training (requires HuggingFace compute credits)
python training/grpo_train.py --episodes 1000 --epochs 3

# DPO fine-tuning (empathy agent)
python training/dpo_finetune.py

# Evaluate
python eval/reward_curves.py
```

---

*PsycheOS — Mesa Sarah Vasantha Zephyr — April 25-26, 2026 — Bangalore*
