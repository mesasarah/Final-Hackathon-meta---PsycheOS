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

# 🧠 PsycheOS — Mental Health Crisis Operating System

> **Meta PyTorch OpenEnv Hackathon × Scaler School of Technology — Grand Finale**
> Solo submission · Mesa Sarah Vasantha Zephyr · April 25–26, 2026 · Bangalore

---

## 🔗 Links

| Resource | URL |
|----------|-----|
| 🚀 Live HF Space | https://huggingface.co/spaces/mesazephyr/psycheos |
| 📝 Mini-blog (HuggingFace) | _[Add after training — April 25]_ |
| 🎥 Demo video (YouTube, <2 min) | _[Add after training — April 25]_ |
| 💻 Training notebook (Colab) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mesazephyr/psycheos/blob/main/training/colab_training.ipynb) |
| 🤗 LoRA weights | _[Add after training — mesazephyr/psycheos-lora]_ |

---

## The Problem

**197 million** Indians live with a mental disorder.
**9,000** psychiatrists exist for a population of 1.4 billion.
That is **one doctor for every 155,000 people** who need one.
**83%** receive zero care.

The gap is not just statistical — it is deadly. Delayed crisis identification and zero continuity between care interactions cost lives. PsycheOS is built to close that gap.

---

## What is PsycheOS?

PsycheOS is simultaneously:

1. **A real deployable app** — open it in your browser right now, talk to it
2. **An OpenEnv RL training environment** — Gymnasium-compatible POMDP with measurable reward signals
3. **A self-improving multi-agent system** — gets measurably better at crisis detection through GRPO + curriculum self-play

---

## How the Environment Works

### What the agent sees (observation space)

```
{
  "message": "<patient's raw text>",
  "session_history": [last 3 turns with distress labels],
  "turn": <int>,
  "session": <int>
}
```

### What the agent does (action space)

```
{
  "distress_level": <1–5>,       # classify severity
  "response": "<text>",          # generate empathetic reply
  "escalate": <true/false>       # route to human counselor?
}
```

### What the agent gets rewarded for

| Signal | Weight | Description |
|--------|--------|-------------|
| Detection F1 | 40% | Accuracy of distress severity classification |
| Empathy DPO | 40% | Psychiatrist preference margin on responses |
| Escalation P/R | 20% | Crisis routing precision — **10x penalty on false negatives** |

The reward is hard to game. An agent that always escalates scores −1 on false positives. An agent that never escalates scores −10 on every missed crisis. The only way to score well is to actually solve the task.

---

## The 5 Agents

```
User message
     │
     ▼
┌─────────────┐     ┌──────────────┐     ┌───────────────────┐
│   Memory    │────▶│   Detector   │────▶│   Empathy Agent   │
│  Retrieval  │     │  Agent (1–5) │     │  (DPO fine-tuned) │
└─────────────┘     └──────────────┘     └───────────────────┘
                                                   │
                                                   ▼
                    ┌──────────────┐     ┌───────────────────┐
                    │  Oversight   │◀────│ Crisis Escalator  │
                    │    Agent     │     │     Agent         │
                    └──────────────┘     └───────────────────┘
                           │
                           ▼
                      Final response
                   (or crisis escalation)
```

| Agent | Role | Bonus prize target |
|-------|------|--------------------|
| 🔍 **Detector** | Classifies distress 1–5 from free text | — |
| 💬 **Empathy** | Generates responses (DPO fine-tuned on psychiatrist preferences) | — |
| 🚨 **Crisis Escalator** | Routes to human counselor (10x FN penalty) | — |
| 👁 **Oversight** | Monitors all agents for contradictions in real time | Fleet AI |
| 🎓 **Curriculum** | Auto-generates harder episodes when agents plateau | Snorkel AI |

---

## Self-Improvement Loop

```
Episode → Reward → Curriculum tracks rolling avg
                         │
              improvement < 2% over 50 episodes?
                    YES ──────────────────────────▶ harder episodes generated
                    NO  ──────────────────────────▶ continue current difficulty
                                │
                     DPO fine-tune empathy agent
                   on psychiatrist preference pairs
                   (preferences shift each epoch —
                    satisfies Snorkel AI requirement)
                                │
                    LoRA adapters updated via GRPO
```

---

## Training Results

> ⚠️ _Placeholder — replace with real numbers and plots after onsite training April 25._

![Reward Curves](plots/reward_curves.png)
_Fig 1. Composite reward, detection F1, and DPO win-rate over 500 training episodes (baseline vs trained on same axes)._

| Metric | Baseline (untrained) | After training |
|--------|---------------------|----------------|
| Detection F1 | 0.61 | — |
| Escalation Precision | 0.71 | — |
| DPO Win-Rate vs baseline | 51% | — |
| Avg Composite Reward | 0.35 | — |

---

## Hackathon Theme Coverage

| Theme | Coverage |
|-------|----------|
| Theme 1 — Multi-Agent Interactions | ✅ 5 agents cooperating, competing, overriding each other |
| Theme 3 — World Modeling (Personalized) | ✅ Dynamic patient world across multi-session POMDP |
| Theme 4 — Self-Improvement | ✅ Curriculum self-play + DPO adaptation per epoch |
| Theme 5 — Wild Card | ✅ First mental health crisis RL environment |

**Bonus prizes targeted:** Fleet AI · Snorkel AI · Halluminate

---

## Run Locally

```bash
git clone https://huggingface.co/spaces/mesazephyr/psycheos
cd psycheos
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

## Train (onsite with HF compute credits)

```bash
# GRPO training loop
python training/grpo_train.py --episodes 1000 --epochs 3

# DPO fine-tuning — empathy agent
python training/dpo_finetune.py

# Evaluate + generate reward curve plots
python eval/reward_curves.py
```

---

## Repository Structure

```
psycheos/
├── openenv.yaml               ← OpenEnv manifest (required by judges)
├── README.md                  ← This file
├── requirements.txt
├── app/
│   └── streamlit_app.py       ← Live browser UI
├── environment/
│   └── psycheos_env.py        ← OpenEnv POMDP environment
├── agents/
│   ├── agents.py              ← All 5 agent implementations
│   └── graph.py               ← LangGraph state machine
├── memory/
│   └── faiss_store.py         ← FAISS + RAG cross-session memory
├── reward/
│   └── reward_functions.py    ← 5 composable reward signals
├── training/
│   ├── grpo_train.py          ← GRPO training loop
│   ├── dpo_finetune.py        ← DPO fine-tuning script
│   └── colab_training.ipynb   ← Runnable Colab notebook
├── eval/
│   └── reward_curves.py       ← Evaluation + matplotlib plots
└── plots/
    └── reward_curves.png      ← Added after training
```

---

## Technical Stack

| Layer | Tech |
|-------|------|
| Environment | OpenEnv + Gymnasium POMDP |
| Agents | LangGraph state machine |
| Memory | FAISS + sentence-transformers RAG |
| Training | GRPO · HuggingFace TRL · Unsloth |
| Adapters | LoRA (r=16) · 4-bit quantization |
| Self-improvement | DPO on psychiatrist preferences + curriculum self-play |
| UI | Streamlit · live agent activations |

---

*PsycheOS — Mesa Sarah Vasantha Zephyr — mesazephyr1516@gmail.com*
