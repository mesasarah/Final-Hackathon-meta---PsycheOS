"""
PsycheOS — DPO Fine-Tuning for Empathy Agent
Simulates psychiatrist personas labeling preferred vs rejected responses.
Satisfies Snorkel AI bonus requirement (changing expert preferences).
"""

import json
import random
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── Psychiatrist Persona Definitions ─────────────────────────────────────────

PSYCHIATRIST_PERSONAS = [
    {
        "name": "Dr. Priya Nair",
        "specialty": "Crisis intervention",
        "preferences": {
            "preferred_length": "long",
            "tone": "warm_direct",
            "priorities": ["safety", "validation", "action"],
            "dislikes": ["minimizing", "advice_giving_early", "cheerfulness"],
        },
    },
    {
        "name": "Dr. Arjun Mehta",
        "specialty": "CBT-focused",
        "preferences": {
            "preferred_length": "medium",
            "tone": "structured_warm",
            "priorities": ["reflection", "grounding", "validation"],
            "dislikes": ["open_loops", "vague_comfort"],
        },
    },
    {
        "name": "Dr. Sarah Thomas",
        "specialty": "Trauma-informed care",
        "preferences": {
            "preferred_length": "medium",
            "tone": "gentle_slow",
            "priorities": ["safety", "pacing", "no_pressure"],
            "dislikes": ["urgency", "clinical_language", "rapid_solutions"],
        },
    },
]


def generate_dpo_pairs(n: int = 500, persona_shift: bool = True) -> list:
    """
    Generate preferred/rejected response pairs for DPO training.
    persona_shift=True simulates changing expert preferences over time.
    """
    from agents.agents import empathy_agent

    pairs = []
    for i in range(n):
        distress_level = random.randint(1, 5)

        # Sample message
        messages = {
            1: "I've been okay, just a bit tired.",
            2: "Things have been rough. Hard to stay motivated.",
            3: "I cry every day. I feel so empty inside.",
            4: "I keep thinking everyone would be better off without me.",
            5: "I have a plan. I don't want to be here anymore.",
        }
        message = messages[distress_level]

        # Select persona (shifts at n//2 if persona_shift=True)
        if persona_shift and i > n // 2:
            persona = random.choice(PSYCHIATRIST_PERSONAS[1:])
        else:
            persona = PSYCHIATRIST_PERSONAS[0]

        # Generate two candidate responses
        r1 = empathy_agent(message, distress_level)
        r2 = empathy_agent(message, distress_level)

        # Score based on persona preferences
        def score_for_persona(response: str, persona: dict) -> float:
            score = 0.5
            prefs = persona["preferences"]
            words = response.lower().split()

            if prefs["preferred_length"] == "long" and len(words) >= 40:
                score += 0.2
            elif prefs["preferred_length"] == "medium" and 20 <= len(words) <= 50:
                score += 0.2

            for p in prefs["priorities"]:
                if p == "safety" and any(w in response.lower() for w in ["safe", "here", "help"]):
                    score += 0.1
                if p == "validation" and any(w in response.lower() for w in ["valid", "hear", "understand"]):
                    score += 0.1
                if p == "action" and "?" in response:
                    score += 0.05

            for d in prefs["dislikes"]:
                if d == "minimizing" and any(w in response.lower() for w in ["just", "try to", "simply"]):
                    score -= 0.15
                if d == "cheerfulness" and any(w in response.lower() for w in ["great", "wonderful"]):
                    score -= 0.2

            return round(max(0.0, min(1.0, score)), 3)

        s1 = score_for_persona(r1["response"], persona)
        s2 = score_for_persona(r2["response"], persona)

        if s1 >= s2:
            chosen = r1["response"]
            rejected = r2["response"]
        else:
            chosen = r2["response"]
            rejected = r1["response"]

        pairs.append({
            "prompt": f"[Distress {distress_level}] Patient: {message}",
            "chosen": chosen,
            "rejected": rejected,
            "persona": persona["name"],
            "distress_level": distress_level,
            "score_chosen": max(s1, s2),
            "score_rejected": min(s1, s2),
        })

    return pairs


def run_dpo_finetuning(
    model_name: str = "unsloth/llama-3-8b-instruct",
    n_pairs: int = 500,
    output_dir: str = "./psycheos_dpo_checkpoints",
):
    """Run DPO fine-tuning on the empathy agent."""
    print("=" * 60)
    print("PsycheOS DPO Fine-Tuning — Empathy Agent")
    print("=" * 60)

    print(f"Generating {n_pairs} preference pairs from psychiatrist personas...")
    pairs = generate_dpo_pairs(n=n_pairs)

    # Save dataset
    os.makedirs(output_dir, exist_ok=True)
    dataset_path = os.path.join(output_dir, "dpo_dataset.json")
    with open(dataset_path, "w") as f:
        json.dump(pairs, f, indent=2)
    print(f"Dataset saved: {dataset_path}")

    # Stats
    avg_score = sum(p["score_chosen"] for p in pairs) / len(pairs)
    win_margin = sum(p["score_chosen"] - p["score_rejected"] for p in pairs) / len(pairs)
    print(f"Avg chosen score:  {avg_score:.3f}")
    print(f"Avg win margin:    {win_margin:.3f}")
    print()

    try:
        from unsloth import FastLanguageModel
        from trl import DPOTrainer, DPOConfig
        import torch
        from datasets import Dataset

        print("Loading model with Unsloth...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=1024,
            load_in_4bit=True,
        )
        model = FastLanguageModel.get_peft_model(
            model, r=8, lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
        )

        dataset = Dataset.from_list(pairs)

        config = DPOConfig(
            output_dir=output_dir,
            num_train_epochs=2,
            per_device_train_batch_size=2,
            learning_rate=5e-5,
            beta=0.1,
            logging_steps=10,
        )

        trainer = DPOTrainer(
            model=model,
            ref_model=None,
            args=config,
            train_dataset=dataset,
            tokenizer=tokenizer,
        )

        print("Starting DPO training...")
        trainer.train()
        model.save_pretrained(output_dir)
        print(f"DPO fine-tuning complete. Saved to {output_dir}")

    except ImportError as e:
        print(f"Dependencies not available: {e}")
        print("DPO dataset generated and ready for training onsite.")
        print("Install: pip install unsloth trl datasets")


if __name__ == "__main__":
    run_dpo_finetuning()
