"""
PsycheOS — GRPO Training Loop
Uses Unsloth + HF TRL for compute-efficient LoRA fine-tuning.
Run in Google Colab with HuggingFace compute credits.

Usage:
    python training/grpo_train.py --episodes 1000 --epochs 3
"""

import argparse
import json
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def train_psycheos(
    model_name: str = "unsloth/llama-3-8b-instruct",
    n_episodes: int = 1000,
    epochs: int = 3,
    batch_size: int = 4,
    lr: float = 2e-4,
    output_dir: str = "./psycheos_checkpoints",
):
    """
    Main GRPO training loop for PsycheOS.
    Trains the empathy + detector agents via GRPO with LoRA adapters.
    """
    print("=" * 60)
    print("PsycheOS GRPO Training Loop")
    print("=" * 60)
    print(f"Model:    {model_name}")
    print(f"Episodes: {n_episodes}")
    print(f"Epochs:   {epochs}")
    print(f"LR:       {lr}")
    print()

    # ── Load model with Unsloth ───────────────────────────────────────────────
    try:
        from unsloth import FastLanguageModel
        print("Loading model with Unsloth (4-bit)...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=2048,
            load_in_4bit=True,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            use_gradient_checkpointing=True,
        )
        print("Model loaded.\n")
    except ImportError:
        print("Unsloth not available. Install: pip install unsloth")
        print("Continuing with mock training loop for demonstration...\n")
        model = None
        tokenizer = None

    # ── Generate training episodes ────────────────────────────────────────────
    print(f"Generating {n_episodes} patient episodes...")
    from environment.psycheos_env import PsycheOSEnv
    from reward.reward_functions import composite_reward

    env = PsycheOSEnv()
    episodes = env.generate_episodes(n=n_episodes)
    print(f"Generated {len(episodes)} episodes across 5 difficulty levels.\n")

    # ── Format prompts ────────────────────────────────────────────────────────
    def format_prompt(episode: dict) -> str:
        obs = episode["observation"]
        return f"""You are a compassionate mental health support agent.

Patient message: {obs['message']}
Session: {obs['session']}
Prior context: {obs.get('session_history', [])}

Respond with JSON:
{{
  "distress_level": <1-5>,
  "response": "<empathetic response>",
  "escalate": <true/false>
}}"""

    def reward_fn(response: str, episode: dict) -> float:
        """Reward function for GRPO."""
        import json as j
        try:
            clean = response.strip().replace("```json", "").replace("```", "")
            parsed = j.loads(clean)
            pred_level = int(parsed.get("distress_level", 1))
            resp_text = parsed.get("response", "")
            escalated = bool(parsed.get("escalate", False))
        except Exception:
            return -1.0

        rewards = composite_reward(
            pred_level=pred_level,
            true_level=episode["true_distress"],
            response=resp_text,
            escalated=escalated,
            ground_truth_escalate=episode["ground_truth_escalate"],
            distress_level=pred_level,
        )
        return rewards["composite"]

    # ── GRPO Training ─────────────────────────────────────────────────────────
    if model is not None:
        try:
            from trl import GRPOTrainer, GRPOConfig

            # Build dataset
            dataset = [
                {
                    "prompt": format_prompt(ep),
                    "episode": ep,
                }
                for ep in episodes
            ]

            config = GRPOConfig(
                output_dir=output_dir,
                num_train_epochs=epochs,
                per_device_train_batch_size=batch_size,
                learning_rate=lr,
                logging_steps=10,
                save_steps=100,
                warmup_ratio=0.1,
                lr_scheduler_type="cosine",
                bf16=True,
            )

            trainer = GRPOTrainer(
                model=model,
                config=config,
                train_dataset=dataset,
                reward_funcs=[lambda outputs, refs: [reward_fn(o, r["episode"]) for o, r in zip(outputs, refs)]],
            )

            print("Starting GRPO training...")
            trainer.train()
            print(f"\nTraining complete. Checkpoints saved to {output_dir}")
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

        except ImportError as e:
            print(f"TRL not available: {e}")
            print("Install: pip install trl")
            _mock_training(episodes, reward_fn, format_prompt, epochs)
    else:
        _mock_training(episodes, reward_fn, format_prompt, epochs)


def _mock_training(episodes, reward_fn, format_prompt, epochs):
    """
    Mock training loop to demonstrate reward curves without a real model.
    Useful for testing the environment and reward functions.
    """
    import random
    import json

    print("Running mock training loop (no model)...")
    print("This demonstrates the reward curve shape.\n")

    reward_log = []
    for epoch in range(epochs):
        epoch_rewards = []
        for i, ep in enumerate(episodes[:100]):
            # Simulate improving agent
            base_quality = 0.3 + epoch * 0.1 + i * 0.001
            mock_response = json.dumps({
                "distress_level": ep["true_distress"] + random.randint(-1, 1),
                "response": "I hear you and I'm here to support you through this.",
                "escalate": ep["ground_truth_escalate"],
            })
            r = reward_fn(mock_response, ep)
            epoch_rewards.append(r)

        avg = sum(epoch_rewards) / len(epoch_rewards)
        reward_log.append(avg)
        print(f"Epoch {epoch+1}/{epochs} | Avg Reward: {avg:.4f} | Episodes: {len(epoch_rewards)}")

    print("\nMock training complete.")
    print(f"Reward progression: {' → '.join(f'{r:.3f}' for r in reward_log)}")

    # Save log
    with open("./psycheos_reward_log.json", "w") as f:
        json.dump(reward_log, f)
    print("Reward log saved to ./psycheos_reward_log.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PsycheOS with GRPO")
    parser.add_argument("--model", default="unsloth/llama-3-8b-instruct")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--output_dir", default="./psycheos_checkpoints")
    args = parser.parse_args()

    train_psycheos(
        model_name=args.model,
        n_episodes=args.episodes,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        output_dir=args.output_dir,
    )
