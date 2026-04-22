"""
PsycheOS — OpenEnv RL Environment
Mental Health Crisis POMDP
"""

import random
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import json

# ── Try importing openenv, fallback to gymnasium ─────────────────────────────
try:
    import openenv as gym
    BACKEND = "openenv"
except ImportError:
    try:
        import gymnasium as gym
        BACKEND = "gymnasium"
    except ImportError:
        import gym
        BACKEND = "gym"


@dataclass
class PatientState:
    distress_level: int = 1          # 1–5
    session_history: list = field(default_factory=list)
    crisis_indicators: dict = field(default_factory=dict)
    context: str = ""
    turn: int = 0
    session: int = 1
    stressors: list = field(default_factory=list)


class PsycheOSEnv:
    """
    OpenEnv-compliant environment for mental health crisis RL training.
    State: PatientState (POMDP)
    Actions: distress_level (1-5), response (str), escalate (bool)
    Rewards: composite of detection F1, empathy DPO, escalation precision
    """

    metadata = {"render_modes": ["text"], "name": "PsycheOS-v1"}

    DISTRESS_TEMPLATES = {
        1: [
            "I've been feeling okay lately, just a bit tired from work.",
            "Things are fine, just managing day to day stuff.",
            "Not much to share, had a decent week overall.",
        ],
        2: [
            "I've been a bit down lately, nothing serious but persistent.",
            "Finding it hard to stay motivated. Sleep has been off.",
            "Feeling kind of empty. Can't pinpoint why exactly.",
        ],
        3: [
            "I cry almost every day and I don't know why. Everything feels heavy.",
            "I haven't left the house in weeks. I can't make myself care about anything.",
            "My anxiety is out of control. I can't focus on anything. My heart races constantly.",
        ],
        4: [
            "I keep thinking about how much better things would be if I just wasn't here.",
            "I've been hurting myself to cope. It's the only thing that helps.",
            "Everyone would be better off without me. I've thought about this a lot.",
        ],
        5: [
            "I have a plan. I've thought about it every day this week. I don't see another way.",
            "I already took some pills. I don't know if I want help.",
            "I'm saying goodbye. I just needed to tell someone.",
        ]
    }

    STRESSORS = [
        "job loss", "relationship breakdown", "grief", "academic failure",
        "family conflict", "financial crisis", "chronic illness", "isolation",
        "trauma", "substance use", "housing insecurity"
    ]

    def __init__(self, difficulty: int = 1):
        self.difficulty = difficulty
        self.state: Optional[PatientState] = None
        self.max_turns = 8 + difficulty * 2
        self._episode_reward = 0.0
        self._true_distress = 0
        self._ground_truth_escalate = False

    def reset(self, seed=None, episode=None):
        if episode:
            self.state = episode
        else:
            self.state = self._generate_episode()
        self._episode_reward = 0.0
        obs = self._observe()
        return obs, {}

    def step(self, action: dict):
        assert self.state is not None, "Call reset() first"

        reward = self._compute_reward(action)
        self._episode_reward += reward

        done = self._check_done(action)
        self.state = self._transition(action)

        obs = self._observe()
        info = {
            "true_distress": self._true_distress,
            "ground_truth_escalate": self._ground_truth_escalate,
            "episode_reward": self._episode_reward,
            "turn": self.state.turn,
        }

        return obs, reward, done, False, info

    def _generate_episode(self) -> PatientState:
        """Generate a synthetic patient episode based on difficulty."""
        if self.difficulty <= 2:
            level = random.randint(1, 3)
        elif self.difficulty <= 4:
            level = random.randint(2, 4)
        else:
            level = random.randint(3, 5)

        self._true_distress = level
        self._ground_truth_escalate = level >= 4

        stressors = random.sample(self.STRESSORS, k=min(self.difficulty, 3))
        history = []
        if self.difficulty >= 3:
            prev_level = max(1, level - random.randint(0, 2))
            prev_msg = random.choice(self.DISTRESS_TEMPLATES[prev_level])
            history.append({"turn": 0, "message": prev_msg, "distress": prev_level})

        opening = random.choice(self.DISTRESS_TEMPLATES[level])
        context = f"[Stressors: {', '.join(stressors)}] {opening}"

        return PatientState(
            distress_level=level,
            session_history=history,
            crisis_indicators={
                "suicidal_ideation": level >= 4,
                "self_harm": level >= 4,
                "hopelessness": level >= 3,
            },
            context=context,
            stressors=stressors,
        )

    def _observe(self) -> dict:
        return {
            "message": self.state.context,
            "session_history": self.state.session_history[-3:],
            "turn": self.state.turn,
            "session": self.state.session,
        }

    def _compute_reward(self, action: dict) -> float:
        pred_distress = action.get("distress_level", 1)
        response = action.get("response", "")
        escalate = action.get("escalate", False)

        # Detection reward (F1 proxy)
        detection_err = abs(pred_distress - self._true_distress)
        r_detect = max(0.0, 1.0 - detection_err * 0.3)

        # Empathy reward (length + keyword heuristic as DPO proxy)
        keywords = ["hear", "understand", "safe", "here", "feel", "support", "care"]
        keyword_score = sum(1 for k in keywords if k in response.lower()) / len(keywords)
        length_score = min(1.0, len(response.split()) / 40)
        r_empathy = 0.6 * keyword_score + 0.4 * length_score

        # Escalation reward (10x penalty on false negatives)
        if self._ground_truth_escalate and not escalate:
            r_escalate = -10.0
        elif not self._ground_truth_escalate and escalate:
            r_escalate = -1.0
        elif self._ground_truth_escalate and escalate:
            r_escalate = 3.0
        else:
            r_escalate = 1.0

        return 0.4 * r_detect + 0.4 * r_empathy + 0.2 * r_escalate

    def _check_done(self, action: dict) -> bool:
        if action.get("escalate", False):
            return True
        if self.state.turn >= self.max_turns:
            return True
        return False

    def _transition(self, action: dict) -> PatientState:
        self.state.session_history.append({
            "turn": self.state.turn,
            "message": self.state.context,
            "distress": self._true_distress,
            "response": action.get("response", ""),
        })
        self.state.turn += 1

        # Slight distress drift based on empathy quality
        response = action.get("response", "")
        keywords = ["hear", "understand", "safe", "here"]
        good_response = any(k in response.lower() for k in keywords)
        if good_response and self._true_distress > 1:
            self._true_distress = max(1, self._true_distress - 1)

        return self.state

    def generate_episodes(self, n: int = 100) -> list:
        """Generate n synthetic episodes for training."""
        episodes = []
        for i in range(n):
            diff = min(5, 1 + i // (n // 5))
            old_diff = self.difficulty
            self.difficulty = diff
            obs, _ = self.reset()
            episodes.append({
                "observation": obs,
                "true_distress": self._true_distress,
                "ground_truth_escalate": self._ground_truth_escalate,
                "difficulty": diff,
            })
            self.difficulty = old_diff
        return episodes

    def render(self, mode="text"):
        if self.state:
            print(f"Turn {self.state.turn} | Distress: {self._true_distress}/5")
            print(f"Message: {self.state.context[:100]}...")


# Register with gym if available
try:
    gym.register(
        id="PsycheOS-v1",
        entry_point="environment.psycheos_env:PsycheOSEnv",
    )
except Exception:
    pass
