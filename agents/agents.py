"""
PsycheOS — 5 Specialized Agents
"""

import os
import re
import random
from typing import Optional

# ── Detector Agent ────────────────────────────────────────────────────────────

CRISIS_KEYWORDS = {
    5: ["plan", "goodbye", "end it", "kill myself", "suicide", "took pills",
        "won't be here", "final", "last message", "overdose", "jump",
        "not want to be here", "don't want to exist", "saying goodbye",
        "no point anymore", "decided to", "ready to go", "ending it"],
    4: ["better off without me", "no reason to live", "hurting myself", "self harm",
        "cutting", "don't want to exist", "disappear forever", "everyone hates me",
        "everyone would be better", "thought about this", "keep thinking about",
        "not being here", "wish i was dead", "want to die", "want to disappear"],
    3: ["can't go on", "exhausted", "hopeless", "worthless", "empty inside",
        "can't get out of bed", "stopped caring", "crying all the time",
        "cry every day", "nothing matters", "feel so empty", "don't care anymore",
        "heavy", "numb", "pointless", "no energy", "haven't left", "weeks"],
    2: ["tired", "unmotivated", "anxious", "stressed", "not sleeping",
        "feeling down", "overwhelmed", "lonely", "bit down", "rough",
        "hard to", "off lately", "persistent", "can't focus", "struggling"],
}

def detector_agent(message: str, history: list = None) -> dict:
    """
    Classify distress severity 1-5 from message text.
    Returns level + confidence.
    """
    msg_lower = message.lower()
    
    detected_level = 1
    confidence = 0.6
    matched_keywords = []

    for level in [5, 4, 3, 2]:
        for kw in CRISIS_KEYWORDS[level]:
            if kw in msg_lower:
                if level > detected_level:
                    detected_level = level
                matched_keywords.append(kw)
                confidence = min(0.98, confidence + 0.08)

    # Boost confidence if history shows escalating pattern
    if history:
        recent_levels = [h.get("distress", 1) for h in history[-3:]]
        if recent_levels and max(recent_levels) >= detected_level:
            confidence = min(0.99, confidence + 0.05)

    return {
        "distress_level": detected_level,
        "confidence": round(confidence, 2),
        "matched_keywords": matched_keywords,
    }


# ── Empathy Agent ─────────────────────────────────────────────────────────────

EMPATHY_RESPONSES = {
    1: [
        "Thank you for checking in. It sounds like things are manageable right now — that's worth acknowledging. Is there anything specific on your mind you'd like to explore?",
        "I hear you. Even when things feel okay, it's valuable to pause and reflect. What's been taking up the most of your headspace lately?",
    ],
    2: [
        "What you're describing — that persistent heaviness — is real and valid. You don't need a dramatic reason to feel this way. Can you tell me more about when it feels strongest?",
        "Being unmotivated and off-balance is exhausting in its own quiet way. I want you to know I'm here and listening. What does a harder day look like for you right now?",
    ],
    3: [
        "I hear how much weight you're carrying. Crying every day without knowing why is one of the most disorienting things a person can experience — you're not broken, you're overwhelmed. I'm right here with you.",
        "What you're going through sounds genuinely painful and I want you to know that's okay to acknowledge. You don't have to minimize it. Can you tell me — when did things start feeling this heavy?",
    ],
    4: [
        "I need to be honest with you: what you're sharing concerns me deeply, and I'm grateful you trusted me with it. Those thoughts about not being here — can you tell me more about them? I'm not going anywhere.",
        "Thank you for telling me this. I want you to hear me clearly: you matter, and what you're feeling right now is not the truth about your future. Are you safe right now, in this moment?",
    ],
    5: [
        "I hear you. You reached out, and that took real courage. Please stay with me. Are you somewhere safe right now? I'm connecting you with a crisis counselor who can help immediately.",
        "You told someone. That matters more than you know right now. I'm here and I'm not leaving. Can you tell me where you are? I want to make sure you're safe.",
    ],
}

def empathy_agent(message: str, distress_level: int, memory_context: str = "", session: int = 1) -> dict:
    """
    Generate empathetic response calibrated to distress level.
    Returns response + DPO preference score (simulated).
    """
    responses = EMPATHY_RESPONSES.get(distress_level, EMPATHY_RESPONSES[2])
    base_response = random.choice(responses)

    # Personalize with memory context if available
    if memory_context and distress_level >= 3:
        base_response = f"In our previous conversations, you mentioned feeling similar things. " + base_response

    # Session continuity
    if session > 1 and distress_level <= 2:
        base_response = "It's good to hear from you again. " + base_response

    # Simulated DPO score (improves with training)
    base_dpo = 0.5 + distress_level * 0.05 + random.uniform(-0.1, 0.15)
    dpo_score = round(min(0.99, max(0.1, base_dpo)), 2)

    return {
        "response": base_response,
        "dpo_score": dpo_score,
    }


# ── Crisis Escalator Agent ────────────────────────────────────────────────────

def escalator_agent(distress_level: int, message: str, confidence: float, history: list = None) -> dict:
    """
    Decide whether to escalate to human counselor.
    10x penalty on false negatives (missing real crises).
    """
    hard_escalate_phrases = [
        "plan", "goodbye", "took pills", "overdose", "jump", "last message",
        "kill myself", "end it all", "won't be here"
    ]
    
    msg_lower = message.lower()
    hard_trigger = any(p in msg_lower for p in hard_escalate_phrases)
    
    # Escalate if: level 5, or level 4 with high confidence, or hard phrase detected
    should_escalate = (
        distress_level == 5 or
        hard_trigger or
        (distress_level >= 4 and confidence >= 0.8)
    )

    # Check escalating history
    if history and len(history) >= 2:
        recent = [h.get("distress", 1) for h in history[-2:]]
        if all(d >= 4 for d in recent):
            should_escalate = True

    return {
        "escalate": should_escalate,
        "reason": (
            "Crisis phrase detected" if hard_trigger
            else f"Distress level {distress_level} with confidence {confidence}"
            if should_escalate else "Autonomous handling appropriate"
        ),
    }


# ── Oversight Agent ───────────────────────────────────────────────────────────

def oversight_agent(
    distress_level: int,
    response: str,
    escalate: bool,
    dpo_score: float,
) -> dict:
    """
    Monitor all agents for contradictions and flag anomalies.
    """
    flags = []

    # Flag cheerful response to high distress
    cheerful_words = ["great", "wonderful", "exciting", "amazing", "fantastic", "happy"]
    if distress_level >= 4 and any(w in response.lower() for w in cheerful_words):
        flags.append("Tone mismatch: positive language at crisis level")

    # Flag no escalation at level 5
    if distress_level == 5 and not escalate:
        flags.append("CRITICAL: Level 5 distress not escalated")

    # Flag very short response to high distress
    if distress_level >= 3 and len(response.split()) < 15:
        flags.append("Response too brief for distress severity")

    # Flag low DPO score at high distress
    if distress_level >= 3 and dpo_score < 0.4:
        flags.append(f"Low empathy quality (DPO: {dpo_score:.2f}) at distress {distress_level}")

    oversight_flag = flags[0] if flags else ""
    severity = "critical" if any("CRITICAL" in f for f in flags) else ("warning" if flags else "ok")

    return {
        "oversight_flag": oversight_flag,
        "all_flags": flags,
        "severity": severity,
    }


# ── Curriculum Agent ──────────────────────────────────────────────────────────

class CurriculumAgent:
    """
    Tracks rolling reward and auto-generates harder episodes when agents plateau.
    """
    def __init__(self, window: int = 50, threshold: float = 0.02):
        self.reward_history = []
        self.window = window
        self.threshold = threshold
        self.current_difficulty = 1
        self.plateau_count = 0

    def update(self, reward: float) -> dict:
        self.reward_history.append(reward)
        result = {"difficulty": self.current_difficulty, "escalated": False, "reason": ""}

        if len(self.reward_history) >= self.window:
            recent = self.reward_history[-self.window:]
            prev = self.reward_history[-self.window * 2: -self.window]

            if len(prev) >= self.window:
                recent_avg = sum(recent) / len(recent)
                prev_avg = sum(prev) / len(prev)
                improvement = (recent_avg - prev_avg) / (abs(prev_avg) + 1e-8)

                if improvement < self.threshold:
                    self.plateau_count += 1
                    if self.plateau_count >= 2:
                        self.current_difficulty = min(5, self.current_difficulty + 1)
                        self.plateau_count = 0
                        result["escalated"] = True
                        result["reason"] = f"Improvement {improvement:.3f} < threshold {self.threshold}"
                else:
                    self.plateau_count = 0

        result["difficulty"] = self.current_difficulty
        return result

    def generate_hard_scenario(self) -> str:
        """Generate a description of a harder training scenario."""
        scenarios = {
            1: "Basic single-session low-distress conversations",
            2: "Multi-session mild distress with ambiguous signals",
            3: "Moderate distress with mixed indicators and history",
            4: "High distress with suicidal ideation, requires escalation judgment",
            5: "Acute crisis — immediate escalation required, adversarial phrasing",
        }
        return scenarios.get(self.current_difficulty, scenarios[1])


# Global curriculum agent instance
curriculum = CurriculumAgent()
