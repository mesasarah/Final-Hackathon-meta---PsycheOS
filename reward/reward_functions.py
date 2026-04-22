"""
PsycheOS — Reward Functions
5 distinct reward signals for the RL training loop.
"""

import math
from typing import Optional


def detection_reward(pred_level: int, true_level: int) -> float:
    """
    F1-proxy reward for distress classification.
    Penalizes under-detection more than over-detection.
    """
    err = true_level - pred_level
    if err == 0:
        return 1.0
    elif err > 0:
        # Under-detecting is worse (missed crises)
        return max(0.0, 1.0 - err * 0.4)
    else:
        # Over-detecting is bad but less bad
        return max(0.0, 1.0 - abs(err) * 0.25)


def empathy_reward(response: str, distress_level: int, dpo_score: Optional[float] = None) -> float:
    """
    DPO preference margin reward.
    If dpo_score provided (from psychiatrist persona), use it directly.
    Otherwise compute from heuristic keywords.
    """
    if dpo_score is not None:
        return float(dpo_score)

    empathy_keywords = {
        "hear": 0.15, "understand": 0.15, "safe": 0.12, "here": 0.10,
        "feel": 0.10, "support": 0.12, "care": 0.10, "not alone": 0.16,
        "valid": 0.12, "matter": 0.12, "grateful": 0.08,
    }
    harmful_keywords = ["just", "cheer up", "could be worse", "stop overthinking", "get over"]

    score = 0.3  # base
    for kw, weight in empathy_keywords.items():
        if kw in response.lower():
            score += weight

    for hw in harmful_keywords:
        if hw in response.lower():
            score -= 0.2

    # Length bonus (40+ words is good for distress >= 3)
    word_count = len(response.split())
    if distress_level >= 3:
        score += min(0.2, word_count / 200)

    return round(min(1.0, max(0.0, score)), 3)


def escalation_reward(escalated: bool, ground_truth_escalate: bool) -> float:
    """
    Binary escalation precision/recall.
    10x penalty on false negatives (missing real crises).
    """
    if ground_truth_escalate and not escalated:
        return -10.0   # False negative: CRITICAL
    elif ground_truth_escalate and escalated:
        return 3.0     # True positive: Crisis correctly caught
    elif not ground_truth_escalate and escalated:
        return -1.0    # False positive: Unnecessary escalation
    else:
        return 1.0     # True negative: Correct autonomous handling


def memory_reward(retrieved_context: str, ground_truth_context: str) -> float:
    """
    ROUGE-1 proxy for memory retrieval relevance.
    """
    if not retrieved_context or not ground_truth_context:
        return 0.0

    retrieved_words = set(retrieved_context.lower().split())
    truth_words = set(ground_truth_context.lower().split())

    if not truth_words:
        return 0.0

    overlap = len(retrieved_words & truth_words)
    precision = overlap / max(len(retrieved_words), 1)
    recall = overlap / max(len(truth_words), 1)

    if precision + recall == 0:
        return 0.0

    f1 = 2 * precision * recall / (precision + recall)
    return round(f1, 3)


def curriculum_reward(episode_difficulty: int, prev_difficulty: int, agent_reward: float) -> float:
    """
    Reward for curriculum agent's difficulty escalation decisions.
    Good: escalate when agent plateaus. Bad: escalate when agent still improving.
    """
    escalated = episode_difficulty > prev_difficulty

    if escalated and agent_reward > 0.7:
        # Escalated but agent was doing well — premature
        return -0.3
    elif escalated and agent_reward <= 0.5:
        # Escalated because agent plateaued — correct
        return 1.0
    elif not escalated and agent_reward > 0.7:
        # Correctly staying at difficulty
        return 0.5
    else:
        # Not escalating when should
        return 0.0


def composite_reward(
    pred_level: int,
    true_level: int,
    response: str,
    escalated: bool,
    ground_truth_escalate: bool,
    distress_level: int,
    dpo_score: Optional[float] = None,
    retrieved_context: str = "",
    ground_truth_context: str = "",
) -> dict:
    """
    Compute all reward signals and return the composite.
    Weights: detect 0.4, empathy 0.4, escalation 0.2
    """
    r_detect = detection_reward(pred_level, true_level)
    r_empathy = empathy_reward(response, distress_level, dpo_score)
    r_escalate = escalation_reward(escalated, ground_truth_escalate)
    r_memory = memory_reward(retrieved_context, ground_truth_context)

    composite = (
        0.40 * r_detect +
        0.40 * r_empathy +
        0.20 * max(-1.0, min(1.0, r_escalate / 3.0))  # normalize escalation
    )

    return {
        "composite": round(composite, 4),
        "detection": round(r_detect, 4),
        "empathy": round(r_empathy, 4),
        "escalation": round(r_escalate, 4),
        "memory": round(r_memory, 4),
    }
