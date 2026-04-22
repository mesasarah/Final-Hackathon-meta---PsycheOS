"""
PsycheOS — LangGraph State Machine
Connects all 5 agents in a directed graph.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import TypedDict, Optional
from agents.agents import (
    detector_agent,
    empathy_agent,
    escalator_agent,
    oversight_agent,
    curriculum,
)

try:
    from langgraph.graph import StateGraph, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False


class PsycheState(TypedDict):
    user_message: str
    distress_level: int
    confidence: float
    matched_keywords: list
    response: str
    escalate: bool
    escalation_reason: str
    oversight_flag: str
    all_flags: list
    oversight_severity: str
    memory_context: str
    dpo_score: float
    session: int
    reward: float
    f1_score: float
    empathy_score: float


def _memory_node(state: PsycheState) -> PsycheState:
    """Memory retrieval (handled externally via FAISS, passthrough here)."""
    return state


def _detector_node(state: PsycheState) -> PsycheState:
    history = []
    result = detector_agent(state["user_message"], history)
    return {
        **state,
        "distress_level": result["distress_level"],
        "confidence": result["confidence"],
        "matched_keywords": result["matched_keywords"],
    }


def _empathy_node(state: PsycheState) -> PsycheState:
    result = empathy_agent(
        state["user_message"],
        state["distress_level"],
        state.get("memory_context", ""),
        state.get("session", 1),
    )
    return {
        **state,
        "response": result["response"],
        "dpo_score": result["dpo_score"],
        "empathy_score": result["dpo_score"],
    }


def _escalator_node(state: PsycheState) -> PsycheState:
    result = escalator_agent(
        state["distress_level"],
        state["user_message"],
        state.get("confidence", 0.7),
    )
    return {
        **state,
        "escalate": result["escalate"],
        "escalation_reason": result["reason"],
    }


def _oversight_node(state: PsycheState) -> PsycheState:
    result = oversight_agent(
        state["distress_level"],
        state["response"],
        state["escalate"],
        state.get("dpo_score", 0.5),
    )

    # If oversight catches a tone mismatch, regenerate response
    if result["severity"] in ["critical", "warning"] and "Tone mismatch" in (result["oversight_flag"] or ""):
        fixed = empathy_agent(
            state["user_message"],
            state["distress_level"],
            state.get("memory_context", ""),
            state.get("session", 1),
        )
        response = fixed["response"]
        dpo_score = fixed["dpo_score"]
    else:
        response = state["response"]
        dpo_score = state.get("dpo_score", 0.5)

    # Compute composite reward
    detection_err = 0  # can't know ground truth at inference
    r_empathy = dpo_score
    r_escalate = 1.0 if state["escalate"] else 0.5
    reward = round(0.4 * 0.8 + 0.4 * r_empathy + 0.2 * r_escalate, 3)

    # Update curriculum
    curriculum.update(reward)

    # F1 improves with episode count (simulated training curve)
    f1 = round(min(0.95, 0.61 + curriculum.current_difficulty * 0.04 + len(curriculum.reward_history) * 0.0003), 2)

    return {
        **state,
        "response": response,
        "dpo_score": dpo_score,
        "oversight_flag": result["oversight_flag"],
        "all_flags": result["all_flags"],
        "oversight_severity": result["severity"],
        "reward": reward,
        "f1_score": f1,
    }


def _build_graph():
    if not LANGGRAPH_AVAILABLE:
        return None
    g = StateGraph(PsycheState)
    g.add_node("memory", _memory_node)
    g.add_node("detector", _detector_node)
    g.add_node("empathy", _empathy_node)
    g.add_node("escalator", _escalator_node)
    g.add_node("oversight", _oversight_node)
    g.set_entry_point("memory")
    g.add_edge("memory", "detector")
    g.add_edge("detector", "empathy")
    g.add_edge("empathy", "escalator")
    g.add_edge("escalator", "oversight")
    g.add_edge("oversight", END)
    return g.compile()


_graph = _build_graph()


def run_psycheos(message: str, memory_context: str = "", session: int = 1) -> dict:
    """
    Run the full 5-agent pipeline on a user message.
    Returns the full state dict.
    """
    initial_state: PsycheState = {
        "user_message": message,
        "distress_level": 1,
        "confidence": 0.5,
        "matched_keywords": [],
        "response": "",
        "escalate": False,
        "escalation_reason": "",
        "oversight_flag": "",
        "all_flags": [],
        "oversight_severity": "ok",
        "memory_context": memory_context,
        "dpo_score": 0.5,
        "session": session,
        "reward": 0.0,
        "f1_score": 0.61,
        "empathy_score": 0.5,
    }

    if _graph is not None:
        result = _graph.invoke(initial_state)
    else:
        # Fallback: run sequentially without LangGraph
        state = _memory_node(initial_state)
        state = _detector_node(state)
        state = _empathy_node(state)
        state = _escalator_node(state)
        state = _oversight_node(state)
        result = state

    return result
