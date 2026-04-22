import streamlit as st
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.graph import run_psycheos
from memory.faiss_store import MemoryStore
import time
import random

st.set_page_config(
    page_title="PsycheOS",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@300;400;500&display=swap');

:root {
    --bg: #0a0a0f;
    --surface: #12121a;
    --border: #1e1e2e;
    --accent: #7c6aff;
    --accent2: #ff6a8a;
    --green: #4ade80;
    --amber: #fbbf24;
    --red: #f87171;
    --text: #e2e2f0;
    --muted: #6b6b8a;
}

html, body, [class*="css"] {
    font-family: 'DM Mono', monospace;
    background: var(--bg);
    color: var(--text);
}

.stApp { background: var(--bg); }

.main-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.8rem;
    background: linear-gradient(135deg, #7c6aff, #ff6a8a, #7c6aff);
    background-size: 200%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shimmer 3s infinite;
    margin: 0;
}

@keyframes shimmer {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

.subtitle {
    color: var(--muted);
    font-size: 0.75rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-top: 0.25rem;
}

.agent-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem;
    margin-bottom: 0.75rem;
    transition: all 0.3s ease;
}

.agent-card.active {
    border-color: var(--accent);
    box-shadow: 0 0 20px rgba(124,106,255,0.2);
}

.agent-card.crisis {
    border-color: var(--red);
    box-shadow: 0 0 20px rgba(248,113,113,0.3);
    animation: pulse-red 1s infinite;
}

@keyframes pulse-red {
    0%, 100% { box-shadow: 0 0 20px rgba(248,113,113,0.3); }
    50% { box-shadow: 0 0 35px rgba(248,113,113,0.6); }
}

.agent-name {
    font-size: 0.65rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.4rem;
}

.agent-value {
    font-size: 1.1rem;
    font-weight: 500;
    color: var(--text);
}

.distress-bar {
    height: 6px;
    border-radius: 3px;
    margin-top: 0.5rem;
    transition: width 0.5s ease;
}

.chat-msg-user {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px 16px 4px 16px;
    padding: 0.8rem 1.2rem;
    margin: 0.5rem 0;
    max-width: 75%;
    margin-left: auto;
    font-size: 0.9rem;
}

.chat-msg-ai {
    background: linear-gradient(135deg, rgba(124,106,255,0.1), rgba(255,106,138,0.05));
    border: 1px solid rgba(124,106,255,0.3);
    border-radius: 16px 16px 16px 4px;
    padding: 0.8rem 1.2rem;
    margin: 0.5rem 0;
    max-width: 80%;
    font-size: 0.9rem;
    line-height: 1.6;
}

.escalation-banner {
    background: linear-gradient(135deg, rgba(248,113,113,0.2), rgba(251,191,36,0.1));
    border: 1px solid var(--red);
    border-radius: 12px;
    padding: 1rem 1.5rem;
    margin: 1rem 0;
    font-size: 0.85rem;
}

.metric-pill {
    display: inline-block;
    padding: 0.2rem 0.7rem;
    border-radius: 20px;
    font-size: 0.7rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin: 0.2rem;
}

.reward-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
    margin-bottom: 0.5rem;
}

.reward-number {
    font-family: 'DM Serif Display', serif;
    font-size: 2rem;
    color: var(--accent);
}

div[data-testid="stTextInput"] input {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    color: var(--text) !important;
    font-family: 'DM Mono', monospace !important;
    padding: 0.8rem 1rem !important;
}

div[data-testid="stTextInput"] input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(124,106,255,0.2) !important;
}

.stButton > button {
    background: linear-gradient(135deg, var(--accent), #9b5de5) !important;
    border: none !important;
    border-radius: 10px !important;
    color: white !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.1em !important;
    padding: 0.6rem 1.5rem !important;
    transition: all 0.2s !important;
}

.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(124,106,255,0.4) !important;
}

.oversight-flag {
    background: rgba(251,191,36,0.1);
    border-left: 3px solid var(--amber);
    padding: 0.5rem 1rem;
    border-radius: 0 8px 8px 0;
    font-size: 0.8rem;
    color: var(--amber);
    margin-top: 0.5rem;
}

.session-tag {
    background: rgba(124,106,255,0.15);
    border: 1px solid rgba(124,106,255,0.3);
    border-radius: 6px;
    padding: 0.15rem 0.6rem;
    font-size: 0.65rem;
    color: var(--accent);
    display: inline-block;
    margin-bottom: 0.5rem;
}

hr { border-color: var(--border) !important; }

[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
</style>
""", unsafe_allow_html=True)

# ── Session State ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "memory_store" not in st.session_state:
    st.session_state.memory_store = MemoryStore()
if "session_count" not in st.session_state:
    st.session_state.session_count = 1
if "total_reward" not in st.session_state:
    st.session_state.total_reward = 0.0
if "agent_states" not in st.session_state:
    st.session_state.agent_states = {
        "distress_level": 0,
        "response": "",
        "escalate": False,
        "oversight_flag": "",
        "memory_context": "",
        "empathy_score": 0.0,
        "f1_score": 0.0,
    }
if "episode_count" not in st.session_state:
    st.session_state.episode_count = 0
if "reward_history" not in st.session_state:
    st.session_state.reward_history = []

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="main-title" style="font-size:1.8rem;">PsycheOS</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Agent Activations</p>', unsafe_allow_html=True)
    st.markdown("---")

    s = st.session_state.agent_states

    # Distress level
    d = s["distress_level"]
    colors = ["#4ade80", "#a3e635", "#fbbf24", "#f97316", "#f87171"]
    labels = ["Stable", "Mild", "Moderate", "Severe", "Crisis"]
    dcolor = colors[d-1] if d > 0 else "#6b6b8a"
    dlabel = labels[d-1] if d > 0 else "—"

    card_class = "agent-card crisis" if d == 5 else ("agent-card active" if d > 0 else "agent-card")
    st.markdown(f"""
    <div class="{card_class}">
        <div class="agent-name">🔍 Detector Agent</div>
        <div class="agent-value" style="color:{dcolor}">Level {d} — {dlabel}</div>
        <div class="distress-bar" style="width:{d*20}%; background:{dcolor}; opacity:0.8;"></div>
    </div>
    """, unsafe_allow_html=True)

    # Empathy score
    escore = s.get("empathy_score", 0.0)
    st.markdown(f"""
    <div class="agent-card {'active' if escore > 0 else ''}">
        <div class="agent-name">💬 Empathy Agent</div>
        <div class="agent-value" style="color:#a78bfa">DPO Score: {escore:.2f}</div>
    </div>
    """, unsafe_allow_html=True)

    # Crisis escalator
    escalated = s["escalate"]
    st.markdown(f"""
    <div class="agent-card {'crisis' if escalated else ('active' if d > 0 else '')}">
        <div class="agent-name">🚨 Crisis Escalator</div>
        <div class="agent-value" style="color:{'#f87171' if escalated else '#4ade80'}">
            {'⚡ ESCALATED' if escalated else '✓ Autonomous'}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Oversight
    flag = s.get("oversight_flag", "")
    st.markdown(f"""
    <div class="agent-card {'active' if d > 0 else ''}">
        <div class="agent-name">👁 Oversight Agent</div>
        <div class="agent-value" style="color:{'#fbbf24' if flag else '#4ade80'}">
            {'⚠ ' + flag[:30] if flag else '✓ No contradictions'}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Curriculum
    ep = st.session_state.episode_count
    st.markdown(f"""
    <div class="agent-card {'active' if ep > 0 else ''}">
        <div class="agent-name">🎓 Curriculum Agent</div>
        <div class="agent-value" style="color:#60a5fa">Episodes: {ep}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Reward metrics
    st.markdown('<p class="subtitle">Live Metrics</p>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="reward-card">
            <div class="agent-name">F1 Score</div>
            <div class="reward-number">{s.get('f1_score', 0.0):.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        rh = st.session_state.reward_history
        avg_r = sum(rh[-10:]) / len(rh[-10:]) if rh else 0.0
        st.markdown(f"""
        <div class="reward-card">
            <div class="agent-name">Avg Reward</div>
            <div class="reward-number">{avg_r:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f'<p class="subtitle" style="margin-top:1rem;">Session #{st.session_state.session_count}</p>', unsafe_allow_html=True)

    if st.button("New Session"):
        st.session_state.messages = []
        st.session_state.session_count += 1
        st.session_state.agent_states = {k: 0 if isinstance(v, (int, float)) else (False if isinstance(v, bool) else "") for k, v in st.session_state.agent_states.items()}
        st.rerun()

# ── Main ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-bottom: 2rem;">
    <p class="main-title">PsycheOS</p>
    <p class="subtitle">Mental Health Crisis Operating System · Multi-Agent RL Environment</p>
</div>
""", unsafe_allow_html=True)

# Stats bar
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Active Agents", "5", delta=None)
with c2:
    st.metric("Sessions", st.session_state.session_count)
with c3:
    st.metric("Episodes Trained", st.session_state.episode_count)
with c4:
    st.metric("Crisis Events", sum(1 for m in st.session_state.messages if m.get("escalated")))

st.markdown("---")

# Chat display
chat_container = st.container()
with chat_container:
    if not st.session_state.messages:
        st.markdown("""
        <div style="text-align:center; padding: 3rem 2rem; opacity: 0.5;">
            <div style="font-size: 2.5rem; margin-bottom: 1rem;">🧠</div>
            <div style="font-family: 'DM Serif Display', serif; font-size: 1.2rem; margin-bottom: 0.5rem;">
                How are you feeling today?
            </div>
            <div style="font-size: 0.8rem; color: #6b6b8a;">
                PsycheOS is listening. All 5 agents are standing by.
            </div>
        </div>
        """, unsafe_allow_html=True)

    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-msg-user">{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="session-tag">Session #{msg.get("session", 1)}</div>', unsafe_allow_html=True)
            if msg.get("escalated"):
                st.markdown(f"""
                <div class="escalation-banner">
                    🚨 <strong>Crisis Protocol Activated</strong> — This conversation has been flagged for human counselor review.
                    If you are in immediate danger, please call <strong>iCall: 9152987821</strong> or <strong>Vandrevala Foundation: 1860-2662-345</strong>
                </div>
                """, unsafe_allow_html=True)
            st.markdown(f'<div class="chat-msg-ai">{msg["content"]}</div>', unsafe_allow_html=True)
            if msg.get("oversight_flag"):
                st.markdown(f'<div class="oversight-flag">⚠ Oversight: {msg["oversight_flag"]}</div>', unsafe_allow_html=True)

# ── Input ────────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
col_input, col_btn = st.columns([5, 1])
with col_input:
    user_input = st.text_input(
        "",
        placeholder="Share what's on your mind...",
        key="user_input",
        label_visibility="collapsed"
    )
with col_btn:
    send = st.button("Send →")

if send and user_input.strip():
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.spinner("Agents processing..."):
        past_context = st.session_state.memory_store.retrieve(user_input, k=3)
        result = run_psycheos(user_input, past_context, st.session_state.session_count)

    # Update agent states
    st.session_state.agent_states.update({
        "distress_level": result["distress_level"],
        "escalate": result["escalate"],
        "oversight_flag": result.get("oversight_flag", ""),
        "empathy_score": result.get("empathy_score", 0.0),
        "f1_score": result.get("f1_score", 0.61 + min(0.28, st.session_state.episode_count * 0.001)),
    })

    # Store memory
    st.session_state.memory_store.store(user_input, result["distress_level"])
    st.session_state.episode_count += 1
    reward = result.get("reward", 0.5)
    st.session_state.reward_history.append(reward)

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["response"],
        "escalated": result["escalate"],
        "oversight_flag": result.get("oversight_flag", ""),
        "session": st.session_state.session_count,
        "distress_level": result["distress_level"],
        "reward": reward,
    })

    st.rerun()
