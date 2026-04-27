"""
app.py - Streamlit UI for VibeFinder 2.0
A full applied AI music recommendation system with RAG + agentic workflow.
"""

import os
import sys
import logging
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VibeFinder 2.0",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title { font-size: 2.8rem; font-weight: 800; color: #1DB954; text-align: center; margin-bottom: 0; }
    .subtitle   { font-size: 1.1rem; color: #aaaaaa; text-align: center; margin-bottom: 2rem; }
    .song-card  { background: #1e1e2e; border-radius: 12px; padding: 1rem 1.4rem;
                  margin-bottom: 0.8rem; border-left: 4px solid #1DB954; }
    .song-title { font-size: 1.1rem; font-weight: 700; color: #ffffff; }
    .song-meta  { font-size: 0.85rem; color: #aaaaaa; margin-top: 0.2rem; }
    .score-badge { background: #1DB954; color: #000; font-weight: 700;
                   padding: 2px 10px; border-radius: 20px; font-size: 0.85rem; }
    .confidence-bar { height: 10px; border-radius: 5px; background: #333; margin-top: 4px; }
    .step-box { background: #12121f; border-radius: 8px; padding: 0.8rem 1rem;
                margin-bottom: 0.5rem; font-size: 0.88rem; color: #cccccc; }
    .step-label { font-weight: 700; color: #1DB954; }
    stButton > button { background-color: #1DB954 !important; color: black !important;
                        font-weight: 700 !important; border-radius: 25px !important; }
</style>
""", unsafe_allow_html=True)

# ── Load resources (cached) ───────────────────────────────────────────────────
@st.cache_resource(show_spinner="🎵 Loading song catalog and building vector store...")
def load_resources():
    from src.recommender import load_songs
    from src.rag import build_vectorstore
    # Always resolve data path relative to project root
    src_dir = os.path.dirname(os.path.abspath(__file__))
    base = os.path.dirname(src_dir)  # go up from src/ to project root
    songs = load_songs(os.path.join(base, "data", "songs.csv"))
    vs = build_vectorstore(songs)
    return songs, vs


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎛️ VibeFinder 2.0")
    st.markdown("---")
    st.markdown("### How it works")
    st.markdown("""
1. 🧠 **PLAN** — AI extracts your preferences
2. 🔍 **RETRIEVE** — RAG searches the catalog
3. 📊 **SCORE** — Content-based ranking
4. ✅ **VALIDATE** — Confidence + guardrails
5. 🎤 **EXPLAIN** — Music critic descriptions
    """)
    st.markdown("---")
    st.markdown("### About")
    st.markdown("VibeFinder 2.0 is an applied AI music recommendation system built with **RAG**, **agentic workflows**, and **few-shot specialization**.")
    st.markdown("---")
    k = st.slider("Number of recommendations", min_value=3, max_value=10, value=5)
    show_steps = st.toggle("Show agent steps", value=True)


# ── Main UI ───────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🎧 VibeFinder 2.0</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered music recommendations · RAG + Agentic Workflow</div>', unsafe_allow_html=True)

# Quick profile buttons
if "active_query" not in st.session_state:
    st.session_state.active_query = ""

st.markdown("#### 🚀 Try a quick profile or type your own:")
col1, col2, col3, col4 = st.columns(4)

btn1 = col1.button("😌 Chill Study")
btn2 = col2.button("💪 Workout")
btn3 = col3.button("🎉 Party")
btn4 = col4.button("😢 Sad Mood")

if btn1:
    st.session_state.active_query = "I want chill lofi music to study late at night"
if btn2:
    st.session_state.active_query = "High energy intense rock music for working out"
if btn3:
    st.session_state.active_query = "Happy upbeat pop music for a party"
if btn4:
    st.session_state.active_query = "Melancholy indie music for a rainy day"

user_query = st.text_input(
    "Or describe what you want:",
    value=st.session_state.active_query,
    placeholder="e.g. I want calm jazz music to relax after work...",
)
# Update session state if user typed something
if user_query != st.session_state.active_query:
    st.session_state.active_query = user_query

run_btn = st.button("🎵 Find My Vibes", use_container_width=True)
# Auto-run when a quick profile button is clicked
auto_run = btn1 or btn2 or btn3 or btn4

# ── Run agent ─────────────────────────────────────────────────────────────────
if (run_btn or auto_run) and st.session_state.active_query.strip():
    user_query = st.session_state.active_query
    try:
        songs, vs = load_resources()
    except Exception as e:
        st.error(f"❌ Failed to load resources: {e}")
        st.stop()

    from src.agent import run_agent

    with st.spinner("🤖 Agent is thinking..."):
        try:
            result = run_agent(user_query, songs, vs, k=k)
        except Exception as e:
            st.error(f"❌ Agent error: {e}")
            st.stop()

    # ── Agent Steps ───────────────────────────────────────────────────────────
    if show_steps:
        st.markdown("---")
        st.markdown("### 🤖 Agent Reasoning Steps")

        plan = result.get("plan", {})
        with st.expander("🧠 Step 1 — PLAN", expanded=False):
            st.markdown(f"""
<div class="step-box">
<span class="step-label">Genre:</span> {plan.get('favorite_genre', 'N/A')}<br>
<span class="step-label">Mood:</span> {plan.get('favorite_mood', 'N/A')}<br>
<span class="step-label">Energy:</span> {plan.get('target_energy', 'N/A')}<br>
<span class="step-label">Tempo:</span> {plan.get('target_tempo', 'N/A')} BPM<br>
<span class="step-label">Summary:</span> {plan.get('summary', 'N/A')}
</div>
""", unsafe_allow_html=True)

        retrieved = result.get("retrieved", [])
        with st.expander(f"🔍 Step 2 — RETRIEVE ({len(retrieved)} songs found)", expanded=False):
            for song in retrieved:
                st.markdown(f"""
<div class="step-box">
<span class="step-label">{song['title']}</span> by {song['artist']}
&nbsp;·&nbsp; {song['genre']} / {song['mood']}
&nbsp;·&nbsp; similarity: <b>{song['similarity_score']:.3f}</b>
</div>
""", unsafe_allow_html=True)

        scored = result.get("scored", [])
        with st.expander("📊 Step 3 — SCORE", expanded=False):
            for song, score, reasons in scored:
                st.markdown(f"""
<div class="step-box">
<span class="step-label">{song['title']}</span> — Score: {score:.2f}<br>
<span style="color:#888">{' | '.join(reasons)}</span>
</div>
""", unsafe_allow_html=True)

        evaluation = result.get("evaluation", {})
        conf = evaluation.get("confidence_report", {})
        with st.expander("✅ Step 4 — VALIDATE", expanded=False):
            confidence = conf.get("confidence", 0)
            level = conf.get("level", "unknown").upper()
            color = {"HIGH": "#1DB954", "MEDIUM": "#f0a500", "LOW": "#e05c5c", "VERY_LOW": "#cc0000"}.get(level, "#888")
            st.markdown(f"""
<div class="step-box">
<span class="step-label">Confidence:</span> <span style="color:{color}">{confidence:.3f} ({level})</span><br>
<span class="step-label">Avg Score:</span> {conf.get('avg_score', 0):.3f} / 5.0<br>
<span class="step-label">Strong Matches:</span> {conf.get('details', 'N/A')}<br>
<span class="step-label">Guardrail:</span> {'✅ PASSED' if evaluation.get('guardrail_passed') else '❌ FAILED'}
</div>
""", unsafe_allow_html=True)
            if evaluation.get("warnings"):
                for w in evaluation["warnings"]:
                    st.warning(f"⚠️ {w}")
            if evaluation.get("issues"):
                for i in evaluation["issues"]:
                    st.error(f"🚫 {i}")

    # ── Recommendations ───────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🎵 Your Recommendations")

    scored = result.get("scored", [])
    explanations = result.get("explanation", "").split("\n\n")

    for i, (song, score, reasons) in enumerate(scored):
        exp_text = explanations[i] if i < len(explanations) else ""
        # Clean numbering from explanation
        if exp_text and exp_text[0].isdigit():
            exp_text = exp_text[3:].strip() if len(exp_text) > 3 else exp_text

        st.markdown(f"""
<div class="song-card">
  <div style="display:flex; justify-content:space-between; align-items:center;">
    <div class="song-title">#{i+1} &nbsp; {song['title']}</div>
    <span class="score-badge">Score: {score:.2f}</span>
  </div>
  <div class="song-meta">
    🎤 {song['artist']} &nbsp;·&nbsp;
    🎸 {song['genre'].title()} &nbsp;·&nbsp;
    😌 {song['mood'].title()} &nbsp;·&nbsp;
    ⚡ Energy: {song['energy']} &nbsp;·&nbsp;
    🥁 {song['tempo_bpm']} BPM
  </div>
  <div style="margin-top:0.6rem; font-size:0.9rem; color:#dddddd; line-height:1.5;">
    {exp_text}
  </div>
</div>
""", unsafe_allow_html=True)

    # ── Confidence meter ──────────────────────────────────────────────────────
    st.markdown("---")
    evaluation = result.get("evaluation", {})
    conf = evaluation.get("confidence_report", {})
    confidence = conf.get("confidence", 0)
    level = conf.get("level", "unknown").upper()
    color = {"HIGH": "#1DB954", "MEDIUM": "#f0a500", "LOW": "#e05c5c", "VERY_LOW": "#cc0000"}.get(level, "#888")
    st.markdown(f"### 📊 Confidence: <span style='color:{color}'>{confidence:.1%} ({level})</span>", unsafe_allow_html=True)
    st.progress(min(confidence, 1.0))

elif (run_btn or auto_run) and not st.session_state.active_query.strip():
    st.warning("⚠️ Please enter a query or click a quick profile!")