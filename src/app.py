"""
app.py - Streamlit UI for VibeFinder 2.0.
A full applied AI music recommendation system with RAG + agentic workflow.
"""

import html
import os
import sys
from copy import deepcopy
from typing import Any, Dict, List, Optional

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Ensure project root is on the import path so `from src...` works in Streamlit.
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.agent import (
    apply_refinement_action,
    profile_from_song,
    refine_recommendations,
    run_agent,
)
from src.config import openai_api_key_help_text, resolve_openai_api_key

REFINEMENT_ACTIONS = [
    ("more_upbeat", "More upbeat"),
    ("more_chill", "More chill"),
    ("more_energetic", "More energetic"),
    ("more_acoustic", "More acoustic"),
    ("faster", "Faster"),
    ("slower", "Slower"),
    ("reset", "Reset"),
]

SESSION_DEFAULTS = {
    "active_query": "",
    "last_run_query": "",
    "base_profile": None,
    "base_result": None,
    "working_profile": None,
    "current_result": None,
    "refinement_history": [],
    "excluded_titles": set(),
    "anchor_title": None,
}

st.set_page_config(
    page_title="Music Finder",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    :root {
        --leaf-1: #1db954;
        --leaf-2: #78d96b;
        --ink-1: #203047;
        --ink-2: #5c6978;
        --mist: #f6faf8;
        --card: rgba(255, 255, 255, 0.88);
        --line: rgba(47, 76, 66, 0.10);
        --glow: 0 18px 45px rgba(29, 73, 46, 0.10);
    }
    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(circle at 10% 10%, rgba(140, 214, 145, 0.16), transparent 26%),
            radial-gradient(circle at 88% 14%, rgba(255, 197, 124, 0.18), transparent 24%),
            linear-gradient(180deg, #f7fbff 0%, #fbfcf8 52%, #f6faf8 100%);
        color: var(--ink-1);
    }
    [data-testid="stHeader"] {
        background: rgba(255, 255, 255, 0.0);
    }
    [data-testid="stSidebar"] {
        background:
            linear-gradient(180deg, rgba(248, 251, 255, 0.96) 0%, rgba(238, 245, 240, 0.96) 100%);
        border-right: 1px solid rgba(70, 92, 78, 0.08);
    }
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p {
        color: var(--ink-1);
    }
    .block-container {
        max-width: 1180px;
        padding-top: 1.4rem;
        padding-bottom: 3rem;
    }
    .hero-panel {
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(34, 76, 54, 0.10);
        border-radius: 30px;
        padding: 2.3rem 2.35rem 2rem 2.35rem;
        margin-bottom: 1.6rem;
        background:
            linear-gradient(135deg, rgba(245, 255, 247, 0.95) 0%, rgba(247, 251, 255, 0.98) 50%, rgba(255, 248, 239, 0.94) 100%);
        box-shadow: var(--glow);
    }
    .hero-panel::before {
        content: "";
        position: absolute;
        inset: auto -5rem -5rem auto;
        width: 14rem;
        height: 14rem;
        border-radius: 999px;
        background: radial-gradient(circle, rgba(29, 185, 84, 0.18) 0%, rgba(29, 185, 84, 0) 70%);
    }
    .hero-panel::after {
        content: "";
        position: absolute;
        inset: -5rem auto auto -5rem;
        width: 12rem;
        height: 12rem;
        border-radius: 999px;
        background: radial-gradient(circle, rgba(255, 187, 112, 0.20) 0%, rgba(255, 187, 112, 0) 72%);
    }
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.4rem 0.85rem;
        border-radius: 999px;
        border: 1px solid rgba(29, 185, 84, 0.18);
        background: rgba(255, 255, 255, 0.72);
        color: #1d7f48;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }
    .main-title {
        margin: 0.75rem 0 0.35rem 0;
        font-size: clamp(3rem, 6vw, 4.9rem);
        line-height: 0.95;
        font-weight: 900;
        letter-spacing: -0.05em;
        color: #163524;
        font-family: "Avenir Next", "Trebuchet MS", sans-serif;
    }
    .subtitle {
        max-width: 48rem;
        margin: 0 0 1.4rem 0;
        font-size: 1.13rem;
        line-height: 1.6;
        color: var(--ink-2);
    }
    .hero-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
        gap: 0.9rem;
        margin-top: 1.1rem;
    }
    .hero-stat {
        border-radius: 20px;
        border: 1px solid rgba(32, 48, 71, 0.08);
        background: rgba(255, 255, 255, 0.72);
        padding: 0.95rem 1rem;
        backdrop-filter: blur(6px);
    }
    .hero-stat strong {
        display: block;
        font-size: 1.5rem;
        line-height: 1;
        color: #173928;
        margin-bottom: 0.3rem;
    }
    .hero-stat span {
        color: var(--ink-2);
        font-size: 0.92rem;
    }
    .surface-title {
        margin: 1rem 0 0.3rem 0;
        font-size: 2.1rem;
        line-height: 1.1;
        font-weight: 800;
        color: var(--ink-1);
        letter-spacing: -0.04em;
    }
    .surface-copy {
        margin: 0 0 1rem 0;
        color: var(--ink-2);
        font-size: 1rem;
    }
    .tune-copy {
        margin: 0 0 0.75rem 0;
        color: var(--ink-2);
        font-size: 0.98rem;
    }
    div[data-testid="stTextInput"] label p {
        color: var(--ink-1);
        font-weight: 700;
        font-size: 1rem;
    }
    div[data-baseweb="input"] {
        background: rgba(255, 255, 255, 0.84);
        border: 1px solid rgba(47, 76, 66, 0.10);
        border-radius: 18px;
        box-shadow: 0 6px 20px rgba(27, 54, 42, 0.05);
    }
    div[data-baseweb="input"]:focus-within {
        border-color: rgba(29, 185, 84, 0.45);
        box-shadow: 0 0 0 4px rgba(29, 185, 84, 0.12);
    }
    div[data-baseweb="input"] input {
        color: var(--ink-1);
        font-size: 1.03rem;
    }
    div.stButton > button {
        min-height: 3rem;
        border-radius: 16px;
        border: 1px solid rgba(61, 82, 71, 0.12);
        background: rgba(255, 255, 255, 0.88);
        color: var(--ink-1);
        font-weight: 700;
        box-shadow: 0 10px 24px rgba(25, 49, 39, 0.05);
        transition: transform 120ms ease, box-shadow 120ms ease, border-color 120ms ease;
    }
    div.stButton > button:hover {
        transform: translateY(-1px);
        border-color: rgba(29, 185, 84, 0.45);
        box-shadow: 0 16px 28px rgba(27, 54, 42, 0.09);
        color: #155b36;
    }
    div.stButton > button[kind="primary"] {
        border: none;
        background: linear-gradient(135deg, var(--leaf-1) 0%, var(--leaf-2) 100%);
        color: #0b2815;
        box-shadow: 0 18px 30px rgba(29, 185, 84, 0.24);
    }
    div.stButton > button[kind="primary"]:hover {
        box-shadow: 0 22px 34px rgba(29, 185, 84, 0.28);
    }
    [data-testid="stExpander"] details {
        border: 1px solid rgba(47, 76, 66, 0.10);
        border-radius: 18px;
        background: rgba(255, 255, 255, 0.78);
        overflow: hidden;
    }
    .song-shell {
        border: 1px solid rgba(34, 76, 54, 0.12);
        border-radius: 24px;
        padding: 1.15rem 1.2rem;
        margin-bottom: 1rem;
        background:
            linear-gradient(180deg, rgba(255, 255, 255, 0.92) 0%, rgba(247, 251, 248, 0.92) 100%);
        box-shadow: 0 16px 34px rgba(18, 48, 30, 0.07);
    }
    .song-head {
        display: flex;
        justify-content: space-between;
        gap: 1rem;
        align-items: flex-start;
    }
    .song-rank {
        display: inline-flex;
        padding: 0.28rem 0.72rem;
        border-radius: 999px;
        background: rgba(29, 185, 84, 0.10);
        color: #13703f;
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-bottom: 0.55rem;
    }
    .song-title {
        margin: 0;
        font-size: 1.45rem;
        line-height: 1.1;
        font-weight: 800;
        color: var(--ink-1);
        letter-spacing: -0.03em;
    }
    .song-artist {
        margin: 0.18rem 0 0 0;
        color: var(--ink-2);
        font-size: 0.98rem;
    }
    .score-pill {
        flex-shrink: 0;
        padding: 0.55rem 0.85rem;
        border-radius: 18px;
        background: linear-gradient(135deg, #163524 0%, #1f6c43 100%);
        color: #f4fff5;
        text-align: right;
        min-width: 92px;
    }
    .score-pill strong {
        display: block;
        font-size: 1.25rem;
        line-height: 1;
    }
    .score-pill span {
        font-size: 0.75rem;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        opacity: 0.88;
    }
    .tag-row,
    .metric-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.55rem;
        margin-top: 0.9rem;
    }
    .tag-chip,
    .metric-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.42rem;
        padding: 0.48rem 0.72rem;
        border-radius: 999px;
        border: 1px solid rgba(34, 76, 54, 0.10);
        background: rgba(255, 255, 255, 0.84);
        color: var(--ink-1);
        font-size: 0.88rem;
    }
    .tag-chip {
        font-weight: 700;
        color: #165838;
    }
    .metric-chip strong {
        color: #11321f;
    }
    .song-explanation {
        margin: 1rem 0 0.1rem 0;
        color: #334255;
        font-size: 0.98rem;
        line-height: 1.7;
    }
    .confidence-shell {
        display: flex;
        justify-content: space-between;
        gap: 1rem;
        align-items: flex-end;
        border: 1px solid rgba(34, 76, 54, 0.10);
        border-radius: 22px;
        padding: 1.15rem 1.2rem;
        background: rgba(255, 255, 255, 0.84);
        box-shadow: 0 14px 30px rgba(18, 48, 30, 0.06);
    }
    .confidence-shell h3 {
        margin: 0.25rem 0 0 0;
        font-size: 1.8rem;
        color: var(--ink-1);
        letter-spacing: -0.04em;
    }
    .confidence-label {
        font-size: 0.8rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #627081;
        font-weight: 800;
    }
    .confidence-copy {
        max-width: 26rem;
        color: var(--ink-2);
        font-size: 0.95rem;
        text-align: right;
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #1db954 0%, #8add72 100%);
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data
def get_catalog_summary() -> Dict[str, int]:
    from src.recommender import load_songs

    src_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(src_dir)
    songs = load_songs(os.path.join(base_dir, "data", "songs.csv"))
    return {
        "count": len(songs),
        "genres": len({song["genre"] for song in songs}),
    }


@st.cache_resource(show_spinner="Loading song catalog and building vector store...")
def load_resources():
    from src.rag import build_vectorstore
    from src.recommender import load_songs

    src_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(src_dir)
    songs = load_songs(os.path.join(base_dir, "data", "songs.csv"))
    vectorstore = build_vectorstore(songs)
    return songs, vectorstore


def render_api_key_notice() -> None:
    """Shows setup guidance when the OpenAI key is missing."""

    st.error(openai_api_key_help_text())
    st.info("For Streamlit Cloud, add this in App settings -> Secrets.")
    st.code('OPENAI_API_KEY = "sk-..."', language="toml")


def render_hero(api_key_ready: bool) -> None:
    catalog = get_catalog_summary()
    status_text = (
        "Live recommendation pipeline ready"
        if api_key_ready
        else "Waiting on OpenAI API configuration"
    )
    st.markdown(
        f"""
<section class="hero-panel">
  <div class="hero-badge">Refinement-Ready Music Discovery</div>
  <h1 class="main-title">Music Finder</h1>
  <p class="subtitle">
    Turn a feeling into a recommendation lane, then keep shaping the results with
    fast refinements, song-based branching, and critic-style explanations.
  </p>
  <div class="hero-grid">
    <div class="hero-stat">
      <strong>{catalog["count"]}</strong>
      <span>curated tracks across {catalog["genres"]} genres</span>
    </div>
    <div class="hero-stat">
      <strong>{len(REFINEMENT_ACTIONS)}</strong>
      <span>one-click tuning actions including reset</span>
    </div>
    <div class="hero-stat">
      <strong>RAG + LLM</strong>
      <span>semantic retrieval, ranking, and vivid explanations</span>
    </div>
    <div class="hero-stat">
      <strong>{html.escape(status_text)}</strong>
      <span>designed for exploration instead of one-shot results</span>
    </div>
  </div>
</section>
""",
        unsafe_allow_html=True,
    )


def init_session_state() -> None:
    for key, default_value in SESSION_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = deepcopy(default_value)


def _copy_result(result: Dict[str, Any]) -> Dict[str, Any]:
    return deepcopy(result)


def _format_optional_value(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    return f"{value:.2f}"


def _store_initial_result(user_query: str, result: Dict[str, Any]) -> None:
    st.session_state.last_run_query = user_query
    st.session_state.base_profile = deepcopy(result["active_profile"])
    st.session_state.base_result = _copy_result(result)
    st.session_state.working_profile = deepcopy(result["active_profile"])
    st.session_state.current_result = _copy_result(result)
    st.session_state.refinement_history = []
    st.session_state.excluded_titles = set()
    st.session_state.anchor_title = None


def _store_refined_result(result: Dict[str, Any]) -> None:
    st.session_state.working_profile = deepcopy(result["active_profile"])
    st.session_state.current_result = _copy_result(result)
    st.session_state.refinement_history = list(result.get("refinement_history", []))


def _build_summary_for_current_branch(history: List[str]) -> str:
    if st.session_state.anchor_title:
        if history:
            return (
                f'Starting from songs like "{st.session_state.anchor_title}", '
                f"applied {', '.join(history)}."
            )
        return f'Exploring tracks similar to "{st.session_state.anchor_title}".'

    if history:
        return f"Refined from your request with {', '.join(history)}."
    return "Refined from your original request."


def _run_refinement(action_id: str, k: int, songs: List[Dict[str, Any]], vectorstore: Any) -> None:
    if action_id == "reset":
        st.session_state.working_profile = deepcopy(st.session_state.base_profile)
        st.session_state.current_result = _copy_result(st.session_state.base_result)
        st.session_state.refinement_history = []
        st.session_state.excluded_titles = set()
        st.session_state.anchor_title = None
        st.rerun()

    updated_profile, action_label = apply_refinement_action(
        st.session_state.working_profile,
        action_id,
    )
    history = list(st.session_state.refinement_history) + [action_label]

    with st.spinner("Refining recommendations..."):
        refined_result = refine_recommendations(
            user_query=st.session_state.last_run_query,
            active_profile=updated_profile,
            songs=songs,
            vectorstore=vectorstore,
            k=k,
            refinement_history=history,
            exclude_titles=set(st.session_state.excluded_titles),
            plan_summary=_build_summary_for_current_branch(history),
        )

    _store_refined_result(refined_result)
    st.rerun()


def _run_more_like(song: Dict[str, Any], k: int, songs: List[Dict[str, Any]], vectorstore: Any) -> None:
    anchor_profile = profile_from_song(song)
    excluded_titles = {song["title"]}

    with st.spinner(f'Finding songs like "{song["title"]}"...'):
        refined_result = refine_recommendations(
            user_query=st.session_state.last_run_query,
            active_profile=anchor_profile,
            songs=songs,
            vectorstore=vectorstore,
            k=k,
            refinement_history=[],
            exclude_titles=excluded_titles,
            plan_summary=f'Exploring tracks similar to "{song["title"]}".',
        )

    st.session_state.anchor_title = song["title"]
    st.session_state.excluded_titles = excluded_titles
    _store_refined_result(refined_result)
    st.rerun()


def render_agent_steps(result: Dict[str, Any]) -> None:
    st.markdown("---")
    st.markdown("### Agent Reasoning Steps")

    plan = result.get("plan", {})
    with st.expander("Step 1 - PLAN", expanded=False):
        st.markdown(f"**Genre:** {html.escape(str(plan.get('favorite_genre', 'N/A')))}")
        st.markdown(f"**Mood:** {html.escape(str(plan.get('favorite_mood', 'N/A')))}")
        st.markdown(f"**Energy:** {plan.get('target_energy', 'N/A')}")
        st.markdown(f"**Tempo:** {plan.get('target_tempo', 'N/A')} BPM")
        st.markdown(f"**Summary:** {html.escape(str(plan.get('summary', 'N/A')))}")

    retrieved = result.get("retrieved", [])
    with st.expander(f"Step 2 - RETRIEVE ({len(retrieved)} songs found)", expanded=False):
        if not retrieved:
            st.info("No retrieved songs were available for this profile.")
        for song in retrieved:
            st.markdown(
                f"- **{html.escape(song['title'])}** by {html.escape(song['artist'])} | "
                f"{html.escape(song['genre'])} / {html.escape(song['mood'])} | "
                f"similarity: `{song['similarity_score']:.3f}`"
            )

    scored = result.get("scored", [])
    with st.expander("Step 3 - SCORE", expanded=False):
        if not scored:
            st.info("No scored results were available.")
        for song, score, reasons in scored:
            safe_reasons = " | ".join(html.escape(reason) for reason in reasons)
            st.markdown(
                f"- **{html.escape(song['title'])}** score `{score:.2f}`  \n"
                f"  {safe_reasons}"
            )

    evaluation = result.get("evaluation", {})
    conf = evaluation.get("confidence_report", {})
    with st.expander("Step 4 - VALIDATE", expanded=False):
        st.markdown(
            f"**Confidence:** `{conf.get('confidence', 0):.3f}` "
            f"({str(conf.get('level', 'unknown')).upper()})"
        )
        st.markdown(f"**Average score:** `{conf.get('avg_score', 0):.3f}` / 5.0")
        st.markdown(f"**Strong matches:** {html.escape(str(conf.get('details', 'N/A')))}")
        st.markdown(
            f"**Guardrail:** {'PASSED' if evaluation.get('guardrail_passed') else 'FAILED'}"
        )
        for warning in evaluation.get("warnings", []):
            st.warning(warning)
        for issue in evaluation.get("issues", []):
            st.error(issue)


def render_refinement_bar(
    songs: List[Dict[str, Any]],
    vectorstore: Any,
    k: int,
) -> None:
    st.markdown("### Tune these results")
    st.markdown(
        '<p class="tune-copy">Nudge the current vibe brighter, slower, sharper, or more acoustic without starting over.</p>',
        unsafe_allow_html=True,
    )
    history = list(st.session_state.refinement_history)
    if history:
        st.caption("Current refinements: " + " -> ".join(history))
    elif st.session_state.anchor_title:
        st.caption(f'Current branch: songs similar to "{st.session_state.anchor_title}"')
    else:
        st.caption("Use quick refinements to keep tuning the current result set.")

    columns = st.columns(len(REFINEMENT_ACTIONS))
    for column, (action_id, label) in zip(columns, REFINEMENT_ACTIONS):
        if column.button(label, key=f"refine_{action_id}", use_container_width=True):
            _run_refinement(action_id, k, songs, vectorstore)


def render_recommendations(
    result: Dict[str, Any],
    songs: List[Dict[str, Any]],
    vectorstore: Any,
    k: int,
) -> None:
    st.markdown("---")
    st.markdown("### Your Recommendations")

    render_refinement_bar(songs, vectorstore, k)

    scored = result.get("scored", [])
    explanations = result.get("explanations", [])

    if not scored:
        st.warning("No recommendations are available for this profile.")
        return

    for index, (song, score, _) in enumerate(scored):
        explanation = explanations[index] if index < len(explanations) else ""
        st.markdown(
            (
                "<div class='song-shell'>"
                "<div class='song-head'>"
                "<div>"
                f"<span class='song-rank'>#{index + 1} Match</span>"
                f"<h4 class='song-title'>{html.escape(song['title'])}</h4>"
                f"<p class='song-artist'>{html.escape(song['artist'])}</p>"
                "</div>"
                "<div class='score-pill'>"
                f"<strong>{score:.2f}</strong>"
                "<span>Score</span>"
                "</div>"
                "</div>"
                "<div class='tag-row'>"
                f"<span class='tag-chip'>{html.escape(song['genre'].title())}</span>"
                f"<span class='tag-chip'>{html.escape(song['mood'].title())}</span>"
                "</div>"
                "<div class='metric-row'>"
                f"<span class='metric-chip'>Energy <strong>{song['energy']:.2f}</strong></span>"
                f"<span class='metric-chip'>Tempo <strong>{song['tempo_bpm']:.0f} BPM</strong></span>"
                f"<span class='metric-chip'>Acousticness <strong>{song['acousticness']:.2f}</strong></span>"
                f"<span class='metric-chip'>Valence <strong>{song['valence']:.2f}</strong></span>"
                "</div>"
                f"<p class='song-explanation'>{html.escape(explanation)}</p>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )
        if st.button("More like this", key=f"more_like_song_{song['id']}"):
            _run_more_like(song, k, songs, vectorstore)


def render_confidence(result: Dict[str, Any]) -> None:
    st.markdown("---")
    evaluation = result.get("evaluation", {})
    conf = evaluation.get("confidence_report", {})
    confidence = float(conf.get("confidence", 0.0))
    level = str(conf.get("level", "unknown")).upper()

    color = {
        "HIGH": "#1DB954",
        "MEDIUM": "#f0a500",
        "LOW": "#e05c5c",
        "VERY_LOW": "#cc0000",
    }.get(level, "#888888")

    summary = result.get("plan", {}).get("summary", "Confidence reflects how tightly the current list matches the active profile.")
    st.markdown(
        (
            "<div class='confidence-shell'>"
            "<div>"
            "<span class='confidence-label'>Recommendation confidence</span>"
            f"<h3><span style='color:{color}'>{confidence:.1%}</span> ({level})</h3>"
            "</div>"
            f"<div class='confidence-copy'>{html.escape(str(summary))}</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )
    st.progress(min(confidence, 1.0))

    profile = result.get("active_profile", {})
    st.caption(
        "Active profile: "
        f"genre={profile.get('favorite_genre', 'N/A')}, "
        f"mood={profile.get('favorite_mood', 'N/A')}, "
        f"energy={profile.get('target_energy', 0.0):.2f}, "
        f"tempo={profile.get('target_tempo', 0.0):.0f}, "
        f"acousticness={_format_optional_value(profile.get('target_acousticness'))}, "
        f"valence={_format_optional_value(profile.get('target_valence'))}"
    )


init_session_state()

with st.sidebar:
    st.markdown("## Music Finder")
    st.markdown("---")
    st.markdown("### How it works")
    st.markdown(
        """
1. PLAN - AI extracts your preferences
2. RETRIEVE - RAG searches the catalog
3. SCORE - Content-based ranking
4. VALIDATE - Confidence + guardrails
5. EXPLAIN - Music critic descriptions
"""
    )
    st.markdown("---")
    st.markdown("### About")
    st.markdown(
        "Music Finder is an applied AI music recommendation system built with "
        "**RAG**, **agentic workflows**, and **few-shot specialization**."
    )
    st.markdown("---")
    k = st.slider("Number of recommendations", min_value=3, max_value=10, value=5)
    show_steps = st.toggle("Show agent steps", value=True)

api_key_ready = bool(resolve_openai_api_key())
render_hero(api_key_ready)
if not api_key_ready:
    render_api_key_notice()

quick_profiles = {
    "Chill Study": "I want chill lofi music to study late at night",
    "Workout": "High energy intense rock music for working out",
    "Party": "Happy upbeat pop music for a party",
    "Sad Mood": "Melancholy indie music for a rainy day",
}

st.markdown('<h2 class="surface-title">Start With A Mood</h2>', unsafe_allow_html=True)
st.markdown(
    '<p class="surface-copy">Pick a launch point, describe a scene, or type the kind of energy you want the system to chase.</p>',
    unsafe_allow_html=True,
)
profile_columns = st.columns(len(quick_profiles))
button_results = {}
for column, (label, query_text) in zip(profile_columns, quick_profiles.items()):
    button_results[label] = column.button(label, use_container_width=True)
    if button_results[label]:
        st.session_state.active_query = query_text

user_query = st.text_input(
    "Describe what you want:",
    value=st.session_state.active_query,
    placeholder="e.g. I want calm jazz music to relax after work...",
)
if user_query != st.session_state.active_query:
    st.session_state.active_query = user_query

run_btn = st.button("Find My Vibes", use_container_width=True, type="primary")
auto_run = any(button_results.values())

if (run_btn or auto_run) and st.session_state.active_query.strip():
    if not api_key_ready:
        st.stop()
    try:
        songs, vectorstore = load_resources()
    except Exception as exc:
        st.error(f"Failed to load resources: {exc}")
        st.stop()

    with st.spinner("Agent is thinking..."):
        try:
            fresh_result = run_agent(st.session_state.active_query, songs, vectorstore, k=k)
        except Exception as exc:
            st.error(f"Agent error: {exc}")
            st.stop()
    _store_initial_result(st.session_state.active_query, fresh_result)
elif (run_btn or auto_run) and not st.session_state.active_query.strip():
    st.warning("Please enter a query or click a quick profile.")

current_result = st.session_state.current_result
if current_result:
    if not api_key_ready:
        st.stop()
    try:
        songs, vectorstore = load_resources()
    except Exception as exc:
        st.error(f"Failed to load resources: {exc}")
        st.stop()

    if show_steps:
        render_agent_steps(current_result)
    render_recommendations(current_result, songs, vectorstore, k)
    render_confidence(current_result)
