"""
agent.py - Agentic workflow for VibeFinder 2.0.
Implements a multi-step reasoning pipeline:
  Step 1: PLAN   - understand the user's request
  Step 2: RETRIEVE - RAG lookup of semantically similar songs
  Step 3: SCORE  - run content-based scoring on retrieved songs
  Step 4: VALIDATE - check confidence and guardrails
  Step 5: EXPLAIN  - generate a music-critic style explanation

Each step is observable (printed to console and logged).
"""

import os
import logging
from typing import List, Dict, Any
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

from src.rag import build_vectorstore, retrieve_similar_songs, format_retrieved_context
from src.recommender import load_songs, recommend_songs

load_dotenv()
logger = logging.getLogger(__name__)

# ── Few-shot music critic persona (specialization stretch feature) ──────────
MUSIC_CRITIC_SYSTEM_PROMPT = """You are VibeBot, an expert music curator and critic with deep knowledge of genres, moods, and sonic textures.

Your personality:
- Speak like a knowledgeable but approachable music journalist
- Use specific musical vocabulary (timbre, groove, dynamics, texture)
- Always explain WHY a song fits, not just THAT it fits
- Be enthusiastic but precise — no generic praise like "great song"
- Keep each recommendation to 2-3 sentences maximum

Few-shot examples of your style:

User: Why does "Library Rain" fit a chill lofi study session?
VibeBot: "Library Rain" earns its spot through its unhurried 72 BPM pulse and high acousticness (0.86), which creates that warm, analog texture lofi listeners crave. The low energy (0.35) means it stays out of your way cognitively — perfect background architecture for deep focus.

User: Why does "Storm Runner" fit an intense rock workout?
VibeBot: "Storm Runner" is built for physical exertion — its 152 BPM tempo drives relentless forward momentum while the 0.91 energy score means it never lets up. The intense mood tag is earned, not labeled: this track has the sonic weight to push through a final rep.

Now respond in this same style for the songs you are asked to explain."""


def run_agent(
    user_query: str,
    songs: List[Dict],
    vectorstore: Any,
    k: int = 5
) -> Dict:
    """
    Runs the full 5-step agentic pipeline and returns a structured result.

    Args:
        user_query: Natural language request e.g. "I want chill music to study"
        songs: Full song catalog
        vectorstore: Pre-built ChromaDB vectorstore
        k: Number of recommendations to return

    Returns:
        Dict with keys: plan, retrieved, scored, confidence, explanation, recommendations
    """

    result = {}
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    # ── STEP 1: PLAN ────────────────────────────────────────────────────────
    print("\n" + "="*50)
    print("🧠 STEP 1: PLAN — Understanding your request...")
    print("="*50)

    plan_prompt = f"""A user said: "{user_query}"

Extract their music preferences as JSON with these exact keys:
{{
  "favorite_genre": "genre name or unknown",
  "favorite_mood": "mood word or unknown",
  "target_energy": 0.0 to 1.0 as float,
  "target_tempo": BPM as integer,
  "summary": "one sentence summary of what they want"
}}

Return ONLY the JSON, no other text."""

    plan_response = llm.invoke([HumanMessage(content=plan_prompt)])
    import json, re
    raw = plan_response.content.strip()
    # strip markdown code fences if present
    raw = re.sub(r"```json|```", "", raw).strip()

    try:
        plan = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Could not parse plan JSON, using defaults")
        plan = {
            "favorite_genre": "pop",
            "favorite_mood": "happy",
            "target_energy": 0.7,
            "target_tempo": 120,
            "summary": user_query
        }

    result["plan"] = plan
    print(f"  🎯 Genre: {plan.get('favorite_genre')}")
    print(f"  🎯 Mood:  {plan.get('favorite_mood')}")
    print(f"  🎯 Energy: {plan.get('target_energy')}")
    print(f"  🎯 Tempo: {plan.get('target_tempo')} BPM")
    print(f"  📝 Summary: {plan.get('summary')}")
    logger.info(f"Plan extracted: {plan}")

    # ── STEP 2: RETRIEVE ────────────────────────────────────────────────────
    print("\n" + "="*50)
    print("🔍 STEP 2: RETRIEVE — Searching song catalog via RAG...")
    print("="*50)

    rag_query = (
        f"{plan.get('favorite_genre')} {plan.get('favorite_mood')} music "
        f"energy {plan.get('target_energy')} tempo {plan.get('target_tempo')} bpm"
    )

    retrieved = retrieve_similar_songs(rag_query, vectorstore, k=k * 2)
    result["retrieved"] = retrieved

    for song in retrieved:
        print(f"  📀 {song['title']} by {song['artist']} — similarity: {song['similarity_score']:.3f}")

    # ── STEP 3: SCORE ────────────────────────────────────────────────────────
    print("\n" + "="*50)
    print("📊 STEP 3: SCORE — Running content-based scoring...")
    print("="*50)

    user_prefs = {
        "favorite_genre": plan.get("favorite_genre", "pop"),
        "favorite_mood": plan.get("favorite_mood", "happy"),
        "target_energy": float(plan.get("target_energy", 0.7)),
        "target_tempo": float(plan.get("target_tempo", 120)),
    }

    # Score only the RAG-retrieved songs (not full catalog)
    retrieved_song_titles = {s["title"] for s in retrieved}
    retrieved_full = [s for s in songs if s["title"] in retrieved_song_titles]

    # Fall back to full catalog if retrieval returned too few
    pool = retrieved_full if len(retrieved_full) >= k else songs

    scored = recommend_songs(user_prefs, pool, k=k)
    result["scored"] = scored

    for song, score, reasons in scored:
        print(f"  🎵 {song['title']} — Score: {score:.2f} | {', '.join(reasons[:2])}")

    # ── STEP 4: VALIDATE ─────────────────────────────────────────────────────
    print("\n" + "="*50)
    print("✅ STEP 4: VALIDATE — Checking confidence and guardrails...")
    print("="*50)

    scores = [score for _, score, _ in scored]
    max_possible = 5.0
    avg_score = sum(scores) / len(scores) if scores else 0
    confidence = round(avg_score / max_possible, 3)

    top_score = scores[0] if scores else 0
    score_spread = round(top_score - scores[-1], 3) if len(scores) > 1 else 0

    guardrail_passed = confidence >= 0.4
    result["confidence"] = confidence
    result["guardrail_passed"] = guardrail_passed

    print(f"  📈 Average score: {avg_score:.2f} / {max_possible}")
    print(f"  📊 Confidence: {confidence:.3f}")
    print(f"  📉 Score spread: {score_spread:.3f}")
    print(f"  🛡️  Guardrail: {'✅ PASSED' if guardrail_passed else '⚠️  LOW CONFIDENCE — results may be generic'}")

    if not guardrail_passed:
        logger.warning(f"Low confidence ({confidence}) for query: {user_query}")

    # ── STEP 5: EXPLAIN ──────────────────────────────────────────────────────
    print("\n" + "="*50)
    print("🎤 STEP 5: EXPLAIN — Generating music critic explanations...")
    print("="*50)

    context = format_retrieved_context(retrieved)
    top_songs_text = "\n".join(
        [f"- \"{s['title']}\" by {s['artist']} ({s['genre']}, {s['mood']}, energy={s['energy']})"
         for s, _, _ in scored]
    )

    explain_prompt = f"""The user asked for: "{user_query}"

Retrieved context from catalog:
{context}

Top recommended songs after scoring:
{top_songs_text}

Write a short music critic explanation for each recommended song explaining why it fits the user's request. Use your VibeBot persona."""

    explanation_response = llm.invoke([
        SystemMessage(content=MUSIC_CRITIC_SYSTEM_PROMPT),
        HumanMessage(content=explain_prompt)
    ])

    explanation = explanation_response.content.strip()
    result["explanation"] = explanation
    result["recommendations"] = scored

    print("\n" + explanation)

    print("\n" + "="*50)
    print("🏁 AGENT COMPLETE")
    print("="*50)

    return result
