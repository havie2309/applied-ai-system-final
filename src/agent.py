"""
agent.py - Agentic workflow for VibeFinder 2.0.
Steps: PLAN -> RETRIEVE -> SCORE -> VALIDATE -> EXPLAIN
"""

import os
import json
import re
import logging
from typing import List, Dict, Any
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from src.rag import build_vectorstore, retrieve_similar_songs, format_retrieved_context
from src.recommender import load_songs, recommend_songs
from src.evaluator import run_evaluation

load_dotenv()
logger = logging.getLogger(__name__)

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


def run_agent(user_query: str, songs: List[Dict], vectorstore: Any, k: int = 5) -> Dict:
    result = {}
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, openai_api_key=os.getenv("OPENAI_API_KEY"))

    # STEP 1: PLAN
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
    raw = re.sub(r"```json|```", "", plan_response.content.strip()).strip()
    try:
        plan = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Could not parse plan JSON, using defaults")
        plan = {"favorite_genre": "pop", "favorite_mood": "happy", "target_energy": 0.7, "target_tempo": 120, "summary": user_query}

    result["plan"] = plan
    print(f"  🎯 Genre:   {plan.get('favorite_genre')}")
    print(f"  🎯 Mood:    {plan.get('favorite_mood')}")
    print(f"  🎯 Energy:  {plan.get('target_energy')}")
    print(f"  🎯 Tempo:   {plan.get('target_tempo')} BPM")
    print(f"  📝 Summary: {plan.get('summary')}")

    # STEP 2: RETRIEVE
    print("\n" + "="*50)
    print("🔍 STEP 2: RETRIEVE — Searching song catalog via RAG...")
    print("="*50)

    rag_query = f"{plan.get('favorite_genre')} {plan.get('favorite_mood')} music energy {plan.get('target_energy')} tempo {plan.get('target_tempo')} bpm"
    retrieved = retrieve_similar_songs(rag_query, vectorstore, k=k * 2)
    result["retrieved"] = retrieved
    for song in retrieved:
        print(f"  📀 {song['title']} by {song['artist']} — similarity: {song['similarity_score']:.3f}")

    # STEP 3: SCORE
    print("\n" + "="*50)
    print("📊 STEP 3: SCORE — Running content-based scoring...")
    print("="*50)

    user_prefs = {
        "favorite_genre": plan.get("favorite_genre", "pop"),
        "favorite_mood": plan.get("favorite_mood", "happy"),
        "target_energy": float(plan.get("target_energy", 0.7)),
        "target_tempo": float(plan.get("target_tempo", 120)),
    }
    retrieved_song_titles = {s["title"] for s in retrieved}
    retrieved_full = [s for s in songs if s["title"] in retrieved_song_titles]
    pool = retrieved_full if len(retrieved_full) >= k else songs
    scored = recommend_songs(user_prefs, pool, k=k)
    result["scored"] = scored
    for song, score, reasons in scored:
        print(f"  🎵 {song['title']} — Score: {score:.2f} | {', '.join(reasons[:2])}")

    # STEP 4: VALIDATE
    print("\n" + "="*50)
    print("✅ STEP 4: VALIDATE — Running evaluator and guardrails...")
    print("="*50)

    evaluation = run_evaluation(user_query, user_prefs, scored)
    result["evaluation"] = evaluation
    if not evaluation["guardrail_passed"]:
        logger.warning(f"Guardrail failed for query: '{user_query}'")

    # STEP 5: EXPLAIN
    print("\n" + "="*50)
    print("🎤 STEP 5: EXPLAIN — Generating music critic explanations...")
    print("="*50)

    context = format_retrieved_context(retrieved)
    top_songs_text = "\n".join([f"- \"{s['title']}\" by {s['artist']} ({s['genre']}, {s['mood']}, energy={s['energy']})" for s, _, _ in scored])
    explain_prompt = f"""The user asked for: "{user_query}"

Retrieved context from catalog:
{context}

Top recommended songs after scoring:
{top_songs_text}

Write a short music critic explanation for each recommended song explaining why it fits the user's request. Use your VibeBot persona."""

    explanation_response = llm.invoke([SystemMessage(content=MUSIC_CRITIC_SYSTEM_PROMPT), HumanMessage(content=explain_prompt)])
    explanation = explanation_response.content.strip()
    result["explanation"] = explanation
    result["recommendations"] = scored
    print("\n" + explanation)
    print("\n" + "="*50)
    print("🏁 AGENT COMPLETE")
    print("="*50)
    return result