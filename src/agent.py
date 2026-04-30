"""
agent.py - Agentic workflow for VibeFinder 2.0.
Steps: PLAN -> RETRIEVE -> SCORE -> VALIDATE -> EXPLAIN
"""

import json
import logging
import os
import re
from copy import deepcopy
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.config import openai_api_key_help_text, resolve_openai_api_key
from src.evaluator import run_evaluation
from src.rag import format_retrieved_context, retrieve_similar_songs
from src.recommender import recommend_songs

load_dotenv()
logger = logging.getLogger(__name__)

DEFAULT_PROFILE = {
    "favorite_genre": "pop",
    "favorite_mood": "happy",
    "target_energy": 0.7,
    "target_tempo": 120.0,
    "target_acousticness": None,
    "target_valence": None,
}

MUSIC_CRITIC_SYSTEM_PROMPT = """You are VibeBot, an expert music curator and critic with deep knowledge of genres, moods, and sonic textures.

Your personality:
- Speak like a knowledgeable but approachable music journalist
- Use specific musical vocabulary (timbre, groove, dynamics, texture)
- Always explain WHY a song fits, not just THAT it fits
- Be enthusiastic but precise with no generic praise
- Keep each recommendation to 2-3 sentences maximum

Few-shot examples of your style:

User: Why does "Library Rain" fit a chill lofi study session?
VibeBot: "Library Rain" earns its spot through its unhurried 72 BPM pulse and high acousticness (0.86), which creates that warm, analog texture lofi listeners crave. The low energy (0.35) means it stays out of your way cognitively, making it strong background architecture for deep focus.

User: Why does "Storm Runner" fit an intense rock workout?
VibeBot: "Storm Runner" is built for physical exertion: its 152 BPM tempo drives relentless forward momentum while the 0.91 energy score means it never lets up. The intense mood tag is earned, not labeled, because the track has enough sonic weight to push through a final rep.

Return JSON only in the requested format."""


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _coerce_optional_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    return float(value)


def sanitize_profile(user_prefs: Dict[str, Any]) -> Dict[str, Any]:
    """Returns a normalized profile dict with clamped numeric values."""

    profile = deepcopy(DEFAULT_PROFILE)
    profile["favorite_genre"] = str(
        user_prefs.get("favorite_genre", DEFAULT_PROFILE["favorite_genre"])
    )
    profile["favorite_mood"] = str(
        user_prefs.get("favorite_mood", DEFAULT_PROFILE["favorite_mood"])
    )
    profile["target_energy"] = _clamp(
        float(user_prefs.get("target_energy", DEFAULT_PROFILE["target_energy"])),
        0.0,
        1.0,
    )
    profile["target_tempo"] = _clamp(
        float(user_prefs.get("target_tempo", DEFAULT_PROFILE["target_tempo"])),
        40.0,
        220.0,
    )

    target_acousticness = _coerce_optional_float(user_prefs.get("target_acousticness"))
    profile["target_acousticness"] = (
        None if target_acousticness is None else _clamp(target_acousticness, 0.0, 1.0)
    )

    target_valence = _coerce_optional_float(user_prefs.get("target_valence"))
    profile["target_valence"] = (
        None if target_valence is None else _clamp(target_valence, 0.0, 1.0)
    )
    return profile


def build_plan_from_profile(profile: Dict[str, Any], summary: str) -> Dict[str, Any]:
    """Creates a plan-like payload for the UI from an active profile."""

    return {
        "favorite_genre": profile.get("favorite_genre", "unknown"),
        "favorite_mood": profile.get("favorite_mood", "unknown"),
        "target_energy": round(float(profile.get("target_energy", 0.0)), 2),
        "target_tempo": int(round(float(profile.get("target_tempo", 0.0)))),
        "summary": summary,
    }


def build_rag_query(profile: Dict[str, Any]) -> str:
    """Builds the retrieval query from a normalized preference profile."""

    parts = [
        str(profile.get("favorite_genre", "unknown")),
        str(profile.get("favorite_mood", "unknown")),
        "music",
        f"energy {profile.get('target_energy', 0.0):.2f}",
        f"tempo {profile.get('target_tempo', 0.0):.0f} bpm",
    ]

    acousticness = profile.get("target_acousticness")
    if acousticness is not None:
        parts.append(f"acousticness {float(acousticness):.2f}")

    valence = profile.get("target_valence")
    if valence is not None:
        parts.append(f"valence {float(valence):.2f}")

    return " ".join(parts)


def _print_step_header(step_name: str, detail: str) -> None:
    print("\n" + "=" * 50)
    print(f"{step_name} - {detail}")
    print("=" * 50)


def _build_result(
    plan: Dict[str, Any],
    retrieved: List[Dict[str, Any]],
    scored: List[Tuple[Dict[str, Any], float, List[str]]],
    evaluation: Dict[str, Any],
    explanations: List[str],
    active_profile: Dict[str, Any],
    refinement_history: List[str],
    mode: str,
) -> Dict[str, Any]:
    return {
        "plan": plan,
        "retrieved": retrieved,
        "scored": scored,
        "evaluation": evaluation,
        "explanations": explanations,
        "active_profile": deepcopy(active_profile),
        "refinement_history": list(refinement_history),
        "mode": mode,
        "recommendations": scored,
    }


def _retrieve_and_score(
    profile: Dict[str, Any],
    songs: List[Dict[str, Any]],
    vectorstore: Any,
    k: int,
    exclude_titles: Optional[Set[str]] = None,
) -> Tuple[List[Dict[str, Any]], List[Tuple[Dict[str, Any], float, List[str]]]]:
    """Runs retrieval and content-based scoring for a profile."""

    excluded = exclude_titles or set()
    rag_query = build_rag_query(profile)
    retrieved = retrieve_similar_songs(rag_query, vectorstore, k=max(k * 2, 10))
    retrieved = [song for song in retrieved if song["title"] not in excluded]

    available_songs = [song for song in songs if song["title"] not in excluded]
    retrieved_titles = {song["title"] for song in retrieved}
    retrieved_full = [song for song in available_songs if song["title"] in retrieved_titles]
    pool = retrieved_full if len(retrieved_full) >= k else available_songs
    scored = recommend_songs(profile, pool, k=k, exclude_titles=excluded)
    return retrieved, scored


def _build_local_explanations(
    scored: Sequence[Tuple[Dict[str, Any], float, List[str]]],
    refinement_history: Sequence[str],
) -> List[str]:
    """Builds deterministic explanations for refinement results."""

    history_text = ", ".join(refinement_history)
    explanations: List[str] = []

    for song, _, reasons in scored:
        lower_reasons = [reason.lower() for reason in reasons]
        phrases: List[str] = []

        if any("genre match" in reason for reason in lower_reasons):
            phrases.append(f"stays rooted in {song['genre']}")
        if any("mood match" in reason for reason in lower_reasons):
            phrases.append(f"keeps the {song['mood']} mood in focus")
        if any("energy similarity" in reason for reason in lower_reasons):
            phrases.append(
                f"lands near your energy target with a {song['energy']:.2f} energy score"
            )
        if any("tempo similarity" in reason for reason in lower_reasons):
            phrases.append(f"sits close to your requested pace at {song['tempo_bpm']:.0f} BPM")
        if any("acousticness similarity" in reason for reason in lower_reasons):
            phrases.append(
                f"leans into an acoustic texture with acousticness at {song['acousticness']:.2f}"
            )
        if any("valence similarity" in reason for reason in lower_reasons):
            phrases.append(f"matches the brighter emotional tone with valence at {song['valence']:.2f}")

        if not phrases:
            phrases.append("still overlaps with the updated profile across several attributes")

        lead = f'"{song["title"]}" fits the refined direction because it '
        body = " and ".join(phrases[:3])
        if history_text:
            explanations.append(f"{lead}{body}. It remains a strong pick after {history_text.lower()}.")
        else:
            explanations.append(f"{lead}{body}. It should feel close to the track that anchored this branch.")

    return explanations


def _parse_json_payload(raw_content: str) -> Any:
    cleaned = re.sub(r"```json|```", "", raw_content.strip()).strip()
    return json.loads(cleaned)


def _generate_llm_explanations(
    user_query: str,
    retrieved: List[Dict[str, Any]],
    scored: Sequence[Tuple[Dict[str, Any], float, List[str]]],
    llm: ChatOpenAI,
) -> List[str]:
    """Returns explanations aligned to scored results."""

    if not scored:
        return []

    context = format_retrieved_context(retrieved)
    top_songs_text = "\n".join(
        [
            (
                f'- "{song["title"]}" by {song["artist"]} '
                f'({song["genre"]}, {song["mood"]}, energy={song["energy"]}, '
                f'tempo={song["tempo_bpm"]}, acousticness={song["acousticness"]}, '
                f'valence={song["valence"]})'
            )
            for song, _, _ in scored
        ]
    )
    explain_prompt = f"""The user asked for: "{user_query}"

Retrieved context from catalog:
{context}

Top recommended songs after scoring:
{top_songs_text}

Return ONLY JSON in this shape:
{{
  "explanations": [
    "short explanation for song 1",
    "short explanation for song 2"
  ]
}}

The explanations list must contain exactly {len(scored)} items and must stay in the same order as the recommended songs."""

    response = llm.invoke(
        [
            SystemMessage(content=MUSIC_CRITIC_SYSTEM_PROMPT),
            HumanMessage(content=explain_prompt),
        ]
    )

    try:
        payload = _parse_json_payload(response.content)
        explanations = payload.get("explanations", [])
        if isinstance(explanations, list) and len(explanations) == len(scored):
            return [str(item).strip() for item in explanations]
    except (json.JSONDecodeError, AttributeError, TypeError):
        logger.warning("Could not parse explanation JSON, using local fallback")

    return _build_local_explanations(scored, [])


def profile_from_song(song: Dict[str, Any]) -> Dict[str, Any]:
    """Creates a refinement profile centered on a specific song."""

    return sanitize_profile(
        {
            "favorite_genre": song["genre"],
            "favorite_mood": song["mood"],
            "target_energy": song["energy"],
            "target_tempo": song["tempo_bpm"],
            "target_acousticness": song["acousticness"],
            "target_valence": song["valence"],
        }
    )


def apply_refinement_action(profile: Dict[str, Any], action_id: str) -> Tuple[Dict[str, Any], str]:
    """Applies a refinement action and returns the updated profile plus label."""

    updated = sanitize_profile(profile)

    if action_id == "more_upbeat":
        updated["favorite_mood"] = "happy"
        updated["target_energy"] = _clamp(updated["target_energy"] + 0.10, 0.0, 1.0)
        updated["target_tempo"] = _clamp(updated["target_tempo"] + 8.0, 40.0, 220.0)
        target_valence = updated.get("target_valence")
        baseline_valence = 0.5 if target_valence is None else float(target_valence)
        updated["target_valence"] = _clamp(baseline_valence + 0.15, 0.0, 1.0)
        return updated, "More upbeat"

    if action_id == "more_chill":
        updated["favorite_mood"] = "chill"
        updated["target_energy"] = _clamp(updated["target_energy"] - 0.10, 0.0, 1.0)
        updated["target_tempo"] = _clamp(updated["target_tempo"] - 8.0, 40.0, 220.0)
        return updated, "More chill"

    if action_id == "more_energetic":
        updated["favorite_mood"] = "energetic"
        updated["target_energy"] = _clamp(updated["target_energy"] + 0.15, 0.0, 1.0)
        updated["target_tempo"] = _clamp(updated["target_tempo"] + 10.0, 40.0, 220.0)
        return updated, "More energetic"

    if action_id == "more_acoustic":
        target_acousticness = updated.get("target_acousticness")
        baseline_acousticness = 0.5 if target_acousticness is None else float(target_acousticness)
        updated["target_acousticness"] = _clamp(baseline_acousticness + 0.20, 0.0, 1.0)
        return updated, "More acoustic"

    if action_id == "faster":
        updated["target_tempo"] = _clamp(updated["target_tempo"] + 12.0, 40.0, 220.0)
        return updated, "Faster"

    if action_id == "slower":
        updated["target_tempo"] = _clamp(updated["target_tempo"] - 12.0, 40.0, 220.0)
        return updated, "Slower"

    raise ValueError(f"Unsupported refinement action: {action_id}")


def run_agent(user_query: str, songs: List[Dict[str, Any]], vectorstore: Any, k: int = 5) -> Dict[str, Any]:
    """Runs the initial planner-driven workflow."""

    api_key = resolve_openai_api_key()
    if not api_key:
        raise RuntimeError(openai_api_key_help_text())

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7,
        openai_api_key=api_key,
    )

    _print_step_header("STEP 1: PLAN", "Understanding your request...")
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
    try:
        plan = _parse_json_payload(plan_response.content)
    except json.JSONDecodeError:
        logger.warning("Could not parse plan JSON, using defaults")
        plan = {
            "favorite_genre": DEFAULT_PROFILE["favorite_genre"],
            "favorite_mood": DEFAULT_PROFILE["favorite_mood"],
            "target_energy": DEFAULT_PROFILE["target_energy"],
            "target_tempo": DEFAULT_PROFILE["target_tempo"],
            "summary": user_query,
        }

    raw_profile = {
        "favorite_genre": plan.get("favorite_genre", DEFAULT_PROFILE["favorite_genre"]),
        "favorite_mood": plan.get("favorite_mood", DEFAULT_PROFILE["favorite_mood"]),
        "target_energy": plan.get("target_energy", DEFAULT_PROFILE["target_energy"]),
        "target_tempo": plan.get("target_tempo", DEFAULT_PROFILE["target_tempo"]),
        "target_acousticness": plan.get("target_acousticness"),
        "target_valence": plan.get("target_valence"),
    }

    active_profile = sanitize_profile(raw_profile)
    plan = build_plan_from_profile(active_profile, str(plan.get("summary", user_query)))

    print(f"  Genre:   {plan.get('favorite_genre')}")
    print(f"  Mood:    {plan.get('favorite_mood')}")
    print(f"  Energy:  {plan.get('target_energy')}")
    print(f"  Tempo:   {plan.get('target_tempo')} BPM")
    print(f"  Summary: {plan.get('summary')}")

    _print_step_header("STEP 2: RETRIEVE", "Searching song catalog via RAG...")
    retrieved, scored = _retrieve_and_score(active_profile, songs, vectorstore, k=k)
    for song in retrieved:
        print(
            f"  {song['title']} by {song['artist']} - similarity: "
            f"{song['similarity_score']:.3f}"
        )

    _print_step_header("STEP 3: SCORE", "Running content-based scoring...")
    for song, score, reasons in scored:
        print(f"  {song['title']} - Score: {score:.2f} | {', '.join(reasons[:2])}")

    _print_step_header("STEP 4: VALIDATE", "Running evaluator and guardrails...")
    evaluation = run_evaluation(user_query, raw_profile, scored)
    if not evaluation["guardrail_passed"]:
        logger.warning("Guardrail failed for query: '%s'", user_query)

    _print_step_header("STEP 5: EXPLAIN", "Generating music critic explanations...")
    explanations = _generate_llm_explanations(user_query, retrieved, scored, llm)
    for explanation in explanations:
        print(f"  {explanation}")

    print("\n" + "=" * 50)
    print("AGENT COMPLETE")
    print("=" * 50)

    return _build_result(
        plan=plan,
        retrieved=retrieved,
        scored=scored,
        evaluation=evaluation,
        explanations=explanations,
        active_profile=active_profile,
        refinement_history=[],
        mode="initial",
    )


def refine_recommendations(
    user_query: str,
    active_profile: Dict[str, Any],
    songs: List[Dict[str, Any]],
    vectorstore: Any,
    k: int = 5,
    refinement_history: Optional[Iterable[str]] = None,
    exclude_titles: Optional[Set[str]] = None,
    plan_summary: Optional[str] = None,
) -> Dict[str, Any]:
    """Runs the fast refinement path without planner or LLM explanations."""

    history = list(refinement_history or [])
    profile = sanitize_profile(active_profile)

    _print_step_header("STEP 2: RETRIEVE", "Refreshing recommendations from the active profile...")
    retrieved, scored = _retrieve_and_score(
        profile,
        songs,
        vectorstore,
        k=k,
        exclude_titles=exclude_titles,
    )
    for song in retrieved:
        print(
            f"  {song['title']} by {song['artist']} - similarity: "
            f"{song['similarity_score']:.3f}"
        )

    _print_step_header("STEP 3: SCORE", "Applying fast local reranking...")
    for song, score, reasons in scored:
        print(f"  {song['title']} - Score: {score:.2f} | {', '.join(reasons[:2])}")

    _print_step_header("STEP 4: VALIDATE", "Recomputing confidence and guardrails...")
    evaluation = run_evaluation(user_query, profile, scored)
    if not evaluation["guardrail_passed"]:
        logger.warning("Guardrail failed during refinement for query: '%s'", user_query)

    explanations = _build_local_explanations(scored, history)
    summary = plan_summary or (
        f"Refined from your request with {', '.join(history)}."
        if history
        else "Refined from a selected recommendation."
    )

    return _build_result(
        plan=build_plan_from_profile(profile, summary),
        retrieved=retrieved,
        scored=scored,
        evaluation=evaluation,
        explanations=explanations,
        active_profile=profile,
        refinement_history=history,
        mode="refined",
    )
