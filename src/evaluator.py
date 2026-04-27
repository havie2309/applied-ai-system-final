"""
evaluator.py - Reliability, confidence scoring, and guardrails for VibeFinder 2.0.
Tracks what the AI does, measures how confident results are,
and enforces safety rules before returning recommendations.
"""

import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Tuple, Any

logger = logging.getLogger(__name__)

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "vibefinder.log")

# File handler so every run is saved to disk
file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s"))
logging.getLogger().addHandler(file_handler)


# ── Guardrail Rules ──────────────────────────────────────────────────────────

VALID_GENRES = {
    "pop", "lofi", "rock", "jazz", "metal", "classical",
    "r&b", "electronic", "indie", "folk", "ambient",
    "synthwave", "indie pop", "rnb", "unknown"
}

VALID_MOODS = {
    "happy", "chill", "intense", "relaxed", "melancholy",
    "energetic", "peaceful", "moody", "focused", "unknown"
}

MIN_CONFIDENCE_THRESHOLD = 0.40
MIN_RESULTS_REQUIRED = 3


def validate_user_prefs(user_prefs: Dict) -> Tuple[bool, List[str]]:
    """
    Guardrail: Validates that user preferences are within acceptable bounds.
    Returns (is_valid, list_of_warnings).
    """
    warnings = []

    genre = user_prefs.get("favorite_genre", "").lower()
    if genre not in VALID_GENRES:
        warnings.append(f"Unrecognized genre '{genre}' — results may be less accurate")

    mood = user_prefs.get("favorite_mood", "").lower()
    if mood not in VALID_MOODS:
        warnings.append(f"Unrecognized mood '{mood}' — results may be less accurate")

    energy = user_prefs.get("target_energy", -1)
    if not (0.0 <= float(energy) <= 1.0):
        warnings.append(f"Energy {energy} out of range [0.0–1.0] — clamping to nearest bound")

    tempo = user_prefs.get("target_tempo", -1)
    if not (40 <= float(tempo) <= 220):
        warnings.append(f"Tempo {tempo} BPM out of range [40–220] — results may be unreliable")

    is_valid = len(warnings) == 0
    return is_valid, warnings


def compute_confidence(scored_results: List[Tuple[Dict, float, List[str]]]) -> Dict:
    """
    Computes a confidence report from scored recommendation results.

    Confidence is based on:
    - Average score relative to max possible (5.0)
    - Whether top results have genre/mood matches
    - Score spread between top and bottom result
    """
    if not scored_results:
        return {"confidence": 0.0, "level": "none", "details": "No results to evaluate"}

    scores = [score for _, score, _ in scored_results]
    max_possible = 5.0

    avg_score = sum(scores) / len(scores)
    top_score = scores[0]
    bottom_score = scores[-1]
    spread = round(top_score - bottom_score, 3)

    # Count how many top results have explicit genre/mood matches
    strong_matches = 0
    for _, _, reasons in scored_results:
        reason_text = " ".join(reasons)
        if "genre match" in reason_text and "mood match" in reason_text:
            strong_matches += 1

    match_ratio = strong_matches / len(scored_results)
    confidence = round((avg_score / max_possible) * 0.7 + match_ratio * 0.3, 3)

    if confidence >= 0.75:
        level = "high"
    elif confidence >= 0.50:
        level = "medium"
    elif confidence >= 0.40:
        level = "low"
    else:
        level = "very_low"

    return {
        "confidence": confidence,
        "level": level,
        "avg_score": round(avg_score, 3),
        "top_score": round(top_score, 3),
        "spread": spread,
        "strong_matches": strong_matches,
        "match_ratio": round(match_ratio, 3),
        "details": f"{strong_matches}/{len(scored_results)} results had both genre and mood match"
    }


def enforce_guardrails(
    user_prefs: Dict,
    scored_results: List[Tuple],
    confidence_report: Dict
) -> Tuple[bool, List[str]]:
    """
    Enforces safety guardrails before returning results to the user.
    Returns (passed, list_of_issues).
    """
    issues = []

    if len(scored_results) < MIN_RESULTS_REQUIRED:
        issues.append(f"Too few results ({len(scored_results)} < {MIN_RESULTS_REQUIRED} required)")

    if confidence_report["confidence"] < MIN_CONFIDENCE_THRESHOLD:
        issues.append(
            f"Confidence too low ({confidence_report['confidence']:.3f} < {MIN_CONFIDENCE_THRESHOLD}) "
            f"— consider broadening your preferences"
        )

    passed = len(issues) == 0
    return passed, issues


def log_interaction(
    query: str,
    user_prefs: Dict,
    confidence_report: Dict,
    guardrail_passed: bool,
    warnings: List[str],
    issues: List[str]
) -> str:
    """
    Logs a full interaction to disk as JSON for audit and review.
    Returns the log entry ID.
    """
    entry_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log_entry = {
        "id": entry_id,
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "user_prefs": user_prefs,
        "confidence": confidence_report,
        "guardrail_passed": guardrail_passed,
        "warnings": warnings,
        "issues": issues,
    }

    log_path = os.path.join(LOG_DIR, f"interaction_{entry_id}.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_entry, f, indent=2)

    logger.info(f"Interaction logged: {entry_id} | confidence={confidence_report['confidence']} | guardrail={'PASS' if guardrail_passed else 'FAIL'}")
    return entry_id


def print_evaluation_report(
    confidence_report: Dict,
    guardrail_passed: bool,
    warnings: List[str],
    issues: List[str],
    entry_id: str
) -> None:
    """Prints a formatted evaluation report to the console."""
    print("\n" + "="*50)
    print("📋 EVALUATION REPORT")
    print("="*50)
    print(f"  🆔 Interaction ID : {entry_id}")
    print(f"  📊 Confidence     : {confidence_report['confidence']:.3f} ({confidence_report['level'].upper()})")
    print(f"  🎯 Avg Score      : {confidence_report['avg_score']:.3f} / 5.0")
    print(f"  🏆 Top Score      : {confidence_report['top_score']:.3f}")
    print(f"  📉 Score Spread   : {confidence_report['spread']:.3f}")
    print(f"  ✅ Strong Matches : {confidence_report['details']}")

    if warnings:
        print("\n  ⚠️  Warnings:")
        for w in warnings:
            print(f"     - {w}")

    if issues:
        print("\n  🚫 Guardrail Issues:")
        for i in issues:
            print(f"     - {i}")

    print(f"\n  🛡️  Guardrail: {'✅ PASSED' if guardrail_passed else '❌ FAILED'}")
    print("="*50)


def run_evaluation(
    query: str,
    user_prefs: Dict,
    scored_results: List[Tuple]
) -> Dict:
    """
    Full evaluation pipeline: validate → score confidence → enforce guardrails → log.
    Returns a complete evaluation report dict.
    """
    # 1. Validate inputs
    is_valid, warnings = validate_user_prefs(user_prefs)

    # 2. Compute confidence
    confidence_report = compute_confidence(scored_results)

    # 3. Enforce guardrails
    guardrail_passed, issues = enforce_guardrails(user_prefs, scored_results, confidence_report)

    # 4. Log interaction
    entry_id = log_interaction(query, user_prefs, confidence_report, guardrail_passed, warnings, issues)

    # 5. Print report
    print_evaluation_report(confidence_report, guardrail_passed, warnings, issues, entry_id)

    return {
        "entry_id": entry_id,
        "confidence_report": confidence_report,
        "guardrail_passed": guardrail_passed,
        "warnings": warnings,
        "issues": issues,
    }