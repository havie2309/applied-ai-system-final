"""
test_harness.py - Automated test harness for VibeFinder 2.0.
Runs the system on predefined inputs and prints a summary report
with pass/fail scores, confidence ratings, and guardrail results.

This is the stretch feature test harness — it tests the full pipeline
end-to-end and evaluates reliability across multiple scenarios.
"""

import os
import sys
import json
import time
import logging
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.recommender import load_songs, recommend_songs
from src.rag import build_vectorstore, retrieve_similar_songs
from src.evaluator import compute_confidence, validate_user_prefs, enforce_guardrails

logging.basicConfig(level=logging.WARNING)  # suppress info logs during tests

# ── Test Cases ────────────────────────────────────────────────────────────────

TEST_CASES = [
    {
        "id": "TC01",
        "name": "Chill Lofi Study",
        "query": "I want chill lofi music to study late at night",
        "user_prefs": {"favorite_genre": "lofi", "favorite_mood": "chill", "target_energy": 0.3, "target_tempo": 75},
        "expected_genre": "lofi",
        "expected_mood": "chill",
        "min_confidence": 0.6,
        "min_results": 5,
    },
    {
        "id": "TC02",
        "name": "Intense Rock Workout",
        "query": "High energy intense rock for working out",
        "user_prefs": {"favorite_genre": "rock", "favorite_mood": "intense", "target_energy": 0.9, "target_tempo": 155},
        "expected_genre": "rock",
        "expected_mood": "intense",
        "min_confidence": 0.6,
        "min_results": 5,
    },
    {
        "id": "TC03",
        "name": "Happy Pop Party",
        "query": "Happy upbeat pop music for a party",
        "user_prefs": {"favorite_genre": "pop", "favorite_mood": "happy", "target_energy": 0.85, "target_tempo": 128},
        "expected_genre": "pop",
        "expected_mood": "happy",
        "min_confidence": 0.6,
        "min_results": 5,
    },
    {
        "id": "TC04",
        "name": "Jazz Relaxation",
        "query": "Relaxing jazz music for a quiet evening",
        "user_prefs": {"favorite_genre": "jazz", "favorite_mood": "relaxed", "target_energy": 0.4, "target_tempo": 90},
        "expected_genre": "jazz",
        "expected_mood": "relaxed",
        "min_confidence": 0.5,
        "min_results": 3,
    },
    {
        "id": "TC05",
        "name": "Metal Intensity",
        "query": "Extreme metal music as intense as possible",
        "user_prefs": {"favorite_genre": "metal", "favorite_mood": "intense", "target_energy": 0.98, "target_tempo": 180},
        "expected_genre": "metal",
        "expected_mood": "intense",
        "min_confidence": 0.5,
        "min_results": 3,
    },
    {
        "id": "TC06",
        "name": "Unknown Genre Guardrail",
        "query": "I want some vaporwave dreampop fusion music",
        "user_prefs": {"favorite_genre": "vaporwave", "favorite_mood": "dreamy", "target_energy": 0.5, "target_tempo": 100},
        "expected_genre": "unknown",
        "expected_mood": "unknown",
        "min_confidence": 0.0,   # expect low confidence
        "min_results": 1,
        "expect_warnings": True,
    },
    {
        "id": "TC07",
        "name": "Classical Peaceful",
        "query": "Peaceful classical music for reading",
        "user_prefs": {"favorite_genre": "classical", "favorite_mood": "peaceful", "target_energy": 0.2, "target_tempo": 60},
        "expected_genre": "classical",
        "expected_mood": "peaceful",
        "min_confidence": 0.4,
        "min_results": 3,
    },
    {
        "id": "TC08",
        "name": "Energy Out of Range Guardrail",
        "query": "Music with extreme energy",
        "user_prefs": {"favorite_genre": "electronic", "favorite_mood": "energetic", "target_energy": 1.5, "target_tempo": 140},
        "expected_genre": "electronic",
        "expected_mood": "energetic",
        "min_confidence": 0.0,
        "min_results": 1,
        "expect_warnings": True,
    },
]


# ── Individual Test Runner ────────────────────────────────────────────────────

def run_single_test(tc: Dict, songs: List[Dict], vectorstore) -> Dict:
    """Runs one test case and returns a result dict."""
    result = {
        "id": tc["id"],
        "name": tc["name"],
        "checks": {},
        "passed": 0,
        "failed": 0,
        "warnings_triggered": False,
        "confidence": 0.0,
        "confidence_level": "none",
        "guardrail_passed": False,
        "errors": [],
    }

    try:
        # 1. Validate preferences
        is_valid, warnings = validate_user_prefs(tc["user_prefs"])
        result["warnings_triggered"] = len(warnings) > 0

        # 2. Run RAG retrieval
        rag_query = (
            f"{tc['user_prefs']['favorite_genre']} {tc['user_prefs']['favorite_mood']} "
            f"music energy {tc['user_prefs']['target_energy']} tempo {tc['user_prefs']['target_tempo']} bpm"
        )
        retrieved = retrieve_similar_songs(rag_query, vectorstore, k=10)

        # 3. Score songs
        retrieved_titles = {s["title"] for s in retrieved}
        pool = [s for s in songs if s["title"] in retrieved_titles]
        if len(pool) < 5:
            pool = songs
        scored = recommend_songs(tc["user_prefs"], pool, k=5)

        # 4. Compute confidence
        conf = compute_confidence(scored)
        result["confidence"] = conf["confidence"]
        result["confidence_level"] = conf["level"]

        # 5. Enforce guardrails
        guardrail_passed, issues = enforce_guardrails(tc["user_prefs"], scored, conf)
        result["guardrail_passed"] = guardrail_passed

        # ── Checks ────────────────────────────────────────────────────────────

        # Check: Enough results returned
        check_results = len(scored) >= tc["min_results"]
        result["checks"]["min_results"] = check_results
        if check_results: result["passed"] += 1
        else: result["failed"] += 1

        # Check: Confidence meets threshold
        check_conf = result["confidence"] >= tc["min_confidence"]
        result["checks"]["min_confidence"] = check_conf
        if check_conf: result["passed"] += 1
        else: result["failed"] += 1

        # Check: Top result matches expected genre (if not unknown)
        if tc["expected_genre"] != "unknown" and scored:
            top_song = scored[0][0]
            genre_match = top_song["genre"].lower() == tc["expected_genre"].lower()
            result["checks"]["top_genre_match"] = genre_match
            if genre_match: result["passed"] += 1
            else: result["failed"] += 1
        else:
            result["checks"]["top_genre_match"] = "skipped (unknown genre)"

        # Check: Warnings triggered when expected
        if tc.get("expect_warnings"):
            check_warn = result["warnings_triggered"]
            result["checks"]["warnings_triggered"] = check_warn
            if check_warn: result["passed"] += 1
            else: result["failed"] += 1

        # Check: RAG returned results
        check_rag = len(retrieved) > 0
        result["checks"]["rag_returned_results"] = check_rag
        if check_rag: result["passed"] += 1
        else: result["failed"] += 1

        # Check: Scores are in valid range
        all_valid_scores = all(0 <= score <= 5.5 for _, score, _ in scored)
        result["checks"]["valid_score_range"] = all_valid_scores
        if all_valid_scores: result["passed"] += 1
        else: result["failed"] += 1

        result["top_songs"] = [s["title"] for s, _, _ in scored[:3]]

    except Exception as e:
        result["errors"].append(str(e))
        result["failed"] += 1

    return result


# ── Test Harness Runner ───────────────────────────────────────────────────────

def run_harness():
    print("\n" + "█" * 60)
    print("  🧪 VIBEFINDER 2.0 — AUTOMATED TEST HARNESS")
    print("█" * 60)

    # Load resources
    print("\n⏳ Loading songs and building vector store...")
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    songs = load_songs(os.path.join(base, "data", "songs.csv"))
    vectorstore = build_vectorstore(songs)
    print(f"✅ Loaded {len(songs)} songs | {len(TEST_CASES)} test cases queued\n")

    results = []
    total_passed = 0
    total_failed = 0
    total_checks = 0

    for tc in TEST_CASES:
        print(f"  ▶ Running {tc['id']}: {tc['name']}...", end=" ", flush=True)
        start = time.time()
        r = run_single_test(tc, songs, vectorstore)
        elapsed = round(time.time() - start, 2)
        results.append(r)

        total_passed += r["passed"]
        total_failed += r["failed"]
        total_checks += r["passed"] + r["failed"]

        status = "✅ PASS" if r["failed"] == 0 and not r["errors"] else "❌ FAIL"
        print(f"{status} ({r['passed']}/{r['passed']+r['failed']} checks, {elapsed}s)")

    # ── Summary Report ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  📋 TEST HARNESS SUMMARY REPORT")
    print("=" * 60)

    pass_rate = round(total_passed / total_checks * 100, 1) if total_checks > 0 else 0
    avg_confidence = round(sum(r["confidence"] for r in results) / len(results), 3)

    print(f"\n  {'Metric':<30} {'Value'}")
    print(f"  {'-'*48}")
    print(f"  {'Total Test Cases':<30} {len(TEST_CASES)}")
    print(f"  {'Total Checks Run':<30} {total_checks}")
    print(f"  {'Checks Passed':<30} {total_passed}")
    print(f"  {'Checks Failed':<30} {total_failed}")
    print(f"  {'Pass Rate':<30} {pass_rate}%")
    print(f"  {'Avg Confidence Score':<30} {avg_confidence}")
    print(f"  {'Guardrails Triggered':<30} {sum(1 for r in results if r['warnings_triggered'])}/{len(results)}")

    print(f"\n  {'ID':<6} {'Test Name':<28} {'Checks':<10} {'Confidence':<14} {'Status'}")
    print(f"  {'-'*70}")
    for r in results:
        total = r["passed"] + r["failed"]
        status = "✅ PASS" if r["failed"] == 0 and not r["errors"] else "❌ FAIL"
        print(f"  {r['id']:<6} {r['name']:<28} {r['passed']}/{total:<8} {r['confidence']:.3f} ({r['confidence_level']:<8}) {status}")
        if r["errors"]:
            for e in r["errors"]:
                print(f"         ⚠️  Error: {e}")

    print("\n" + "=" * 60)
    print(f"  🏁 RESULT: {total_passed}/{total_checks} checks passed ({pass_rate}%)")
    if pass_rate >= 80:
        print("  🟢 System reliability: GOOD")
    elif pass_rate >= 60:
        print("  🟡 System reliability: ACCEPTABLE")
    else:
        print("  🔴 System reliability: NEEDS IMPROVEMENT")
    print("=" * 60 + "\n")

    # Save report to JSON
    report_path = os.path.join(base, "logs", "test_harness_report.json")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        json.dump({
            "summary": {
                "total_cases": len(TEST_CASES),
                "total_checks": total_checks,
                "passed": total_passed,
                "failed": total_failed,
                "pass_rate": pass_rate,
                "avg_confidence": avg_confidence,
            },
            "results": [
                {k: v for k, v in r.items() if k != "errors" or v}
                for r in results
            ]
        }, f, indent=2)
    print(f"  📄 Full report saved to: {report_path}\n")


if __name__ == "__main__":
    run_harness()
