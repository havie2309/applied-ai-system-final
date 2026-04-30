"""
recommender.py - Core content-based filtering logic.
Migrated and extended from Module 3 VibeFinder 1.0.
"""

import csv
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class Song:
    """Represents a song and its attributes."""

    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float


@dataclass
class UserProfile:
    """Represents a user's taste preferences."""

    favorite_genre: str
    favorite_mood: str
    target_energy: float
    target_tempo: float
    target_acousticness: Optional[float] = None
    target_valence: Optional[float] = None


class Recommender:
    """OOP implementation of the recommendation logic."""

    def __init__(self, songs: List[Song]):
        self.songs = songs

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        """Returns top k songs scored against the user profile."""

        user_prefs = {
            "favorite_genre": user.favorite_genre,
            "favorite_mood": user.favorite_mood,
            "target_energy": user.target_energy,
            "target_tempo": user.target_tempo,
            "target_acousticness": user.target_acousticness,
            "target_valence": user.target_valence,
        }
        scored = sorted(
            self.songs,
            key=lambda song: score_song(user_prefs, song)[0],
            reverse=True,
        )
        return scored[:k]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        """Explains why a song was recommended."""

        score, reasons = score_song(
            {
                "favorite_genre": user.favorite_genre,
                "favorite_mood": user.favorite_mood,
                "target_energy": user.target_energy,
                "target_tempo": user.target_tempo,
                "target_acousticness": user.target_acousticness,
                "target_valence": user.target_valence,
            },
            song.__dict__,
        )
        del score
        return ", ".join(reasons)


def load_songs(csv_path: str) -> List[Dict]:
    """Loads songs from a CSV file and returns a list of dictionaries."""

    songs: List[Dict] = []
    try:
        with open(csv_path, newline="", encoding="utf-8") as file_handle:
            reader = csv.DictReader(file_handle)
            for row in reader:
                row["id"] = int(row["id"])
                row["energy"] = float(row["energy"])
                row["tempo_bpm"] = float(row["tempo_bpm"])
                row["valence"] = float(row["valence"])
                row["danceability"] = float(row["danceability"])
                row["acousticness"] = float(row["acousticness"])
                songs.append(row)
        logger.info("Loaded %s songs from %s", len(songs), csv_path)
    except FileNotFoundError:
        logger.error("Songs file not found: %s", csv_path)
        raise
    except Exception as exc:
        logger.error("Error loading songs: %s", exc)
        raise
    return songs


def _optional_similarity(
    target_value: Optional[float],
    song_value: float,
    weight: float,
    label: str,
) -> Tuple[float, Optional[str]]:
    """Scores an optional numeric preference when a target is present."""

    if target_value is None:
        return 0.0, None

    similarity = max(0.0, 1.0 - abs(song_value - target_value))
    weighted_score = similarity * weight
    reason = f"{label} similarity (+{weighted_score:.2f})"
    return weighted_score, reason


def score_song(user_prefs: Dict, song: Dict) -> Tuple[float, List[str]]:
    """Scores a single song against user preferences using weighted attributes."""

    score = 0.0
    reasons: List[str] = []

    if song["genre"] == user_prefs["favorite_genre"]:
        score += 2.0
        reasons.append("genre match (+2.0)")

    if song["mood"] == user_prefs["favorite_mood"]:
        score += 1.5
        reasons.append("mood match (+1.5)")

    energy_sim = max(0.0, 1.0 - abs(song["energy"] - user_prefs["target_energy"]))
    score += energy_sim
    reasons.append(f"energy similarity (+{energy_sim:.2f})")

    tempo_sim = max(
        0.0,
        1.0 - abs(song["tempo_bpm"] - user_prefs["target_tempo"]) / 120,
    )
    tempo_score = tempo_sim * 0.5
    score += tempo_score
    reasons.append(f"tempo similarity (+{tempo_score:.2f})")

    acoustic_score, acoustic_reason = _optional_similarity(
        user_prefs.get("target_acousticness"),
        float(song["acousticness"]),
        0.25,
        "acousticness",
    )
    score += acoustic_score
    if acoustic_reason:
        reasons.append(acoustic_reason)

    valence_score, valence_reason = _optional_similarity(
        user_prefs.get("target_valence"),
        float(song["valence"]),
        0.25,
        "valence",
    )
    score += valence_score
    if valence_reason:
        reasons.append(valence_reason)

    return score, reasons


def recommend_songs(
    user_prefs: Dict,
    songs: List[Dict],
    k: int = 5,
    exclude_titles: Optional[Set[str]] = None,
) -> List[Tuple[Dict, float, List[str]]]:
    """Scores all songs, sorts by score descending, and returns the top k."""

    if not songs:
        logger.warning("No songs provided to recommend_songs")
        return []

    excluded = exclude_titles or set()
    scored: List[Tuple[Dict, float, List[str]]] = []
    for song in songs:
        if song["title"] in excluded:
            continue
        score, reasons = score_song(user_prefs, song)
        scored.append((song, score, reasons))

    scored.sort(key=lambda item: item[1], reverse=True)
    logger.info("Returning top %s recommendations", k)
    return scored[:k]
