"""
recommender.py - Core content-based filtering logic.
Migrated and extended from Module 3 VibeFinder 1.0.
"""

import csv
import logging
from typing import List, Dict, Tuple
from dataclasses import dataclass

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
    likes_acoustic: bool


class Recommender:
    """OOP implementation of the recommendation logic."""

    def __init__(self, songs: List[Song]):
        self.songs = songs

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        """Returns top k songs scored against the user profile."""
        scored = sorted(self.songs, key=lambda s: self._score(user, s), reverse=True)
        return scored[:k]

    def _score(self, user: UserProfile, song: Song) -> float:
        """Calculates a match score between a user and a song."""
        score = 0.0
        if song.genre == user.favorite_genre:
            score += 2.0
        if song.mood == user.favorite_mood:
            score += 1.5
        score += 1.0 - abs(song.energy - user.target_energy)
        return score

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        """Explains why a song was recommended."""
        reasons = []
        if song.genre == user.favorite_genre:
            reasons.append("genre match")
        if song.mood == user.favorite_mood:
            reasons.append("mood match")
        energy_sim = 1.0 - abs(song.energy - user.target_energy)
        reasons.append(f"energy similarity ({energy_sim:.2f})")
        return ", ".join(reasons)


def load_songs(csv_path: str) -> List[Dict]:
    """Loads songs from a CSV file and returns a list of dictionaries."""
    songs = []
    try:
        with open(csv_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                row['id'] = int(row['id'])
                row['energy'] = float(row['energy'])
                row['tempo_bpm'] = float(row['tempo_bpm'])
                row['valence'] = float(row['valence'])
                row['danceability'] = float(row['danceability'])
                row['acousticness'] = float(row['acousticness'])
                songs.append(row)
        logger.info(f"Loaded {len(songs)} songs from {csv_path}")
    except FileNotFoundError:
        logger.error(f"Songs file not found: {csv_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading songs: {e}")
        raise
    return songs


def score_song(user_prefs: Dict, song: Dict) -> Tuple[float, List[str]]:
    """Scores a single song against user preferences using weighted attributes."""
    score = 0.0
    reasons = []

    if song['genre'] == user_prefs['favorite_genre']:
        score += 2.0
        reasons.append('genre match (+2.0)')

    if song['mood'] == user_prefs['favorite_mood']:
        score += 1.5
        reasons.append('mood match (+1.5)')

    energy_sim = 1.0 - abs(song['energy'] - user_prefs['target_energy'])
    score += energy_sim
    reasons.append(f'energy similarity (+{energy_sim:.2f})')

    tempo_sim = max(0.0, 1.0 - abs(song['tempo_bpm'] - user_prefs['target_tempo']) / 120)
    score += tempo_sim * 0.5
    reasons.append(f'tempo similarity (+{tempo_sim * 0.5:.2f})')

    return (score, reasons)


def recommend_songs(user_prefs: Dict, songs: List[Dict], k: int = 5) -> List[Tuple[Dict, float, List[str]]]:
    """Scores all songs, sorts by score descending, and returns the top k."""
    if not songs:
        logger.warning("No songs provided to recommend_songs")
        return []

    scored = []
    for song in songs:
        score, reasons = score_song(user_prefs, song)
        scored.append((song, score, reasons))
    scored = sorted(scored, key=lambda x: x[1], reverse=True)
    logger.info(f"Returning top {k} recommendations")
    return scored[:k]
