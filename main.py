"""
main.py - Command line runner for VibeFinder 2.0
"""

import logging
import os
from src.recommender import load_songs, recommend_songs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main() -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    songs_path = os.path.join(base_dir, "data", "songs.csv")

    songs = load_songs(songs_path)
    print(f"✅ Loaded {len(songs)} songs\n")

    profiles = [
        {
            "name": "High-Energy Pop",
            "prefs": {"favorite_genre": "pop", "favorite_mood": "happy", "target_energy": 0.9, "target_tempo": 128}
        },
        {
            "name": "Chill Lofi",
            "prefs": {"favorite_genre": "lofi", "favorite_mood": "chill", "target_energy": 0.3, "target_tempo": 75}
        },
        {
            "name": "Intense Rock",
            "prefs": {"favorite_genre": "rock", "favorite_mood": "intense", "target_energy": 0.85, "target_tempo": 150}
        },
    ]

    for profile in profiles:
        print("=" * 45)
        print(f"🎵 Profile: {profile['name']}")
        print("=" * 45)
        results = recommend_songs(profile["prefs"], songs, k=5)
        for song, score, reasons in results:
            print(f"  {song['title']} by {song['artist']} — Score: {score:.2f}")
            print(f"    └─ {', '.join(reasons)}")
        print()


if __name__ == "__main__":
    main()
