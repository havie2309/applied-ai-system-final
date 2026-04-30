from langchain_core.documents import Document

from src.agent import apply_refinement_action, profile_from_song, refine_recommendations
from src.recommender import score_song


class FakeVectorStore:
    def __init__(self, results):
        self.results = results

    def similarity_search_with_score(self, query, k=5):
        del query
        return self.results[:k]


def _song(
    song_id,
    title,
    genre,
    mood,
    energy,
    tempo_bpm,
    acousticness,
    valence,
):
    return {
        "id": song_id,
        "title": title,
        "artist": f"Artist {song_id}",
        "genre": genre,
        "mood": mood,
        "energy": energy,
        "tempo_bpm": tempo_bpm,
        "valence": valence,
        "danceability": 0.5,
        "acousticness": acousticness,
    }


def _doc(song, distance):
    return (
        Document(
            page_content=f"{song['title']} by {song['artist']}",
            metadata={
                "title": song["title"],
                "artist": song["artist"],
                "genre": song["genre"],
                "mood": song["mood"],
                "energy": str(song["energy"]),
                "tempo_bpm": str(song["tempo_bpm"]),
            },
        ),
        distance,
    )


def test_apply_refinement_action_clamps_numeric_values():
    profile = {
        "favorite_genre": "pop",
        "favorite_mood": "focused",
        "target_energy": 0.95,
        "target_tempo": 215.0,
        "target_acousticness": None,
        "target_valence": 0.95,
    }

    updated, label = apply_refinement_action(profile, "more_upbeat")

    assert label == "More upbeat"
    assert updated["favorite_mood"] == "happy"
    assert updated["target_energy"] == 1.0
    assert updated["target_tempo"] == 220.0
    assert updated["target_valence"] == 1.0


def test_score_song_uses_optional_refinement_dimensions():
    prefs = {
        "favorite_genre": "lofi",
        "favorite_mood": "chill",
        "target_energy": 0.3,
        "target_tempo": 72.0,
        "target_acousticness": 0.8,
        "target_valence": 0.4,
    }
    song = _song(1, "Library Rain", "lofi", "chill", 0.3, 72.0, 0.8, 0.4)

    score, reasons = score_song(prefs, song)

    assert abs(score - 5.5) < 1e-9
    assert any("acousticness similarity" in reason for reason in reasons)
    assert any("valence similarity" in reason for reason in reasons)


def test_refine_recommendations_returns_structured_result_and_excludes_anchor():
    anchor = _song(1, "Anchor Track", "rock", "intense", 0.88, 150.0, 0.12, 0.62)
    alt_1 = _song(2, "Breaker", "rock", "intense", 0.9, 152.0, 0.10, 0.65)
    alt_2 = _song(3, "Night Drive", "rock", "energetic", 0.84, 148.0, 0.18, 0.60)
    alt_3 = _song(4, "Echo Line", "electronic", "intense", 0.87, 151.0, 0.08, 0.58)
    songs = [anchor, alt_1, alt_2, alt_3]

    vectorstore = FakeVectorStore(
        [
            _doc(anchor, 0.02),
            _doc(alt_1, 0.04),
            _doc(alt_2, 0.08),
            _doc(alt_3, 0.10),
        ]
    )

    result = refine_recommendations(
        user_query="Need something like this",
        active_profile=profile_from_song(anchor),
        songs=songs,
        vectorstore=vectorstore,
        k=3,
        refinement_history=["More acoustic"],
        exclude_titles={anchor["title"]},
        plan_summary='Exploring tracks similar to "Anchor Track".',
    )

    assert result["mode"] == "refined"
    assert result["plan"]["summary"] == 'Exploring tracks similar to "Anchor Track".'
    assert result["refinement_history"] == ["More acoustic"]
    assert isinstance(result["explanations"], list)
    assert len(result["explanations"]) == len(result["scored"])
    assert all(song["title"] != "Anchor Track" for song, _, _ in result["scored"])
    assert all(song["title"] != "Anchor Track" for song in result["retrieved"])
