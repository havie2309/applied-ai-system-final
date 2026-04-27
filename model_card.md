# 🎧 Model Card — VibeFinder 2.0

## 1. Model Name
**VibeFinder 2.0** — Applied AI Music Recommendation System

---

## 2. Base Project
**VibeFinder 1.0 (Module 3)**
A content-based music recommender that scored an 18-song catalog using a weighted formula: genre match (+2.0), mood match (+1.5), energy similarity (+1.0), and tempo similarity (+0.5). It was limited to hardcoded user profiles and exact string matching with no AI integration.

---

## 3. Intended Use
VibeFinder 2.0 recommends songs from a 55-song catalog based on natural language user requests. It is designed as a classroom portfolio project demonstrating RAG, agentic workflows, few-shot specialization, and reliability testing. It is not intended for production use.

---

## 4. How It Works

### Pipeline Overview
1. **PLAN** — GPT-4o-mini parses the user's natural language query into structured preferences (genre, mood, energy, tempo)
2. **RETRIEVE** — OpenAI `text-embedding-ada-002` embeds the query; ChromaDB finds the top-10 semantically similar songs
3. **SCORE** — The original Module 3 weighted scorer re-ranks the retrieved pool
4. **VALIDATE** — Confidence is computed from match ratios and average scores; guardrails check for out-of-range inputs
5. **EXPLAIN** — GPT-4o-mini generates music critic explanations using the VibeBot few-shot persona

### AI Features
| Feature | Details |
|---|---|
| RAG | `text-embedding-ada-002` embeddings + ChromaDB local vector store |
| Agentic Workflow | 5 observable steps with JSON planning and fallback logic |
| Few-Shot Specialization | VibeBot persona with 2 examples constraining musical vocabulary and explanation style |
| Reliability System | Confidence scoring (0–1), input validation guardrails, structured JSON logging |

---

## 5. Data
The catalog contains 55 songs across 12 genres: pop, lofi, rock, jazz, metal, classical, R&B, electronic, indie, folk, ambient, and synthwave. Each song has 10 attributes. The original 18 songs were manually curated for Module 3; 37 additional songs were AI-generated to expand coverage. The dataset skews toward Western popular music styles.

---

## 6. Strengths
- Handles natural language input — users don't need to know genre names
- RAG retrieval captures semantic similarity that exact matching misses
- Every recommendation includes a traceable score and explanation
- Guardrails prevent silent failures for unrecognized inputs
- Full interaction logging enables audit and review
- Observable agent steps make the system interpretable

---

## 7. Limitations and Bias
- **Catalog size:** 55 songs is still small — niche profiles return repeated results
- **Genre dominance:** The weighted scorer still over-prioritizes genre (+2.0 weight)
- **Mood vocabulary:** Exact string matching for mood means "chill" and "relaxed" score differently even though they feel similar
- **Western bias:** The catalog overwhelmingly represents Western popular music
- **Embedding cost:** Every new session re-queries the OpenAI API for embeddings unless ChromaDB cache is used
- **LLM planning errors:** The PLAN step occasionally misclassifies genre for ambiguous queries

---

## 8. Evaluation / Testing Summary

The test harness ran 8 test cases with 6 checks each (48 total checks):

- **Pass rate:** ~95% of checks passed
- **Average confidence:** ~0.65 across all test cases
- **Guardrails:** Correctly triggered warnings for unrecognized genres (TC06) and out-of-range energy values (TC08)
- **Failure pattern:** Confidence drops below threshold when genre is unknown — the system falls back to mood and energy matching, which produces lower but still reasonable results
- **Surprise finding:** RAG retrieval was consistently more accurate than expected for cross-genre queries — semantic embeddings captured "study music" as a concept regardless of genre label

---

## 9. Ethics and Responsible AI

### Limitations
The system cannot discover surprising cross-genre recommendations the way collaborative filtering can. It is bounded by its 55-song catalog and the genres/moods represented in it.

### Potential Misuse
The system is low-risk — it recommends music. However, if scaled to include user behavior data, it could reinforce listening bubbles by only recommending familiar genres.

### Prevention
- The guardrail system flags unrecognized inputs rather than silently failing
- Confidence scores are always shown to the user so they can calibrate trust
- The system is explicitly labeled as a prototype, not a production recommender

---

## 10. AI Collaboration Reflection

### Helpful AI suggestion
When designing the RAG pipeline, Claude suggested converting song attributes into natural language descriptions before embedding (e.g., *"Storm Runner by Voltline is a rock song with an intense mood. Energy level: 0.91..."*) rather than embedding raw CSV rows. This significantly improved retrieval quality because the embeddings captured semantic meaning instead of treating attribute names as tokens.

### Flawed AI suggestion
Claude initially suggested using `langchain.schema` for imports, which caused `ModuleNotFoundError` because newer versions of LangChain moved those classes to `langchain_core.messages` and `langchain_core.documents`. This required debugging and manual correction — a good reminder that AI-generated code must always be verified against the actual installed library versions.

---

## 11. Personal Reflection

Building VibeFinder 2.0 taught me that the difference between a prototype and a real AI system is infrastructure. The core recommendation logic from Module 3 still works — but wrapping it in RAG, an agentic pipeline, guardrails, and a test harness transforms it from a script into a system.

The most valuable lesson was that RAG and scoring work better together than either does alone. RAG narrows the search space semantically; scoring re-ranks with precision. Neither approach is sufficient on its own for a small catalog.

As an AI engineer, this project showed me that reliability is a design choice, not an accident. Every guardrail, log entry, and test case is a deliberate decision to make the system trustworthy.
