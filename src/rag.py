"""
rag.py - Retrieval-Augmented Generation system for VibeFinder 2.0.
Embeds the song catalog using OpenAI embeddings and retrieves
semantically similar songs before the AI generates recommendations.
"""

import os
import logging
from typing import List, Dict
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document

load_dotenv()
logger = logging.getLogger(__name__)

CHROMA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "chroma_db")


def build_song_documents(songs: List[Dict]) -> List[Document]:
    """
    Converts song dictionaries into LangChain Documents.
    Each document's page_content is a natural language description
    so the embedding captures semantic meaning, not just field names.
    """
    docs = []
    for song in songs:
        content = (
            f"{song['title']} by {song['artist']} is a {song['genre']} song "
            f"with a {song['mood']} mood. "
            f"Energy level: {song['energy']}, Tempo: {song['tempo_bpm']} BPM, "
            f"Valence: {song['valence']}, Danceability: {song['danceability']}, "
            f"Acousticness: {song['acousticness']}."
        )
        metadata = {
            "id": str(song["id"]),
            "title": song["title"],
            "artist": song["artist"],
            "genre": song["genre"],
            "mood": song["mood"],
            "energy": str(song["energy"]),
            "tempo_bpm": str(song["tempo_bpm"]),
        }
        docs.append(Document(page_content=content, metadata=metadata))
    logger.info(f"Built {len(docs)} song documents for embedding")
    return docs


def build_vectorstore(songs: List[Dict], force_rebuild: bool = False) -> Chroma:
    """
    Builds or loads a ChromaDB vector store from the song catalog.
    If the DB already exists, it loads it instead of re-embedding.
    Set force_rebuild=True to re-embed everything from scratch.
    """
    embeddings = OpenAIEmbeddings(
        model="text-embedding-ada-002",
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    if os.path.exists(CHROMA_DIR) and not force_rebuild:
        logger.info("Loading existing ChromaDB vector store...")
        vectorstore = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings
        )
        logger.info("Vector store loaded successfully")
        return vectorstore

    logger.info("Building new ChromaDB vector store...")
    docs = build_song_documents(songs)
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )
    logger.info(f"Vector store built and persisted to {CHROMA_DIR}")
    return vectorstore


def retrieve_similar_songs(
    query: str,
    vectorstore: Chroma,
    k: int = 5
) -> List[Dict]:
    """
    Retrieves the top-k most semantically similar songs for a given query.
    The query can be a natural language description like:
    'upbeat happy pop song for working out'
    """
    logger.info(f"RAG query: '{query}' (top {k})")
    results = vectorstore.similarity_search_with_score(query, k=k)

    retrieved = []
    for doc, score in results:
        similarity = round(1 - score, 4)  # convert distance to similarity
        retrieved.append({
            "title": doc.metadata["title"],
            "artist": doc.metadata["artist"],
            "genre": doc.metadata["genre"],
            "mood": doc.metadata["mood"],
            "energy": float(doc.metadata["energy"]),
            "tempo_bpm": float(doc.metadata["tempo_bpm"]),
            "similarity_score": similarity,
            "description": doc.page_content,
        })
        logger.info(f"  Retrieved: {doc.metadata['title']} (similarity: {similarity:.4f})")

    return retrieved


def format_retrieved_context(retrieved_songs: List[Dict]) -> str:
    """
    Formats retrieved songs into a context string to inject into the AI prompt.
    This is the 'augmented' part of RAG — the AI sees this before generating.
    """
    if not retrieved_songs:
        return "No similar songs found in the catalog."

    lines = ["Here are the most relevant songs from the catalog:\n"]
    for i, song in enumerate(retrieved_songs, 1):
        lines.append(
            f"{i}. \"{song['title']}\" by {song['artist']} "
            f"({song['genre']}, {song['mood']}, energy={song['energy']}, "
            f"tempo={song['tempo_bpm']}bpm) — similarity: {song['similarity_score']:.2f}"
        )
    return "\n".join(lines)
