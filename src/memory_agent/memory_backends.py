from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import chromadb
import redis


def estimate_tokens(text: str) -> int:
    return max(1, len(text.split()))


class ConversationBufferMemory:
    """Short-term memory with sliding-window trim behavior."""

    def __init__(self, max_tokens: int = 120) -> None:
        self.max_tokens = max_tokens
        self.messages: list[dict[str, str]] = []

    def add(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})
        self._trim()

    def recent(self) -> list[dict[str, str]]:
        return list(self.messages)

    def clear(self) -> None:
        self.messages.clear()

    def token_count(self) -> int:
        return sum(estimate_tokens(m["content"]) for m in self.messages)

    def _trim(self) -> None:
        while self.messages and self.token_count() > self.max_tokens:
            self.messages.pop(0)


class RedisLongTermMemory:
    """Long-term profile memory in Redis hash."""

    def __init__(self, redis_url: str, key_prefix: str = "lab17") -> None:
        self.client = redis.Redis.from_url(redis_url, decode_responses=True)
        self.key_prefix = key_prefix

    def ping(self) -> None:
        self.client.ping()

    def _profile_key(self, user_id: str) -> str:
        return f"{self.key_prefix}:user:{user_id}:profile"

    def load_profile(self, user_id: str) -> dict[str, str]:
        return {k: v for k, v in self.client.hgetall(self._profile_key(user_id)).items()}

    def update_profile(self, user_id: str, updates: dict[str, str]) -> None:
        if not updates:
            return
        self.client.hset(self._profile_key(user_id), mapping=updates)


class EpisodicJsonMemory:
    """Episodic memory persisted in a JSON log file."""

    def __init__(self, file_path: str) -> None:
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.file_path.exists():
            self.file_path.write_text("[]\n", encoding="utf-8")

    def append_episode(self, episode: dict[str, Any]) -> None:
        episodes = self._read_all()
        episodes.append(episode)
        self.file_path.write_text(json.dumps(episodes, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def retrieve(self, user_id: str, query: str, k: int = 3) -> list[dict[str, Any]]:
        query_terms = set(re.findall(r"\w+", query.lower()))
        scored: list[tuple[int, dict[str, Any]]] = []
        for episode in self._read_all():
            if episode.get("user_id") != user_id:
                continue
            blob = " ".join(
                [
                    str(episode.get("task", "")),
                    str(episode.get("trajectory", "")),
                    str(episode.get("outcome", "")),
                    str(episode.get("reflection", "")),
                ]
            ).lower()
            terms = set(re.findall(r"\w+", blob))
            score = len(query_terms.intersection(terms))
            if score > 0:
                scored.append((score, episode))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [item[1] for item in scored[:k]]

    def _read_all(self) -> list[dict[str, Any]]:
        return json.loads(self.file_path.read_text(encoding="utf-8"))


class ChromaSemanticMemory:
    """Semantic memory backed by Chroma persistent collection."""

    def __init__(self, persist_dir: str, collection_name: str = "lab17_knowledge") -> None:
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(collection_name)

    @staticmethod
    def _embed(text: str, dim: int = 64) -> list[float]:
        vec = [0.0] * dim
        words = re.findall(r"\w+", text.lower())
        if not words:
            return vec
        for word in words:
            digest = hashlib.sha256(word.encode("utf-8")).digest()
            for i in range(dim):
                vec[i] += digest[i % len(digest)] / 255.0
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0:
            return vec
        return [v / norm for v in vec]

    def upsert_documents(self, documents: list[str], source: str = "seed") -> None:
        ids = []
        embeddings = []
        metadatas = []
        for doc in documents:
            doc_id = hashlib.sha1(doc.encode("utf-8")).hexdigest()
            ids.append(doc_id)
            embeddings.append(self._embed(doc))
            metadatas.append({"source": source})
        self.collection.upsert(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)

    def query(self, text: str, k: int = 3) -> list[str]:
        # Hybrid retrieval: keyword overlap first, vector fallback.
        all_docs: list[str] = []
        raw = self.collection.get(include=["documents"])
        if raw and raw.get("documents"):
            all_docs = [doc for doc in raw["documents"] if doc]

        query_terms = set(re.findall(r"\w+", text.lower()))
        query_lower = text.lower()
        scored: list[tuple[int, str]] = []
        for doc in all_docs:
            doc_terms = set(re.findall(r"\w+", doc.lower()))
            score = len(query_terms.intersection(doc_terms))

            doc_lower = doc.lower()
            if "docker" in query_lower and "docker" in doc_lower:
                score += 5
            if "token budget" in query_lower and "token budget" in doc_lower:
                score += 5
            if "redis" in query_lower and "redis" in doc_lower:
                score += 5

            if score > 0:
                scored.append((score, doc))

        scored.sort(key=lambda item: item[0], reverse=True)
        ranked_docs = [item[1] for item in scored[:k]]
        if ranked_docs:
            return ranked_docs

        result = self.collection.query(query_embeddings=[self._embed(text)], n_results=k)
        docs = result.get("documents", [[]])
        return docs[0] if docs else []


@dataclass
class MemorySuite:
    short_term: ConversationBufferMemory
    long_term: RedisLongTermMemory
    episodic: EpisodicJsonMemory
    semantic: ChromaSemanticMemory

    @staticmethod
    def build(redis_url: str, episodic_path: str, chroma_dir: str, short_term_tokens: int = 120) -> "MemorySuite":
        return MemorySuite(
            short_term=ConversationBufferMemory(max_tokens=short_term_tokens),
            long_term=RedisLongTermMemory(redis_url=redis_url),
            episodic=EpisodicJsonMemory(file_path=episodic_path),
            semantic=ChromaSemanticMemory(persist_dir=chroma_dir),
        )


def extract_profile_updates(user_text: str) -> dict[str, str]:
    updates: dict[str, str] = {}
    lowered = user_text.lower()

    name_match = re.search(r"(?:ten toi la |tên tôi là |my name is )([a-zA-Z\s]+)", user_text, re.IGNORECASE)
    if name_match:
        updates["name"] = name_match.group(1).strip()

    allergy_correct = re.search(
        r"d[iị] [uư]ng\s+([\w\s]+?)\s+ch[uứ]\s+kh[oô]ng ph[aả]i\s+([\w\s]+)",
        lowered,
        re.IGNORECASE,
    )
    if allergy_correct:
        updates["allergy"] = allergy_correct.group(1).strip()
    else:
        allergy_match = re.search(r"(?:di ung|dị ứng|allergic to)\s+([\w\s]+)", lowered, re.IGNORECASE)
        if allergy_match:
            updates["allergy"] = allergy_match.group(1).strip()

    like_match = re.search(r"(?:toi thich|tôi thích|i like)\s+([\w\s]+)", lowered, re.IGNORECASE)
    if like_match:
        updates["likes"] = like_match.group(1).strip()

    style_match = re.search(r"(?:phong c[aá]ch|style)\s*[:=]?\s*([\w\s-]+)", lowered, re.IGNORECASE)
    if style_match:
        updates["style"] = style_match.group(1).strip()

    return updates


def maybe_build_episode(user_id: str, user_text: str, assistant_text: str) -> dict[str, Any] | None:
    lowered = user_text.lower()
    if not any(trigger in lowered for trigger in ["debug", "fix", "sua loi", "khac phuc", "worked", "da fix"]):
        return None
    return {
        "user_id": user_id,
        "task": user_text,
        "trajectory": "Asked assistant for troubleshooting guidance",
        "outcome": assistant_text,
        "reflection": "Prefer validated root-cause steps before code changes",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
