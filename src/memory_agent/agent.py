from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

from memory_agent.graph import SimpleStateGraph
from memory_agent.memory_backends import (
    MemorySuite,
    estimate_tokens,
    extract_profile_updates,
    maybe_build_episode,
)


class MemoryState(TypedDict):
    user_id: str
    messages: list[dict[str, str]]
    user_profile: dict[str, str]
    episodes: list[dict[str, Any]]
    semantic_hits: list[str]
    memory_budget: int
    intents: list[str]
    response: str


@dataclass
class AgentConfig:
    memory_budget: int = 500


class MemoryRouter:
    def route(self, query: str) -> list[str]:
        lowered = query.lower()
        routes: list[str] = []
        if any(x in lowered for x in ["ten", "tên", "name", "thich", "thích", "prefer", "di ung", "dị ứng", "allergy"]):
            routes.append("profile")
        if any(x in lowered for x in ["lan truoc", "lần trước", "before", "episode", "debug", "worked", "da fix", "đã fix"]):
            routes.append("episodic")
        if any(x in lowered for x in ["faq", "docker", "redis", "token budget", "semantic", "dinh nghia", "định nghĩa", "what is"]):
            routes.append("semantic")
        if not routes:
            routes = ["profile", "episodic", "semantic"]
        return routes


class MultiMemoryAgent:
    def __init__(self, memory: MemorySuite, use_memory: bool = True, config: AgentConfig | None = None) -> None:
        self.memory = memory
        self.use_memory = use_memory
        self.router = MemoryRouter()
        self.config = config or AgentConfig()
        self.graph = self._build_graph()

    def _build_graph(self) -> SimpleStateGraph:
        graph = SimpleStateGraph()
        graph.add_node("load_memory", self._load_memory)
        graph.add_node("generate_response", self._generate_response)
        graph.add_node("save_memory", self._save_memory)
        graph.add_edge("load_memory", "generate_response")
        graph.add_edge("generate_response", "save_memory")
        return graph

    def reset_short_term(self) -> None:
        self.memory.short_term.clear()

    def ask(self, user_id: str, user_text: str) -> tuple[str, MemoryState]:
        self.memory.short_term.add("user", user_text)
        state: MemoryState = {
            "user_id": user_id,
            "messages": self.memory.short_term.recent(),
            "user_profile": {},
            "episodes": [],
            "semantic_hits": [],
            "memory_budget": self.config.memory_budget,
            "intents": [],
            "response": "",
        }
        final_state = self.graph.run(start="load_memory", state=state)
        self.memory.short_term.add("assistant", final_state["response"])
        return final_state["response"], final_state

    def _load_memory(self, state: dict[str, Any]) -> dict[str, Any]:
        query = state["messages"][-1]["content"]
        intents = self.router.route(query)
        state["intents"] = intents

        if not self.use_memory:
            return state

        user_id = state["user_id"]
        if "profile" in intents:
            state["user_profile"] = self.memory.long_term.load_profile(user_id)
        if "episodic" in intents:
            state["episodes"] = self.memory.episodic.retrieve(user_id=user_id, query=query, k=3)
        if "semantic" in intents:
            state["semantic_hits"] = self.memory.semantic.query(query, k=3)

        state["messages"] = self._trim_context(
            messages=state["messages"],
            profile=state["user_profile"],
            episodes=state["episodes"],
            semantic_hits=state["semantic_hits"],
            budget=state["memory_budget"],
        )
        return state

    def _trim_context(
        self,
        messages: list[dict[str, str]],
        profile: dict[str, str],
        episodes: list[dict[str, Any]],
        semantic_hits: list[str],
        budget: int,
    ) -> list[dict[str, str]]:
        # 4-level token budget split from lecture guidance.
        short_term_budget = int(budget * 0.10)
        long_term_budget = int(budget * 0.04)
        episodic_budget = int(budget * 0.03)
        semantic_budget = int(budget * 0.03)

        trimmed = []
        running = 0

        for msg in reversed(messages):
            cost = estimate_tokens(msg["content"])
            if running + cost > short_term_budget:
                break
            trimmed.append(msg)
            running += cost
        trimmed.reverse()

        profile_copy = dict(profile)
        profile_blob = "; ".join([f"{k}: {v}" for k, v in profile_copy.items()])
        if estimate_tokens(profile_blob) > long_term_budget:
            while profile_copy and estimate_tokens("; ".join([f"{k}: {v}" for k, v in profile_copy.items()])) > long_term_budget:
                profile_copy.pop(next(iter(profile_copy)))

        profile.clear()
        profile.update(profile_copy)

        if episodes:
            episode_tokens = sum(estimate_tokens(json.dumps(ep, ensure_ascii=False)) for ep in episodes)
            if episode_tokens > episodic_budget:
                episodes[:] = episodes[:1]

        if semantic_hits:
            semantic_tokens = sum(estimate_tokens(item) for item in semantic_hits)
            if semantic_tokens > semantic_budget:
                semantic_hits[:] = semantic_hits[:1]

        return trimmed

    def _generate_response(self, state: dict[str, Any]) -> dict[str, Any]:
        query = state["messages"][-1]["content"].lower()
        profile = state.get("user_profile", {})
        episodes = state.get("episodes", [])
        semantic_hits = state.get("semantic_hits", [])

        if any(x in query for x in ["ten", "tên", "name"]):
            answer = f"Ten ban la {profile.get('name', 'minh chua biet')}"
        elif any(x in query for x in ["di ung", "dị ứng", "allergy"]):
            answer = f"Thong tin di ung hien tai: {profile.get('allergy', 'chua co')}"
        elif any(x in query for x in ["thich", "thích", "prefer"]):
            answer = f"So thich da luu: {profile.get('likes', 'chua co')}"
        elif any(x in query for x in ["lan truoc", "lần trước", "before", "debug"]):
            if episodes:
                answer = f"Lan truoc ket qua la: {episodes[0].get('outcome', 'chua ro')}"
            else:
                answer = "Minh khong co episodic memory lien quan."
        elif any(x in query for x in ["docker", "redis", "token budget", "faq", "semantic", "định nghĩa", "what is"]):
            if semantic_hits:
                answer = f"Thong tin semantic: {semantic_hits[0]}"
            else:
                answer = "Minh khong tim thay semantic knowledge phu hop."
        else:
            answer = "Minh da xu ly yeu cau cua ban."

        state["response"] = answer
        return state

    def _save_memory(self, state: dict[str, Any]) -> dict[str, Any]:
        if not self.use_memory:
            return state

        user_id = state["user_id"]
        latest_user_text = state["messages"][-1]["content"]
        updates = extract_profile_updates(latest_user_text)
        self.memory.long_term.update_profile(user_id, updates)

        episode = maybe_build_episode(user_id=user_id, user_text=latest_user_text, assistant_text=state["response"])
        if episode:
            self.memory.episodic.append_episode(episode)

        return state


def load_seed_semantic(memory: MemorySuite, knowledge_file: str) -> None:
    docs = json.loads(Path(knowledge_file).read_text(encoding="utf-8"))
    memory.semantic.upsert_documents(docs, source="lab_seed")
