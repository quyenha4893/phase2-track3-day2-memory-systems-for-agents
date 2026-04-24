from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypedDict


DATA_DIR = Path(__file__).parent / "data"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


class Message(TypedDict):
    role: str
    content: str


class Episode(TypedDict):
    summary: str
    outcome: str


class MemoryState(TypedDict):
    messages: list[Message]
    user_profile: dict[str, str]
    episodes: list[Episode]
    semantic_hits: list[str]
    memory_budget: int
    recent_conversation: list[Message]
    prompt: str


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _word_budget(text: str) -> int:
    return len(text.split())


def _trim_messages(messages: list[Message], memory_budget: int) -> list[Message]:
    kept: list[Message] = []
    running = 0
    for message in reversed(messages):
        cost = _word_budget(message["content"])
        if kept and running + cost > memory_budget:
            break
        kept.append(message)
        running += cost
    return list(reversed(kept))


@dataclass
class ShortTermMemory:
    max_turns: int = 12
    messages: list[Message] = field(default_factory=list)

    def append(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})
        self.messages = self.messages[-self.max_turns :]

    def retrieve(self, budget: int) -> list[Message]:
        return _trim_messages(self.messages, budget)


@dataclass
class LongTermProfileMemory:
    facts: dict[str, str] = field(default_factory=dict)

    def update_from_message(self, text: str) -> None:
        if "?" in text:
            return

        lowered = _normalize(text)
        patterns = {
            "name": [
                r"(?:tôi tên là|mình tên là|my name is)\s+([^.,!?]+)",
            ],
            "allergy": [
                r"(?:tôi dị ứng|mình dị ứng|i am allergic to)\s+([^.,!?]+)",
                r"(?:dị ứng với)\s+([^.,!?]+)",
            ],
            "favorite_language": [
                r"(?:tôi thích ngôn ngữ|mình thích ngôn ngữ|ngôn ngữ tôi thích là)\s+([^.,!?]+)",
                r"(?:i like programming language)\s+([^.,!?]+)",
            ],
            "learning_goal": [
                r"(?:tôi đang học|mình đang học)\s+([^.,!?]+)",
            ],
            "city": [
                r"(?:tôi sống ở|mình sống ở)\s+([^.,!?]+)",
            ],
        }

        for key, regexes in patterns.items():
            for pattern in regexes:
                match = re.search(pattern, lowered, re.IGNORECASE)
                if match:
                    self.facts[key] = match.group(1).strip(" .")

        # Conflict correction takes priority over older facts.
        correction_patterns = {
            "allergy": [
                r"(?:à nhầm|ý tôi là|actually)\s*,?\s*(?:tôi dị ứng|mình dị ứng)?\s*([^.,!?]+?)(?:\s+chứ không phải\s+[^.,!?]+)?$",
                r"(?:dị ứng)\s+([^.,!?]+)\s+(?:chứ không phải|not)\s+([^.,!?]+)",
            ]
        }

        for pattern in correction_patterns["allergy"]:
            match = re.search(pattern, lowered, re.IGNORECASE)
            if match:
                self.facts["allergy"] = match.group(1).strip(" .")

    def retrieve(self) -> dict[str, str]:
        return dict(self.facts)


@dataclass
class EpisodicMemory:
    episodes: list[Episode] = field(default_factory=list)

    def maybe_save(self, user_text: str, assistant_text: str) -> None:
        lowered = _normalize(user_text)
        task_match = re.search(
            r"(?:đã hoàn thành|xong|done with|finished)\s+([^.,!?]+)",
            lowered,
            re.IGNORECASE,
        )
        if task_match:
            task = task_match.group(1).strip(" .")
            self.episodes.append(
                {
                    "summary": f"User completed task: {task}",
                    "outcome": assistant_text,
                }
            )
            return

        lesson_match = re.search(
            r"(?:bài học hôm nay là|lesson learned:)\s+([^.!?]+)",
            lowered,
            re.IGNORECASE,
        )
        if lesson_match:
            lesson = lesson_match.group(1).strip(" .")
            self.episodes.append(
                {
                    "summary": f"User shared a lesson learned: {lesson}",
                    "outcome": "Stored as an episodic memory for future recall.",
                }
            )

    def retrieve(self, query: str, limit: int = 3) -> list[Episode]:
        if not self.episodes:
            return []
        terms = set(_normalize(query).split())
        scored: list[tuple[int, Episode]] = []
        for episode in self.episodes:
            haystack = _normalize(f"{episode['summary']} {episode['outcome']}")
            score = sum(1 for term in terms if term in haystack)
            scored.append((score, episode))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [episode for score, episode in scored[:limit] if score > 0] or self.episodes[-limit:]


@dataclass
class SemanticMemory:
    documents: list[dict[str, str]]

    @classmethod
    def from_json(cls, path: Path) -> "SemanticMemory":
        return cls(documents=json.loads(path.read_text(encoding="utf-8")))

    def retrieve(self, query: str, limit: int = 3) -> list[str]:
        terms = {
            token
            for token in re.findall(r"[a-zA-Z0-9à-ỹÀ-Ỹ_-]+", _normalize(query))
            if len(token) > 1
        }
        scored: list[tuple[int, str]] = []
        for doc in self.documents:
            haystack = _normalize(f"{doc['title']} {doc['content']}")
            score = sum(1 for term in terms if term in haystack)
            if score > 0:
                scored.append((score, f"{doc['title']}: {doc['content']}"))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [text for _, text in scored[:limit]]


def retrieve_memory(
    state: MemoryState,
    short_term: ShortTermMemory,
    profile: LongTermProfileMemory,
    episodic: EpisodicMemory,
    semantic: SemanticMemory,
    query: str,
) -> MemoryState:
    state["user_profile"] = profile.retrieve()
    state["episodes"] = episodic.retrieve(query)
    state["semantic_hits"] = semantic.retrieve(query)
    state["recent_conversation"] = short_term.retrieve(state["memory_budget"])
    return state


def build_prompt(state: MemoryState) -> str:
    profile_lines = (
        "\n".join(f"- {key}: {value}" for key, value in state["user_profile"].items())
        or "- No long-term profile facts yet."
    )
    episodic_lines = (
        "\n".join(f"- {item['summary']} -> {item['outcome']}" for item in state["episodes"])
        or "- No relevant episodes."
    )
    semantic_lines = (
        "\n".join(f"- {item}" for item in state["semantic_hits"])
        or "- No semantic hits."
    )
    recent_lines = (
        "\n".join(f"- {item['role']}: {item['content']}" for item in state["recent_conversation"])
        or "- No recent conversation."
    )

    return (
        "SYSTEM: Use memory carefully, prefer the latest corrected fact, and answer briefly.\n\n"
        "PROFILE MEMORY\n"
        f"{profile_lines}\n\n"
        "EPISODIC MEMORY\n"
        f"{episodic_lines}\n\n"
        "SEMANTIC MEMORY\n"
        f"{semantic_lines}\n\n"
        "RECENT CONVERSATION\n"
        f"{recent_lines}\n"
    )


class SkeletonLangGraphAgent:
    def __init__(self, use_memory: bool = True, memory_budget: int = 90) -> None:
        self.use_memory = use_memory
        self.memory_budget = memory_budget
        self.short_term = ShortTermMemory()
        self.profile = LongTermProfileMemory()
        self.episodic = EpisodicMemory()
        self.semantic = SemanticMemory.from_json(DATA_DIR / "semantic_docs.json")
        self.last_state: MemoryState | None = None

    def route(self, user_text: str) -> MemoryState:
        state: MemoryState = {
            "messages": list(self.short_term.messages),
            "user_profile": {},
            "episodes": [],
            "semantic_hits": [],
            "memory_budget": self.memory_budget,
            "recent_conversation": [],
            "prompt": "",
        }
        if self.use_memory:
            state = retrieve_memory(
                state,
                self.short_term,
                self.profile,
                self.episodic,
                self.semantic,
                user_text,
            )
        else:
            state["recent_conversation"] = self.short_term.retrieve(self.memory_budget)
        state["prompt"] = build_prompt(state)
        self.last_state = state
        return state

    def respond(self, user_text: str) -> str:
        self.short_term.append("user", user_text)
        if self.use_memory:
            self.profile.update_from_message(user_text)

        state = self.route(user_text)
        answer = self._generate_answer(user_text, state)
        self.short_term.append("assistant", answer)

        if self.use_memory:
            self.episodic.maybe_save(user_text, answer)

        return answer

    def _generate_answer(self, user_text: str, state: MemoryState) -> str:
        query = _normalize(user_text)
        profile = state["user_profile"]

        if "tên tôi" in query or "my name" in query:
            return f"Tên của bạn là {profile['name']}." if "name" in profile else "Mình chưa biết tên của bạn."

        if "dị ứng gì" in query or "allergy" in query:
            return (
                f"Bạn đang dị ứng {profile['allergy']}."
                if "allergy" in profile
                else "Mình chưa có thông tin dị ứng của bạn."
            )

        if "thích ngôn ngữ nào" in query or "favorite language" in query:
            return (
                f"Bạn thích {profile['favorite_language']}."
                if "favorite_language" in profile
                else "Mình chưa lưu sở thích ngôn ngữ của bạn."
            )

        if "đang học gì" in query:
            return (
                f"Bạn đang học {profile['learning_goal']}."
                if "learning_goal" in profile
                else "Mình chưa biết bạn đang học gì."
            )

        if "task nào đã hoàn thành" in query or "đã hoàn thành task gì" in query:
            if state["episodes"]:
                return state["episodes"][-1]["summary"].replace("User completed task: ", "Task gần nhất bạn hoàn thành là ")
            return "Mình chưa có episodic memory về task đã hoàn thành."

        if "bài học trước" in query or "lesson learned" in query:
            for episode in reversed(state["episodes"]):
                if "lesson learned" in episode["summary"]:
                    return episode["summary"].replace("User shared a lesson learned: ", "Bài học trước đó là ")
            return "Mình chưa lưu bài học trước đó."

        if state["semantic_hits"] and (
            "docker" in query
            or "semantic" in query
            or "faq" in query
            or "prompt window" in query
            or "vector" in query
            or "ttl" in query
        ):
            return f"Theo semantic memory: {state['semantic_hits'][0]}"

        if "prompt" in query and "bao gồm gì" in query:
            return "Prompt đang được inject từ profile, episodic, semantic và recent conversation."

        return "Mình đã nhận thông tin. Nếu muốn, bạn có thể hỏi lại để kiểm tra agent còn nhớ hay không."


def demo() -> None:
    agent = SkeletonLangGraphAgent(use_memory=True)
    turns = [
        "Tôi tên là Linh.",
        "Tôi dị ứng sữa bò.",
        "À nhầm, tôi dị ứng đậu nành chứ không phải sữa bò.",
        "Tôi thích ngôn ngữ Python.",
        "Tôi đang học memory systems for agents.",
        "Đã hoàn thành lab benchmark 10 hội thoại.",
        "Tên tôi là gì?",
        "Tôi dị ứng gì?",
        "Tôi thích ngôn ngữ nào?",
        "Task nào đã hoàn thành gần đây?",
        "Prompt đang bao gồm gì?",
    ]
    for turn in turns:
        print(f"USER: {turn}")
        print(f"ASSISTANT: {agent.respond(turn)}")
        print()


if __name__ == "__main__":
    demo()
