from __future__ import annotations

import sys
from dataclasses import dataclass

from memory_agent import SkeletonLangGraphAgent

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


@dataclass
class Scenario:
    name: str
    turns: list[str]
    expected_substring: str
    category: str


SCENARIOS = [
    Scenario(
        name="Recall user name after filler turns",
        category="profile recall",
        turns=[
            "Tôi tên là Linh.",
            "Hôm nay tôi đang học về memory stack.",
            "Nhắc tôi uống nước.",
            "Tôi thích code Python.",
            "Giải thích giúp tôi semantic memory.",
            "Tên tôi là gì?",
        ],
        expected_substring="Linh",
    ),
    Scenario(
        name="Allergy conflict update",
        category="conflict update",
        turns=[
            "Tôi dị ứng sữa bò.",
            "À nhầm, tôi dị ứng đậu nành chứ không phải sữa bò.",
            "Tôi dị ứng gì?",
        ],
        expected_substring="đậu nành",
    ),
    Scenario(
        name="Favorite language recall",
        category="profile recall",
        turns=[
            "Tôi thích ngôn ngữ TypeScript.",
            "Cho tôi ví dụ về state router.",
            "Thích ngôn ngữ nào nhỉ?",
        ],
        expected_substring="typescript",
    ),
    Scenario(
        name="Current learning goal recall",
        category="profile recall",
        turns=[
            "Tôi đang học memory systems for agents.",
            "Nói thêm về prompt injection.",
            "Tôi đang học gì?",
        ],
        expected_substring="memory systems for agents",
    ),
    Scenario(
        name="Episodic recall for completed task",
        category="episodic recall",
        turns=[
            "Đã hoàn thành lab benchmark 10 hội thoại.",
            "Task nào đã hoàn thành gần đây?",
        ],
        expected_substring="benchmark 10 hội thoại",
    ),
    Scenario(
        name="Episodic lesson recall",
        category="episodic recall",
        turns=[
            "Bài học hôm nay là dùng docker service name thay vì localhost.",
            "Bài học trước là gì?",
        ],
        expected_substring="docker service name",
    ),
    Scenario(
        name="Semantic retrieval about prompt window",
        category="semantic retrieval",
        turns=[
            "Prompt window gồm những gì trong memory system?",
        ],
        expected_substring="Prompt window",
    ),
    Scenario(
        name="Semantic retrieval about vector store",
        category="semantic retrieval",
        turns=[
            "Vector store dùng để làm gì trong semantic memory?",
        ],
        expected_substring="Vector store",
    ),
    Scenario(
        name="Semantic retrieval about privacy and TTL",
        category="semantic retrieval",
        turns=[
            "TTL và consent quan trọng thế nào với memory?",
        ],
        expected_substring="TTL",
    ),
    Scenario(
        name="Trim budget still keeps salient facts",
        category="trim/token budget",
        turns=[
            "Tôi tên là Mai.",
            "Tôi thích ngôn ngữ Python.",
            "Hãy ghi nhớ giúp tôi điều này.",
            "Đây là một câu filler rất dài để làm phình recent conversation nhưng không nên xóa profile fact quan trọng ngay cả khi token budget bị giới hạn.",
            "Đây là câu filler thứ hai để kiểm tra trim hoạt động.",
            "Tên tôi là gì?",
        ],
        expected_substring="Mai",
    ),
]


def run_agent(use_memory: bool, turns: list[str], memory_budget: int = 45) -> tuple[str, str]:
    agent = SkeletonLangGraphAgent(use_memory=use_memory, memory_budget=memory_budget)
    answer = ""
    for turn in turns:
        answer = agent.respond(turn)
    prompt = agent.last_state["prompt"] if agent.last_state else ""
    return answer, prompt


def main() -> None:
    rows: list[dict[str, str]] = []
    for index, scenario in enumerate(SCENARIOS, start=1):
        no_memory_answer, _ = run_agent(use_memory=False, turns=scenario.turns)
        with_memory_answer, prompt = run_agent(use_memory=True, turns=scenario.turns)
        passed = scenario.expected_substring.lower() in with_memory_answer.lower()
        rows.append(
            {
                "index": str(index),
                "category": scenario.category,
                "scenario": scenario.name,
                "no_memory": no_memory_answer,
                "with_memory": with_memory_answer,
                "pass": "Pass" if passed else "Fail",
                "prompt_words": str(len(prompt.split())),
            }
        )

    print("| # | Category | Scenario | No-memory result | With-memory result | Prompt words | Pass? |")
    print("|---|---|---|---|---|---:|---|")
    for row in rows:
        print(
            f"| {row['index']} | {row['category']} | {row['scenario']} | "
            f"{row['no_memory']} | {row['with_memory']} | {row['prompt_words']} | {row['pass']} |"
        )


if __name__ == "__main__":
    main()
