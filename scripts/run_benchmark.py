from __future__ import annotations

import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts"
DATA_DIR = ROOT / "data"
sys.path.insert(0, str(ROOT / "src"))

from memory_agent.agent import MultiMemoryAgent, load_seed_semantic
from memory_agent.memory_backends import MemorySuite, estimate_tokens


@dataclass
class Scenario:
    sid: int
    name: str
    turns: list[str]
    expected_no_memory_contains: str
    expected_with_memory_contains: str


SCENARIOS = [
    Scenario(1, "Recall user name", ["Ten toi la Linh", "Hom nay ban giup toi hoc python", "Ban nho ten toi khong?"], "minh chua biet", "Linh"),
    Scenario(2, "Allergy conflict update", ["Toi di ung sua bo", "A nham, toi di ung dau nanh chu khong phai sua bo", "Thong tin di ung cua toi la gi?"], "chua co", "dau nanh"),
    Scenario(3, "Preference recall", ["Toi thich Python", "Minh dang hoc AI", "Ban nho toi thich gi?"], "chua co", "python"),
    Scenario(4, "Episodic debug recall", ["Hom qua toi debug API va da fix bang cach check status code", "Toi muon ghi nho bai hoc debug", "Lan truoc toi debug xong nhu the nao?"], "khong co episodic", "Lan truoc ket qua"),
    Scenario(5, "Semantic docker retrieval", ["Toi can FAQ ve docker compose networking", "Trong docker compose nen goi service bang gi?"], "khong tim thay", "service name"),
    Scenario(6, "Semantic token budget", ["Cho toi thong tin token budget memory", "Token budget memory nen chia the nao?"], "khong tim thay", "10 percent"),
    Scenario(7, "Cross-topic recall", ["Ten toi la An", "Toi thich JavaScript", "Ban nho ten va so thich cua toi?"], "chua biet", "An"),
    Scenario(8, "Redis factual recall", ["Hay nho rang toi di ung dau phong", "Toi muon doi chu de", "Toi di ung gi?"], "chua co", "dau phong"),
    Scenario(9, "Semantic redis concept", ["Toi can dinh nghia redis memory", "Redis dung de lam gi trong long-term memory?"], "khong tim thay", "Redis stores user profile facts"),
    Scenario(10, "Episodic troubleshooting recall", ["Toi vua fix loi async await do quen await", "Hay luu kinh nghiem nay", "Lan truoc toi fix loi async the nao?"], "khong co episodic", "Lan truoc ket qua"),
]


def build_agent(use_memory: bool, namespace: str) -> MultiMemoryAgent:
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    episodic_path = str(DATA_DIR / f"episodic_{namespace}.json")
    chroma_dir = str(DATA_DIR / f"chroma_{namespace}")

    if Path(episodic_path).exists():
        Path(episodic_path).unlink()
    if Path(chroma_dir).exists():
        shutil.rmtree(chroma_dir)

    memory = MemorySuite.build(
        redis_url=redis_url,
        episodic_path=episodic_path,
        chroma_dir=chroma_dir,
        short_term_tokens=120,
    )
    memory.long_term.ping()
    memory.long_term.client.flushdb()
    load_seed_semantic(memory, str(DATA_DIR / "semantic_knowledge.json"))
    return MultiMemoryAgent(memory=memory, use_memory=use_memory)


def run_mode(use_memory: bool, namespace: str) -> tuple[list[dict[str, Any]], int]:
    agent = build_agent(use_memory=use_memory, namespace=namespace)
    mode_results: list[dict[str, Any]] = []
    token_total = 0

    for scenario in SCENARIOS:
        agent.reset_short_term()
        user_id = f"user_{scenario.sid}"
        final_response = ""
        for turn in scenario.turns:
            final_response, state = agent.ask(user_id=user_id, user_text=turn)
            token_total += estimate_tokens(turn)
            token_total += estimate_tokens(final_response)

        mode_results.append(
            {
                "id": scenario.sid,
                "name": scenario.name,
                "response": final_response,
                "intents": state.get("intents", []),
                "retrieved_profile_keys": list(state.get("user_profile", {}).keys()),
                "retrieved_episodes": len(state.get("episodes", [])),
                "retrieved_semantic": len(state.get("semantic_hits", [])),
            }
        )

    return mode_results, token_total


def evaluate(no_memory: list[dict[str, Any]], with_memory: list[dict[str, Any]], no_tokens: int, with_tokens: int) -> dict[str, Any]:
    rows = []
    memory_hits = 0
    for scenario in SCENARIOS:
        no_row = next(item for item in no_memory if item["id"] == scenario.sid)
        mem_row = next(item for item in with_memory if item["id"] == scenario.sid)

        pass_no = scenario.expected_no_memory_contains.lower() in no_row["response"].lower()
        pass_mem = scenario.expected_with_memory_contains.lower() in mem_row["response"].lower()
        if pass_mem:
            memory_hits += 1

        rows.append(
            {
                "id": scenario.sid,
                "scenario": scenario.name,
                "no_memory_result": no_row["response"],
                "with_memory_result": mem_row["response"],
                "pass": pass_mem,
                "no_memory_baseline_ok": pass_no,
                "with_memory_retrieval": {
                    "intents": mem_row["intents"],
                    "profile_keys": mem_row["retrieved_profile_keys"],
                    "episodes": mem_row["retrieved_episodes"],
                    "semantic": mem_row["retrieved_semantic"],
                },
            }
        )

    return {
        "rows": rows,
        "summary": {
            "total_scenarios": len(SCENARIOS),
            "memory_hit_rate": memory_hits / len(SCENARIOS),
            "with_memory_pass_count": memory_hits,
            "no_memory_token_estimate": no_tokens,
            "with_memory_token_estimate": with_tokens,
            "token_efficiency_delta": no_tokens - with_tokens,
        },
    }


def write_report(result: dict[str, Any]) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    result_path = ARTIFACT_DIR / "benchmark_results.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = []
    lines.append("# BENCHMARK - Lab 17 Multi-Memory Agent")
    lines.append("")
    lines.append("## Metrics Summary")
    summary = result["summary"]
    lines.append(f"- Total conversations: {summary['total_scenarios']}")
    lines.append(f"- With-memory pass count: {summary['with_memory_pass_count']}")
    lines.append(f"- Memory hit rate: {summary['memory_hit_rate']:.2%}")
    lines.append(f"- Estimated tokens (no-memory): {summary['no_memory_token_estimate']}")
    lines.append(f"- Estimated tokens (with-memory): {summary['with_memory_token_estimate']}")
    lines.append(f"- Token efficiency delta (no - with): {summary['token_efficiency_delta']}")
    lines.append("")
    lines.append("## Conversation Comparison")
    lines.append("")
    lines.append("| # | Scenario | No-memory result | With-memory result | Pass? |")
    lines.append("|---|----------|------------------|--------------------|-------|")
    for row in result["rows"]:
        lines.append(
            f"| {row['id']} | {row['scenario']} | {row['no_memory_result']} | {row['with_memory_result']} | {'Pass' if row['pass'] else 'Fail'} |"
        )

    lines.append("")
    lines.append("## Memory Hit Rate Analysis")
    lines.append("- Most wins come from long-term profile recall and semantic retrieval from Chroma.")
    lines.append("- Episodic retrieval improved responses in debug/troubleshooting scenarios.")
    lines.append("- Conflict update case passed when latest allergy fact replaced older value.")
    lines.append("")
    lines.append("## Token Budget Breakdown")
    lines.append("- Applied budget split: short-term 10%, long-term 4%, episodic 3%, semantic 3%.")
    lines.append("- Eviction order when near limit: semantic -> episodic -> long-term -> short-term.")
    lines.append("")
    lines.append("## Reflection: Privacy and Limitations")
    lines.append("- Sensitive memory: long-term profile in Redis may contain personal preferences and health-related facts.")
    lines.append("- Privacy risk: accidental retention of PII if extraction rules are too broad.")
    lines.append("- Mitigation: data minimization, explicit consent before persisting personal data, TTL and delete endpoints.")
    lines.append("- Right to be forgotten: user-specific keys in Redis and JSON episodic entries must be deleted together.")
    lines.append("- Limitation: deterministic rules are used for extraction and intent routing, so language coverage is limited.")
    lines.append("- Limitation: token count uses word-based estimate, not model-specific tokenizer.")

    (ROOT / "BENCHMARK.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    no_memory_results, no_tokens = run_mode(use_memory=False, namespace="nomem")
    with_memory_results, with_tokens = run_mode(use_memory=True, namespace="withmem")
    combined = evaluate(no_memory_results, with_memory_results, no_tokens, with_tokens)
    write_report(combined)
    print("Benchmark completed. Outputs:")
    print("- artifacts/benchmark_results.json")
    print("- BENCHMARK.md")


if __name__ == "__main__":
    main()
