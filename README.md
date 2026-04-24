# Lab #17 - Multi-Memory Agent

Repo nay cung cap mot bai nop dang source Python thuần cho lab `Build Multi-Memory Agent với LangGraph`.

## Thanh phan chinh

- `memory_agent.py`: skeleton LangGraph-style agent voi `MemoryState`, `retrieve_memory(state)`, router, prompt injection va 4 memory backends.
- `run_benchmark.py`: chay 10 benchmark multi-turn conversations va in bang so sanh `no-memory` vs `with-memory`.
- `data/semantic_docs.json`: semantic knowledge base dung cho keyword-search fallback.
- `BENCHMARK.md`: bang benchmark tong hop ket qua va 10 conversation samples.
- `REFLECTION.md`: privacy risks, deletion/TTL, gioi han ky thuat cua solution.

## 4 memory types

1. Short-term memory: `ShortTermMemory` luu conversation buffer va trim theo `memory_budget`.
2. Long-term profile memory: `LongTermProfileMemory` luu facts nhu ten, di ung, so thich, muc tieu hoc.
3. Episodic memory: `EpisodicMemory` luu cac task da hoan thanh hoac bai hoc da rut ra.
4. Semantic memory: `SemanticMemory` retrieve tu tap tai lieu `data/semantic_docs.json`.

## Router va prompt injection

`SkeletonLangGraphAgent.route()` se:

1. Tao `MemoryState`.
2. Goi `retrieve_memory(...)` de gom profile, episodic, semantic va recent conversation.
3. Trim recent conversation theo word budget.
4. Build prompt co 4 section ro rang:
   - `PROFILE MEMORY`
   - `EPISODIC MEMORY`
   - `SEMANTIC MEMORY`
   - `RECENT CONVERSATION`

## Chay demo

```bash
python memory_agent.py
python run_benchmark.py
```
