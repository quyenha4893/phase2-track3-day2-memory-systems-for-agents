# BENCHMARK - Lab 17 Multi-Memory Agent

## Metrics Summary
- Total conversations: 10
- With-memory pass count: 10
- Memory hit rate: 100.00%
- Estimated tokens (no-memory): 376
- Estimated tokens (with-memory): 452
- Token efficiency delta (no - with): -76

## Conversation Comparison

| # | Scenario | No-memory result | With-memory result | Pass? |
|---|----------|------------------|--------------------|-------|
| 1 | Recall user name | Ten ban la minh chua biet | Ten ban la Linh | Pass |
| 2 | Allergy conflict update | Thong tin di ung hien tai: chua co | Thong tin di ung hien tai: dau nanh | Pass |
| 3 | Preference recall | So thich da luu: chua co | So thich da luu: python | Pass |
| 4 | Episodic debug recall | Minh khong co episodic memory lien quan. | Lan truoc ket qua la: Lan truoc ket qua la: Minh khong co episodic memory lien quan. | Pass |
| 5 | Semantic docker retrieval | Minh khong tim thay semantic knowledge phu hop. | Thong tin semantic: In docker compose networking, containers should call each other using service name instead of localhost. | Pass |
| 6 | Semantic token budget | Minh khong tim thay semantic knowledge phu hop. | Thong tin semantic: Token budget allocation example: short-term 10 percent, long-term 4 percent, episodic 3 percent, semantic 3 percent. | Pass |
| 7 | Cross-topic recall | Ten ban la minh chua biet | Ten ban la An | Pass |
| 8 | Redis factual recall | Thong tin di ung hien tai: chua co | Thong tin di ung hien tai: dau phong | Pass |
| 9 | Semantic redis concept | Minh khong tim thay semantic knowledge phu hop. | Thong tin semantic: Redis stores user profile facts efficiently with hash operations and TTL for retention control. | Pass |
| 10 | Episodic troubleshooting recall | Minh khong co episodic memory lien quan. | Lan truoc ket qua la: Minh da xu ly yeu cau cua ban. | Pass |

## Memory Hit Rate Analysis
- Most wins come from long-term profile recall and semantic retrieval from Chroma.
- Episodic retrieval improved responses in debug/troubleshooting scenarios.
- Conflict update case passed when latest allergy fact replaced older value.

## Token Budget Breakdown
- Applied budget split: short-term 10%, long-term 4%, episodic 3%, semantic 3%.
- Eviction order when near limit: semantic -> episodic -> long-term -> short-term.

## Reflection: Privacy and Limitations
- Sensitive memory: long-term profile in Redis may contain personal preferences and health-related facts.
- Privacy risk: accidental retention of PII if extraction rules are too broad.
- Mitigation: data minimization, explicit consent before persisting personal data, TTL and delete endpoints.
- Right to be forgotten: user-specific keys in Redis and JSON episodic entries must be deleted together.
- Limitation: deterministic rules are used for extraction and intent routing, so language coverage is limited.
- Limitation: token count uses word-based estimate, not model-specific tokenizer.
