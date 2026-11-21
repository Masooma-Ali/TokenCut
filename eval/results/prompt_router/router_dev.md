# Prompt router — dev set, ONNX on CPU (as in the gateway)

3,000 SWE-chat prompts from repositories never seen in training; labels: SWE-chat's own (LLM-annotated). Model: bge-small fine-tuned (Kaggle), exported to ONNX; prior-corrected; brief when the top label is a real task and p ≥ 0.87.

| Value | Result (95% interval) | Target |
|---|---|---|
| All briefs right | 93.7% (92.4%–95.1%) | ≥ 85% |
| Coverage of real tasks | 45.8% (43.9%–47.6%) | ≥ 40% |
| no_brief wrongly briefed | 3.5% (1.8%–5.3%) | ≤ 5% |
| Briefs given | 1,179 of 3,000 prompts | |
| CPU time per prompt (incl. tokenizing) | median 16.2 ms, p95 236.6 ms | ≤ 50 ms p95 |

Without a threshold: accuracy 73.2%, macro-F1 0.726, broad-group accuracy 81.0%.

| Label | Precision | Recall | F1 | n |
|---|---|---|---|---|
| bugfix | 0.76 | 0.80 | 0.78 | 463 |
| explain | 0.72 | 0.73 | 0.73 | 564 |
| feature | 0.66 | 0.68 | 0.67 | 502 |
| no_brief | 0.77 | 0.68 | 0.72 | 463 |
| refactor | 0.64 | 0.68 | 0.66 | 374 |
| run_action | 0.85 | 0.81 | 0.83 | 494 |
| tests | 0.68 | 0.71 | 0.69 | 140 |

| Slice | n | Accuracy |
|---|---|---|
| all | 3000 | 73.2% |
| long (> 510 tokens, cut) | 276 | 73.6% |
| error log / traceback | 102 | 78.4% |
| short (<= 20 chars) | 240 | 85.8% |
| correction / rejection | 1188 | 68.9% |
| agent: Claude Code | 2370 | 72.4% |
| agent: Codex | 461 | 77.0% |
| agent: Copilot CLI | 35 | 77.1% |
| agent: Gemini CLI | 14 | 50.0% |
| agent: OpenCode | 115 | 74.8% |
| agent: Pi | 5 | 80.0% |

Check against Kaggle's dev predictions: max probability difference 1.55e-03, same top label for 100.0% of prompts.
