# Prompt router — test set, ONNX on CPU (as in the gateway)

3,000 SWE-chat prompts from repositories never seen in training; labels: SWE-chat's own (LLM-annotated). Model: bge-small fine-tuned (Kaggle), exported to ONNX; prior-corrected; brief when the top label is a real task and p ≥ 0.87.

| Value | Result (95% interval) | Target |
|---|---|---|
| All briefs right | 92.2% (90.6%–93.7%) | ≥ 85% |
| Coverage of real tasks | 48.0% (46.1%–49.9%) | ≥ 40% |
| no_brief wrongly briefed | 5.2% (3.3%–7.4%) | ≤ 5% |
| Briefs given | 1,233 of 3,000 prompts | |
| CPU time per prompt (incl. tokenizing) | median 14.1 ms, p95 234.5 ms | ≤ 50 ms p95 |

Without a threshold: accuracy 72.1%, macro-F1 0.717, broad-group accuracy 79.6%.

| Label | Precision | Recall | F1 | n |
|---|---|---|---|---|
| bugfix | 0.76 | 0.80 | 0.78 | 570 |
| explain | 0.71 | 0.68 | 0.69 | 505 |
| feature | 0.73 | 0.68 | 0.71 | 563 |
| no_brief | 0.72 | 0.66 | 0.69 | 483 |
| refactor | 0.58 | 0.67 | 0.62 | 318 |
| run_action | 0.79 | 0.80 | 0.80 | 452 |
| tests | 0.69 | 0.78 | 0.73 | 109 |

| Slice | n | Accuracy |
|---|---|---|
| all | 3000 | 72.1% |
| long (> 510 tokens, cut) | 264 | 75.4% |
| error log / traceback | 105 | 88.6% |
| short (<= 20 chars) | 263 | 85.2% |
| correction / rejection | 1347 | 68.1% |
| agent: Claude Code | 2435 | 71.1% |
| agent: Codex | 172 | 80.8% |
| agent: Copilot CLI | 171 | 76.6% |
| agent: Cursor | 42 | 50.0% |
| agent: Gemini CLI | 13 | 61.5% |
| agent: OpenCode | 102 | 84.3% |
| agent: Pi | 65 | 70.8% |
