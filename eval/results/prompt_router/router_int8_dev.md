# Prompt router (router_int8) — dev set, ONNX on CPU (as in the gateway)

3,000 SWE-chat prompts from repositories never seen in training; labels: SWE-chat's own (LLM-annotated). Model: bge-small fine-tuned (Kaggle), exported to ONNX; prior-corrected; brief when the top label is a real task and p ≥ 0.87.

| Value | Result (95% interval) | Target |
|---|---|---|
| All briefs right | 93.4% (91.8%–94.8%) | ≥ 85% |
| Coverage of real tasks | 38.9% (36.9%–40.8%) | ≥ 40% |
| no_brief wrongly briefed | 2.6% (1.1%–4.1%) | ≤ 5% |
| Briefs given | 998 of 3,000 prompts | |
| CPU time per prompt (incl. tokenizing) | median 12.5 ms, p95 207.6 ms | ≤ 50 ms p95 |

Without a threshold: accuracy 72.7%, macro-F1 0.722, broad-group accuracy 79.8%.

| Label | Precision | Recall | F1 | n |
|---|---|---|---|---|
| bugfix | 0.79 | 0.74 | 0.76 | 463 |
| explain | 0.71 | 0.73 | 0.72 | 564 |
| feature | 0.67 | 0.70 | 0.69 | 502 |
| no_brief | 0.69 | 0.71 | 0.70 | 463 |
| refactor | 0.66 | 0.69 | 0.67 | 374 |
| run_action | 0.85 | 0.80 | 0.82 | 494 |
| tests | 0.70 | 0.67 | 0.69 | 140 |

| Slice | n | Accuracy |
|---|---|---|
| all | 3000 | 72.7% |
| long (> 510 tokens, cut) | 276 | 75.0% |
| error log / traceback | 102 | 77.5% |
| short (<= 20 chars) | 240 | 85.0% |
| correction / rejection | 1188 | 67.7% |
| agent: Claude Code | 2370 | 71.9% |
| agent: Codex | 461 | 77.7% |
| agent: Copilot CLI | 35 | 68.6% |
| agent: Gemini CLI | 14 | 50.0% |
| agent: OpenCode | 115 | 73.9% |
| agent: Pi | 5 | 80.0% |

Check against Kaggle's dev predictions: max probability difference 7.95e-01, same top label for 87.6% of prompts.

## Speed: full-precision vs 8-bit (ONNX Runtime, this laptop's CPU, median of 10 runs)

| Model | Threads | 64 tokens | 512 tokens |
|---|---|---|---|
| full precision | 1 | 53.6 ms | 512.8 ms |
| full precision | 4 | 24.1 ms | 241.3 ms |
| full precision | 8 | 26.3 ms | 263.1 ms |
| 8-bit (dynamic) | 1 | 36.9 ms | 418.0 ms |
| 8-bit (dynamic) | 4 | 18.6 ms | 203.7 ms |
| 8-bit (dynamic) | 8 | 21.5 ms | 210.1 ms |

Reading: 8-bit weights save ~15–25% time and 75% size (33 vs 129 MB), but change the
probabilities (same top label as full precision for 87.6% of dev prompts; coverage at 0.87 drops
to 38.9%), so the threshold would need re-tuning. Long prompts stay slow: attention between
activations is not quantized. 4 threads is best (8 threads is slower). Not adopted.
