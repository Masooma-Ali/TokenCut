# Attachments — Research and Plan for TokenCut

*Status: plan (nothing implemented yet). Written 2026-11-12 on branch `Attachments`.*

**Goal.** Real users will attach images, screenshots, PDFs, Word files, spreadsheets and
slides. TokenCut should cut the tokens (and so the price) these attachments cost, add little
latency, and **not lose information the model needs to answer correctly**.

**Bottom line.**
- Attachments are billed as ordinary input tokens, and are re-sent on every later request.
- With a paid API, the only lever is what TokenCut sends: a smaller image, fewer pages, text
  instead of a page picture, or a short note instead of a finished attachment. Pruning
  *inside* the model (research papers) is not possible through an API.
- TokenCut handles no attachments today: images and file documents are forwarded unchanged
  and counted as 0 tokens.
- Recommended first steps carry almost no quality risk: count attachment tokens, stop
  cutting tables with the head/tail rule, replace finished attachments with notes (R10
  extension), drop exact duplicates.

---

## 1. How providers bill attachments (checked in the official docs, Nov 2026)

### Images

| Provider | Formula | Limits / automatic downscaling |
|---|---|---|
| **Claude** | `⌈width / 28⌉ × ⌈height / 28⌉` visual tokens (one token per 28×28-pixel square) | Standard tier: long edge 1,568 px, max 1,568 tokens. High-resolution tier (Claude 4.7 and later): 2,576 px, max 4,784 tokens. Larger images are downsized automatically, keeping the aspect ratio. An oversized image **inside a tool result is rejected** with a validation error instead. |
| **OpenAI**, tile models (gpt-4o, gpt-4.1) | `85 + 170 × tiles` (512-px tiles after scaling: fit 2048×2048, shortest side 768) | `detail: "low"` = 85 tokens flat. 1024×1024 at high = 765. |
| **OpenAI**, patch models (gpt-5.x) | `⌈w/32⌉ × ⌈h/32⌉` × model multiplier (e.g. 1.2 for gpt-5.5) | `detail` = `low` / `high` / `original` / `auto`. On gpt-5.4, `low` can cost **more** than `high`. |
| **Gemini 3 / 4** | per item, set by `media_resolution` | image: low 280, medium 560, high 1,120 (default), ultra_high 2,240. Older Gemini pages quote 258 tokens per image / 768-px tile; the pages conflict. |

**Worked example — 1920×1080 screenshot on Claude**

| What is sent | Tokens |
|---|---|
| Claude 4.7+ (below the 2,576-px limit, sent as is) | 69 × 39 = **2,691** |
| Older Claude (auto-resized to 1456×819) | **1,560** |
| TokenCut resizes to 1568×882 on Claude 4.7+ | 56 × 32 = **1,792 (−33 %)** |
| Resized to 1280×720 | 46 × 26 = 1,196 |

File compression (JPEG quality, PNG → JPEG) saves **no tokens**: only width and height
matter. It can reduce upload size and latency.

### PDFs

| Provider | Processing | Cost |
|---|---|---|
| **Claude** | every page → **extracted text + rendered page image**, both sent | text ≈ 1,500–3,000 tokens per page **plus** the page image; max 600 pages (100 below a 1M context), 128 MB per request |
| **OpenAI** | text **and** page images on vision models | Responses API `input_file` has `detail` = `low` / `high` / `auto` for the page images only; text always included; no per-page figure published |
| **Gemini 3 / 4** | page rendered as an image (+ native text) | low 280, medium 560 (Google: quality "typically saturates" here), high 1,120 per page; Google's pages conflict on whether native text is billed |

### Other files

- **Word (.docx), PowerPoint (.pptx), text, code:** OpenAI extracts text only. Claude's
  document blocks do not accept .docx / .xlsx, so agents convert them to text first.
- **Excel / CSV:** converted to text by agents (OpenAI: first 1,000 rows per sheet).
- **Audio / video (Gemini only):** ~25 tokens per second of audio, 70 per video frame by default.
- **Files API** (`file_id`): keeps request payloads small; the docs do not say it lowers token cost.

### Prompt cache — the constraint every technique must respect

- Cached tokens cost about **0.1×** on Anthropic; writing the cache costs about **1.25×**.
  A finished attachment therefore still costs about 10 % on every later request.
- Anthropic: *adding or removing images anywhere in the prompt invalidates the messages
  cache.* Removing an old attachment costs a one-time cache rewrite of everything after it.
- Claude Code uses the 1-hour cache, so "cache expired" moments are rare.
- Anthropic's computer-use page (seen only in a summary, **to confirm**): keep about the last
  3 screenshots and prune in batches; on Claude 5.1+ avoid client-side pruning because it
  invalidates later thinking blocks.

---

## 2. Research papers

| Work | What it does | Result | Usable by TokenCut? |
|---|---|---|---|
| FastV (arXiv 2403.06764) | drops low-attention image tokens inside the model after layer 2 | ~half the image tokens removed, no average loss (authors) | **No** — inside the model |
| VisionZip (CVPR 2025, arXiv 2412.04467) | keeps dominant image tokens, merges the rest | ~95 % performance with ~10 % of tokens (authors) | **No** |
| Survey of token compression for MLLMs (arXiv 2507.20198) | survey | most visual tokens are redundant | background only |
| UniPruneBench (arXiv 2511.02650) | benchmark of visual token compression | – | background only |
| Token-pruning review (ACL 2026) | taxonomy: vision-side / LLM-side / hybrid | – | background only |
| ReVision (arXiv 2605.11212) | drops patches repeated across consecutive screenshots | −46 % tokens, +3 % success (5 history screenshots, Qwen2.5-VL-7B) | **No** — needs training and model internals; its *pixel-difference* baseline was **worse** than keeping everything |
| GUI screenshot pruning study (arXiv 2603.26041) | prunes old screenshots, keeps the current one whole | recent screenshots matter most | the **lesson** applies: keep the current image, spend less on old ones |
| DeepSeek-OCR (arXiv 2510.18234) | text rendered as images, ~10× compression | ~96 % OCR precision at 9–10× | **No** — needs its own model |

Why they do not apply: providers bill by a formula on the **image size**, before the model
runs. Self-hosted research saves GPU time; it cannot change an API bill. TokenCut can only
change the input: resize, crop, select pages, convert to text, or replace with a note.

---

## 3. How existing tools handle attachments (checked in source or official docs)

| Tool | Images | PDFs | Word / Excel | Token saving |
|---|---|---|---|---|
| Claude Code | paste / drag / path; large images resized* | short PDFs whole, >10 pages in ranges of ≤ 20 pages* | `.xlsx` refused* | 25k-token read cap; subagents |
| Codex CLI | `--image` / `view_image`; resized to max 2048 px (default `high`) | – | – | resizing |
| Claude app | up to 8000×8000 | ≤100 pages text + visuals; 101–1000 pages **text only** | text only, embedded images ignored | – |
| OpenHands | – | **converted to Markdown** (pdfminer) | Markdown (mammoth, xlsx, pptx) | `view` cut at 16k chars |
| Cline | base64 (5 MB in code example) | text only (pdf-parse)† | text (mammoth) | tool results cut to 8k chars |
| Goose | base64, 20 MB, crop option | `extract_text` / `extract_images` tool | separate docx / xlsx tools | agent picks what it needs |
| Gemini CLI, OpenCode, Aider | raw base64 | **raw base64** | – | none found |
| Headroom (proxy) | ML router: low detail / keep / crop / OCR → claims 40–90 % | – | – | no published answer-quality test; needs ~32–100 MB models |
| SHIFT (proxy, Rust, Apache-2.0) | resize by mode (performance → economy) | – | – | e.g. 4000×3000 → 1024×768 (its estimate uses the old ÷750 formula); uses port 8787 like TokenCut |

\* from a summary of Claude Code's docs. † extractor exists; use by the current read tool not confirmed.

Takeaways: text-converting agents (OpenHands, Goose, Cline) already send documents as text;
raw-sending agents (Gemini CLI, OpenCode, Aider) are where TokenCut can save most. Only Codex
and Claude Code resize images themselves.

---

## 4. What TokenCut does today (code on `main`, commit fc84af4)

| Attachment | Arrives as | TokenCut today |
|---|---|---|
| Image / screenshot | image block (user message or tool result) | forwarded unchanged, **counted as 0 tokens** (`SegmentKind.MEDIA`) |
| PDF sent as a file | document block / inlineData | forwarded unchanged, counted as 0 |
| PDF / Word converted to text by the agent | text tool result | ≥ 2,000 tokens → **head + tail + error lines** (35 %) |
| Excel / CSV converted to text | text tool result | same head + tail cut → **middle rows can be lost** (current quality risk) |

---

## 5. Plan

### Principles (quality guard)

1. **The attachment the current task uses gets only low-risk changes.**
2. **Deterministic and first-sight only:** a change is decided when the attachment first
   appears and re-sent byte-identical every turn, so the prompt cache holds.
3. **Finished attachments** may become a short note, only at cache-safe moments (cache event,
   or R10's check that the saving outweighs the cache rewrite).
4. **Never upscale, never shrink twice, never resize below what the task needs.**
5. **Fail open:** any error → send the original.
6. **Every note says how to get the original back** (re-read the file; later CCR retrieval).

### Techniques, per attachment type

| # | Technique | Type | Expected saving | Quality risk | Notes |
|---|---|---|---|---|---|
| 1 | **Count attachment tokens** per provider formula | all | enables the rest | none | image size from the file header, no new library |
| 2 | **Protect tables from head/tail cuts** (like R8 for diffs) | CSV / Excel as text | – | **fixes a current risk** | until a table-aware trimmer exists |
| 3 | **Finished attachments → note** at R10 / cache events | images, PDFs, docs | ≈ 100 % of each old one, on every later request | low | reuses R10's cost check |
| 4 | **Drop exact duplicates** → "same as image #k" | images, files | ≈ 100 % of the duplicate | none | identical bytes |
| 5 | **Downscale only between the tiers** (Claude 4.7+: 1,568–2,576 px → 1,568 px) | images | e.g. 1920×1080: 2,691 → 1,792 (−33 %) | low | the size older Claude models always get; skip for computer use, "read / zoom" requests, dense-text images; shrink oversized tool-result images to avoid the API error |
| 6 | **Crop** to the relevant region | images | 50–90 % (Headroom's figure) | medium | needs a reliable region signal; later |
| 7 | **PDF: text only for text pages**, keep page images for figures / tables / scans | PDFs | ≈ 33–38 % per text-heavy PDF (one user's report) | medium | needs a PDF library (pypdf, BSD; PyMuPDF reported AGPL) |
| 8 | **PDF / PPTX: relevant pages only** + note for the rest | PDFs, slides | depends on length | medium | same idea as code-chunk scoring |
| 9 | **Word: section-aware trimming** instead of head/tail | DOCX as text | depends | low | keep whole relevant sections |
| 10 | **Spreadsheet: header + summary + matching rows** + note | CSV / Excel | large on big sheets | medium | after #2 |
| 11 | **JSON: keep structure + sample items** | JSON | large on big arrays | low | Headroom does this |
| 12 | Provider controls: OpenAI `detail`, Gemini `media_resolution` | images, PDFs, video | varies | medium | per-model table; can cost more on gpt-5.4 |

**Not planned:** pruning inside the model (papers), trained routers / OCR models (against the
model-free design), pixel-difference cropping (worse than baseline in ReVision).

### Order

1 → 2 → 3 → 4 → 5 → 7 → 8 → 10 → 9 → 11 → 6 → 12. Steps 1, 2 and 4 carry essentially no
quality risk.

---

## 6. How to prove quality is not compromised

Build a test set before enabling any technique by default:

- screenshots with small text, error dialogs, UI layouts;
- PDFs with plain text, charts, tables and scanned pages;
- spreadsheets where the answer is in a middle row;
- Word files where the answer is in a middle section;
- multi-turn sessions that reuse an attachment several turns later.

For each item: ask the same question with the original and with TokenCut, compare answers
(exact checks or a judge), billed tokens, cost and latency. Report three arms: raw, provider
cache only, TokenCut. Enable a technique by default only where answers match.

---

## 7. Decisions needed from the team

1. **Phase:** CLAUDE.md says testing only; new features need the team's agreement (as for R8 / R10).
2. **User-pasted attachments:** TokenCut's rule is "never touch the user's messages".
   Techniques 3–12 on pasted files need explicit approval; tool-result attachments can come first.
3. **New libraries** (`requirements.txt` is a shared file): Pillow for resizing; a PDF
   library (licence check).
4. **Claude 5.1+ thinking-block warning:** confirm on Anthropic's computer-use page before #3
   is applied to screenshots.
5. **ODE:** attachment cost per provider and cache state are good bandit features, but ODE is
   deferred until after the mid evaluation.

---

## Sources

- Anthropic: [vision](https://platform.claude.com/docs/en/build-with-claude/vision.md),
  [PDF support](https://platform.claude.com/docs/en/build-with-claude/pdf-support.md),
  [prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching.md),
  [computer use](https://platform.claude.com/docs/en/agents-and-tools/tool-use/computer-use-tool.md)
- OpenAI: [images and vision](https://developers.openai.com/api/docs/guides/images-vision.md),
  [file inputs](https://developers.openai.com/api/docs/guides/file-inputs.md)
- Gemini: [media resolution](https://aistudio.google.com/docs/media-resolution),
  [document processing](https://aistudio.google.com/docs/document-processing),
  [image understanding](https://aistudio.google.com/docs/image-understanding)
- Apps: [Claude app uploads](https://support.claude.com/en/articles/8241126),
  [Claude Code workflows](https://code.claude.com/docs/en/common-workflows),
  [Claude Code issue #80449](https://github.com/anthropics/claude-code/issues/80449)
- Tool source: [Codex](https://github.com/openai/codex),
  [OpenHands editor](https://github.com/OpenHands/openhands-aci),
  [Cline](https://github.com/cline/cline), [Goose](https://github.com/block/goose),
  [Gemini CLI](https://github.com/google-gemini/gemini-cli),
  [OpenCode](https://github.com/anomalyco/opencode), [Aider](https://github.com/Aider-AI/aider),
  [Continue](https://github.com/continuedev/continue),
  [Headroom](https://github.com/headroomlabs-ai/headroom),
  [SHIFT](https://github.com/alohaninja/shift)
- Papers: [FastV](https://arxiv.org/abs/2403.06764), [VisionZip](https://arxiv.org/abs/2412.04467),
  [MLLM token compression survey](https://arxiv.org/abs/2507.20198),
  [UniPruneBench](https://arxiv.org/abs/2511.02650),
  [token pruning review (ACL 2026)](https://aclanthology.org/2026.acl-long.328.pdf),
  [ReVision](https://arxiv.org/abs/2605.11212),
  [GUI screenshot pruning](https://arxiv.org/abs/2603.26041),
  [DeepSeek-OCR](https://arxiv.org/abs/2510.18234)
