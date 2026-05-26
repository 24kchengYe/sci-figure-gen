---
name: sci-figure-gen
description: Generate publication-ready scientific conceptual diagrams and framework figures using AI image generation models via OpenRouter (Gemini/GPT) or MindCraft (gpt-image-2) APIs. Supports CS top-venue iconic style (ICLR/CVPR/ICML) and Nature/Science minimal style, with automatic post-processing and iterative refinement.
---

# Scientific Figure Generator

Generate publication-quality conceptual diagrams and framework figures for academic papers using AI image generation, with multi-backend support and automated post-processing.

## When to Use

- Creating conceptual/framework diagrams for academic papers
- Improving existing hand-drawn or low-quality scientific figures
- Generating figures in CS top-venue style (ICLR/CVPR/ICML) or Nature/Science style
- Any figure that involves icons, flowcharts, or conceptual relationships (NOT data-driven statistical plots)

## Architecture

```
Prompt (+ optional reference image)
  ↓
Backend Selection: OpenRouter (Gemini/GPT) or MindCraft (gpt-image-2)
  ↓
AI Image Generation (with retry + multi-variant)
  ↓
Smart Crop (logo removal) → EDSR 4x Super-Resolution
  ↓
[Optional] img2img iterative refinement (feed output back as reference)
  ↓
Final Output
```

## Prerequisites

1. **API Key** (at least one):
   - OpenRouter: `OPENROUTER_API_KEY` env var (needs proxy for overseas access)
   - MindCraft: `MINDCRAFT_API_KEY` env var (国内直连, no proxy)
2. **Proxy** (OpenRouter only): Set `HTTPS_PROXY` environment variable
3. **Python packages**: `pip install requests Pillow super-image`

## Quick Start

```bash
# OpenRouter backend (default)
python scripts/generate.py --prompt "your prompt" --output figure.png

# MindCraft backend (国内直连)
python scripts/generate.py --prompt "your prompt" --output figure.png --backend mindcraft

# With reference image (img2img refinement)
python scripts/generate.py --prompt "improve layout" --ref draft.png --output refined.png

# Generate 3 variants and pick the best
python scripts/generate.py --prompt-file prompts/my_prompt.txt --output figure.png --attempts 3
```

## Core Workflow

### Step 1: Prepare Reference Image

If you have an existing figure to improve, compress it for API upload:

```python
from scripts.generate import ScienceFigureGenerator
gen = ScienceFigureGenerator()
gen.compress_reference('original.png', 'ref_small.png', max_width=800)
```

**Why compress?** Original images (2000+ px) are too large as base64 for API upload and cause connection timeouts. 800px is the sweet spot.

### Step 2: Write Prompt

A good prompt has 4 parts:

1. **Content** — Exact text, labels, elements, layout
2. **Style** — Font, colors, line weight, overall aesthetic
3. **Constraints** — What NOT to include, specific rules
4. **Anti-logo** — "Leave 15% blank white at bottom. No watermarks."

See `examples/` directory for tested prompts.

### Step 3: Generate (with multiple attempts)

AI models don't always follow all constraints perfectly. Generate 3-5 variants and pick the best:

```python
gen = ScienceFigureGenerator()
for i in range(3):
    gen.generate_and_enhance(prompt, ref_image='ref.png', output=f'variant_{i}.png')
```

### Step 4: Automatic Post-Processing

The pipeline automatically:
1. **Smart crops** the bottom to remove AI watermarks/logos
2. **EDSR 4x upscales** from ~1376px to ~5500px using deep learning

## API Backends

### OpenRouter (海外, 需代理)

| Model | ID | Output Size | Best For |
|-------|------|-------------|----------|
| **Gemini 3.1 Flash Image** | `google/gemini-3.1-flash-image-preview` | ~1376×768 landscape | Best quality, wide format |
| Gemini 2.5 Flash Image | `google/gemini-2.5-flash-image` | ~1376×768 | Cheaper |
| GPT-5 Image Mini | `openai/gpt-5-image-mini` | 1024×1024 square | Better rule-following |

Interface: `/chat/completions` with `modalities: ["image", "text"]`. Returns base64 in `message.images[]` or `content[].image_url`.

### MindCraft 智匠 (国内直连, 无需代理)

| Model | ID | Output Size | Cost | Notes |
|-------|------|-------------|------|-------|
| **gpt-image-2** | `gpt-image-2` | 1024×1024 or 1536×1024 | 300 credits | Best available on MindCraft |

Interface: `/images/generations`. **Critical rules**:
- **MUST use `return_url: True`** — `return_url: False` causes empty response body for large images
- **Prompt max ~2000 characters** — longer prompts silently return empty body but still deduct credits
- Response: `result['data'][0]['file_url']` → separate GET to download the PNG
- **Disable proxy**: `proxies={'http': None, 'https': None}, verify=False`

```python
# MindCraft gpt-image-2 call template
resp = requests.post('https://api.mindcraft.com.cn/v1/images/generations',
    headers={'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'},
    json={'model': 'gpt-image-2', 'prompt': prompt, 'size': '1536x1024', 'return_url': True},
    timeout=600, verify=False, proxies={'http': None, 'https': None})
file_url = resp.json()['data'][0]['file_url']
img_bytes = requests.get(file_url, timeout=120, verify=False).content
```

**Backend selection guide**:
- OpenRouter Gemini: best quality, long prompts OK, but may 403 on some accounts
- MindCraft gpt-image-2: reliable 国内直连, but prompt must be <2000 chars

## Prompt Engineering Guide

### Core Principles for Academic Figures

**The #1 mistake**: treating an AI figure prompt like a text description. A good prompt is a **visual script** — it specifies layout, data flow, icons, colors, and spatial relationships, not paragraphs of content.

**Five rules for effective prompts**:

1. **Icons over text** — replace words with simple pictograms whenever possible (brain icon for neural net, scissors for splitting, paintbrush for regularization, target for matching)
2. **Math symbols over English** — use σ, ℒ, ε, IoU, π instead of writing "loss function", "threshold", "distribution"
3. **Max 2-3 lines per module** — title + one key innovation line + one metric. No paragraphs.
4. **Accuracy numbers belong in Results tables, not method figures** — don't clutter the framework with Dice=0.862, IoU=0.885 etc.
5. **Show data flow, not module internals** — arrows between modules tell the story; what's inside each box is secondary

### Style Presets

**CS Top-Venue Style (ICLR/CVPR/ICML)** — recommended for ML/CV/RS papers:
```
STYLE:
- Pure white background, flat 2D vector
- Low-saturation academic color palette (soft blue, sage green, warm peach)
- Simple line-art pictogram icons inside modules (3-5 strokes each, matched complexity)
- Mathematical notation prominent (σ, ℒ, ε, ∈, ≥)
- Sans-serif font, module titles ~24pt, subtitles ~14pt
- Generous whitespace — each box feels "airy"
- Color-coded grouping bands for logical sections
- Dashed borders for annotation/optional modules
- Thin arrows with short data-flow labels
- NO gradients, NO 3D, NO photorealism, NO clip art
```

**Nature/Science Minimal Style** — for biology, earth science, interdisciplinary:
```
STYLE:
- Font: Helvetica Neue LIGHT — thin, elegant, wide letter-spacing
- Text weight: LIGHT for everything. Headers by SIZE, not bold.
- Colors: minimalist — only 1-2 muted tones on white
- Lines: VERY THIN (0.5-1pt), precise
- Spacing: GENEROUS
- Pure white background
- Ultra-clean, precise, understated elegance
```

### Color Palettes

**CS Academic Palette** (low-saturation, print-friendly):
```
COLOR PALETTE:
- Group A: #C8D8E8 (soft blue), text: #1B2A3A (navy)
- Group B: #C8DCC8 (soft sage), text: #2F4E6B (slate)
- Group C: #E8D0C0 (soft peach), text: #5A3A2A (brown)
- Accent: #2A9D8F (teal) for arrows, highlights, innovation labels
- Warning: #D05040 (muted red-orange) for problem indicators
- Annotation module: #F5F0E5 (ivory) with dashed border
```

**Light Morandi Palette** (sophisticated muted colors):
```
COLOR PALETTE:
- Cool zone: #D6E4EE (light dusty blue)
- Warm zone: #EBDADF (light dusty pink)
- Accent A: #B5C4A8 (Morandi sage)
- Accent B: #D4B5A0 (Morandi terracotta)
- Text: #3D3D3D, Lines: #6B6B6B
```

### Icon Vocabulary for Common CS Concepts

When writing prompts, specify icons using these descriptions:

| Concept | Icon Description |
|---------|-----------------|
| Neural network / model | Brain outline or layered-nodes diagram |
| Segmentation / splitting | Scissors or split-line icon |
| GAN / generation | Paintbrush or magic-wand icon |
| Vectorization | Polygon outline icon |
| Geometric regularization | Right-angle ruler or protractor |
| Matching / alignment | Link/chain icon or crosshair/target |
| Data input | Satellite/camera icon or database cylinder |
| Output / result | Database icon or document with checkmark |
| Annotation / quality | Magnifying glass or stamp icon |
| Split (1→N) | One arrow splitting into multiple |
| Merge (N→1) | Multiple arrows merging into one |
| Loss function | ℒ symbol with subscript |
| Warning / problem | Triangle exclamation or red-orange highlight |

**Icon style rule**: "All icons should be simple line-art pictograms, same visual weight, 3-5 strokes each, matching the academic aesthetic. Think Nature Methods or ICLR figures."

### Prompt Structure Template

A good prompt for a CS framework figure follows this structure:

```
[1. OVERALL DESCRIPTION — 1 sentence]
Create a publication-quality method framework figure for a [field] paper,
in the visual style of [ICLR / CVPR / Nature Communications].

[2. CANVAS AND STYLE — 2-3 lines]
16:9 landscape. Pure white background. Flat 2D vector.
Color palette: [list 4-5 hex colors with roles].
Sans-serif font, titles 24pt, subtitles 14pt.

[3. LAYOUT STRUCTURE — describe spatial arrangement]
Three rows / Two columns / Left-right comparison / Circular flow...
[describe grouping bands, dividers, spatial relationships]

[4. MODULES — for each module:]
Module X: [icon description]. Title "[Name]".
  Line 1: "[key method/formula using math symbols]"
  Line 2: "[one-line metric or innovation highlight]"

[5. CONNECTIONS — arrows and data flow]
Arrow from A → B labeled "[data type]"
Dashed arrow from C bypassing to D (annotation: "[design decision]")
Dotted arrow from E → Output (label: "[role]")

[6. SPECIAL ELEMENTS — output panel, legend, etc.]

[7. NEGATIVE CONSTRAINTS — what NOT to include]

[8. POSITIVE CONSTRAINTS — what MUST be highlighted]
```

### Iterative img2img Refinement

AI-generated figures rarely come out perfect on the first try. Use this workflow:

1. **txt2img**: Generate 3 variants from the text prompt, pick the best layout
2. **img2img round 1**: Feed the best variant back as reference, prompt: "Keep the exact same layout and structure. Fix [specific issues: text overlap, missing arrow, wrong color]."
3. **img2img round 2**: Further refine details: "Adjust spacing between modules. Make the dashed arrow more visible."

This is especially effective with MindCraft gpt-image-2 and OpenRouter Gemini models.

### Common Pitfalls & Solutions

| Problem | Solution |
|---------|----------|
| **Too much text in boxes** | Max 2-3 lines per module. Use math symbols instead of words. Accuracy numbers go in Results table. |
| **No icons, just text boxes** | Specify icon for EVERY module: "Box with a [brain/scissors/paintbrush] icon" |
| **Flat, boring layout** | Use color-coded grouping bands, vary box sizes, add bypass/annotation arrows |
| **Text too tight/cramped** | Specify "generous whitespace" and "each box feels airy" |
| Colors too dark/heavy | Use "LIGHT" palette with specific light hex codes |
| Icons different complexity | "identical complexity, 3-4 strokes each, matched pictogram set" |
| Unwanted extra labels | "DO NOT add any extra text. Only the exact items listed" |
| Element repeats | Generate 3-5 variants; or post-process to remove duplicate |
| AI logo/watermark | "Leave 15% blank white at bottom. No watermarks." + smart_crop |
| **MindCraft empty body** | Use `return_url: True` (never False). Prompt must be <2000 chars. |
| **OpenRouter 403** | Account-level ToS restriction. Switch to MindCraft backend. |

### Anti-Logo Technique

Always append to every prompt:
```
Leave 15% of the image height as PURE BLANK WHITE SPACE at the very bottom.
No watermarks, no logos, no marks in this zone.
```

Then use `smart_crop_bottom()` to find where content ends and crop there.

### Post-Processing Text Edits (PIL)

When you need to change a few words without regenerating the whole figure (e.g. "LLM agent" → "Agent"), use PIL to surgically edit:

```python
from PIL import Image, ImageDraw, ImageFont
import numpy as np

im = Image.open('figure.png')
arr = np.array(im)

# 1. Locate text by scanning for dark pixels in a rough region
roi = arr[y1:y2, x1:x2]
gray = roi.mean(axis=2)
dark = gray < 100
rows = np.where(dark.any(axis=1))[0]
cols = np.where(dark.any(axis=0))[0]

# 2. Fill with white
draw = ImageDraw.Draw(im)
draw.rectangle([exact_x1, exact_y1, exact_x2, exact_y2], fill='white')

# 3. Write new text centered in the same area
font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', font_size)
draw.text((tx, ty), "New Text", fill='#333333', font=font)
im.save('figure.png')
```

This is much faster and more reliable than regenerating — AI models may change other elements unpredictably.

### Iterative Generation Strategy

For constraints that AI models struggle to follow consistently (e.g. "label appears exactly once"), use batch generation:

1. Generate 5 variants with the same prompt
2. Visually inspect each for constraint compliance
3. Pick the best one, then use PIL post-processing for minor fixes
4. Typical success rate: 2 out of 5 variants will follow the constraint

### Content Design Lessons (from real case study)

**Matching complexity across diagram sections:**
- When a diagram has parallel structures (e.g. Physical System vs Social System), internal illustrations must use the same visual weight (same number of strokes, same icon complexity)
- Specify explicitly: "Use 6-8 line strokes total — SAME complexity as the other side"

**Distinguishing types of transformation:**
- Geometric change (shape evolves): show bird's-eye footprint morphing (rectangle → L-shape)
- Semantic change (function evolves): show same building shape but with visible usage change (residential house → add shop-front/awning = commercial). The shape stays similar, the facade changes.
- These must look CLEARLY DIFFERENT from each other in the diagram

**Agent/AI components placement:**
- If the diagram shows an AI-driven simulation, agent cognitive components (Memory, Tools, Planning, Action) belong near the simulation core, not inside domain-specific rule boxes
- Use concise labels: "Agent" rather than "LLM agent" — the paper text provides full context

## Matplotlib Statistical Figure Best Practices

When generating data-driven figures (bar charts, area plots, heatmaps) with matplotlib for academic papers:

### Font Size Hierarchy (Nature/Science standard)
- **Sub-figure titles** (a, b, c): 24-28pt bold
- **Figure annotations** (data labels): 15-19pt bold
- **Axis labels**: 18-21pt
- **Tick labels**: 15-19pt
- **Legend text**: 18-22pt
- **Percentage labels inside bars**: 14-17pt white bold
- **Rule**: when in doubt, go bigger — figures get shrunk in print

### Color Palette for Unified Paper Style
Extract colors from existing figures in the paper for consistency. Example academic palette:
```python
colors = {
    'Category A': '#7BA3BF',   # slate blue-gray
    'Category B': '#D4908A',   # muted rose
    'Category C': '#E2BB7E',   # warm sand/amber
    'Category D': '#98B5A0',   # sage green
}
```
**Key rules:**
- All figures in one paper should share the same palette
- Use low-saturation, gray-toned colors (Morandi style)
- Avoid pure saturated RGB (no #FF0000, #0000FF)
- Test: print in grayscale — categories should still be distinguishable

### Category Merging Strategy
When raw data has too many categories (e.g. 9 methods), merge into 3-4 groups:
- Reduces visual noise in stacked charts and legends
- Put detailed breakdown in figure caption, not in the legend
- Legend shows only group names; caption explains: "AI-Driven includes Deep Learning and Machine Learning"

### Multi-Panel Layout Tips
- Use `gridspec` with explicit `width_ratios` and `wspace` for spacing control
- `wspace=0.28-0.38` prevents panels from being too cramped
- Legend at bottom in one row (`ncol=4`) is cleaner than top with wrapping
- Remove legend frame (`frameon=False`) for cleaner look

### Stacked Area vs Stacked Bar for Temporal Data
- **Area chart**: better for showing trends over 10+ years; smooth flow emphasizes growth patterns
- **Stacked bar**: better for discrete comparisons with few time points
- Add thin boundary lines on top of each area (`ax.plot(years, cumsum[i])`) for clarity
- For incomplete years (e.g. Jan-Mar 2026), use `axvline` with dashed line and italic annotation

### 100% Stacked Bar for Proportional Comparison
- Horizontal bars work better than vertical for long category names
- Show percentage labels (white bold) only if segment >= 10%
- Use white edgecolor between segments for separation

### EDSR Post-Processing Pitfalls
- EDSR 4x super-resolution tends to **darken light backgrounds** — cream/beige becomes yellow
- Fix: normalize near-white pixels after upscaling:
```python
arr = np.array(hr_img).astype(float)
mask = (arr[:,:,0] > 235) & (arr[:,:,1] > 235) & (arr[:,:,2] > 235)
arr[mask] = 255
```
- Always specify "PURE WHITE (#FFFFFF) background" in generation prompts
- Do NOT rely on the original image background being preserved through EDSR

### PIL Text Editing Limitations
- AI-generated text has anti-aliased edges; pixel-level cutting between adjacent letters fails
- When only 1-2 words need changing, **regenerate the whole figure** rather than PIL surgery
- PIL text replacement only works well for isolated text blocks with clear surrounding whitespace
- If you must use PIL: sample background color from nearby text-free area, not from a fixed color

### OpenRouter API Notes
- Gemini image models return images in `message.images[]` field, NOT in `content[].image_url`
- Always check both `content` (list of parts) and `images` (direct field) in the response
- Model IDs change frequently — use `/api/v1/models` endpoint to discover current IDs
- `google/gemini-3.1-flash-image-preview` is currently the best quality option

## API Reference

### `ScienceFigureGenerator`

```python
from scripts.generate import ScienceFigureGenerator

gen = ScienceFigureGenerator(
    api_key=None,           # Uses OPENROUTER_API_KEY env var if None
    proxy=None,             # Uses HTTPS_PROXY env var if None
    model="google/gemini-3.1-flash-image-preview",
    output_dir="./output"
)

# Compress reference image for upload
gen.compress_reference(input_path, output_path, max_width=800)

# Generate image from prompt (with optional reference)
gen.generate(prompt, ref_image=None, output_name="figure.png")

# Smart crop bottom whitespace/logo
gen.smart_crop(image_path)

# AI super-resolution 4x
gen.upscale(input_path, output_path, scale=4)

# Full pipeline: generate + crop + upscale
gen.generate_and_enhance(prompt, ref_image=None, output="figure.png", attempts=1)
```

## File Structure

```
sci-figure-gen/
├── SKILL.md              ← This file
├── scripts/
│   └── generate.py       ← Core generation + post-processing pipeline
├── examples/
│   ├── prompt_conceptual_diagram.txt    ← Example: handshake concept figure
│   └── prompt_framework_diagram.txt     ← Example: 3-column simulation framework
└── .gitignore
```
