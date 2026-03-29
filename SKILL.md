---
name: sci-figure-gen
description: Generate publication-ready scientific conceptual diagrams and framework figures using AI image generation models (Gemini/GPT) via OpenRouter API, with automatic logo removal and AI super-resolution upscaling. Use when asked to create academic figures, conceptual diagrams, framework illustrations, or improve existing scientific figures for journal submission.
---

# Scientific Figure Generator

Generate publication-quality conceptual diagrams and framework figures for academic papers using AI image generation, with automated post-processing pipeline.

## When to Use

- Creating conceptual/framework diagrams for academic papers
- Improving existing hand-drawn or low-quality scientific figures
- Generating Nature/Science journal-style illustrations
- Any figure that involves icons, flowcharts, or conceptual relationships (NOT data-driven statistical plots)

## Architecture

```
Reference Image (compressed) → OpenRouter API (Gemini/GPT) → Smart Crop (logo removal) → EDSR 4x Super-Resolution → Final Output
```

## Prerequisites

1. **OpenRouter API Key**: Set as environment variable `OPENROUTER_API_KEY`
2. **Proxy** (if needed): Set `HTTPS_PROXY` environment variable
3. **Python packages**: `pip install requests Pillow super-image`

## Quick Start

```bash
# Set your API key (never hardcode in scripts!)
export OPENROUTER_API_KEY="sk-or-v1-your-key-here"
export HTTPS_PROXY="http://127.0.0.1:2080"  # if behind proxy

# Generate a figure
python scripts/generate.py --prompt "your prompt" --ref reference.png --output figure.png

# Or use the Python API directly
python -c "
from scripts.generate import ScienceFigureGenerator
gen = ScienceFigureGenerator()
gen.generate_and_enhance('your prompt', ref_image='ref.png', output='figure.png')
"
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

## Available Models

| Model | ID | Output Size | Best For |
|-------|------|-------------|----------|
| **Gemini 3.1 Flash Image** | `google/gemini-3.1-flash-image-preview` | ~1376×768 landscape | Best overall quality, supports wide format |
| Gemini 2.5 Flash Image | `google/gemini-2.5-flash-image` | ~1376×768 | Cheaper alternative |
| GPT-5 Image Mini | `openai/gpt-5-image-mini` | 1024×1024 square | Better rule-following, but square crops content |

**Recommendation**: Use Gemini 3.1 Flash Image for most cases. Generate 3-5 variants to find one that meets all constraints.

## Prompt Engineering Guide

### Style Presets

**Nature Journal Style** (recommended for top-tier journals):
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

**Light Morandi Palette** (sophisticated muted colors):
```
COLOR PALETTE:
- Cool zone: #D6E4EE (light dusty blue)
- Warm zone: #EBDADF (light dusty pink)
- Accent A: #B5C4A8 (Morandi sage)
- Accent B: #D4B5A0 (Morandi terracotta)
- Text: #3D3D3D, Lines: #6B6B6B
```

### Common Pitfalls & Solutions

| Problem | Solution |
|---------|----------|
| Text too tight/cramped | Specify "Segoe UI Light" or "Gill Sans Light", NOT Arial/Helvetica |
| Colors too dark/heavy | Use "LIGHT Morandi" with specific light hex codes |
| Colors too flashy | "RESTRAINED and ELEGANT, only 1-2 muted accent tones" |
| Icons different complexity | "identical complexity, 3-4 strokes each, matched pictogram set" |
| Unwanted extra labels | "DO NOT add any extra text like X, Y. Only the exact items listed" |
| Element repeats (e.g. label appears twice) | Generate 3-5 variants; or post-process to remove duplicate |
| AI logo/watermark | "Leave 15% blank white at bottom. No watermarks." + smart_crop |

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
