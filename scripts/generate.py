# -*- coding: utf-8 -*-
"""
Scientific Figure Generator — AI-powered publication figure pipeline.

Usage:
    python generate.py --prompt "..." --ref reference.png --output figure.png
    python generate.py --prompt-file prompts/my_prompt.txt --output figure.png
    python generate.py --prompt "..." --output figure.png --attempts 3

Environment variables:
    OPENROUTER_API_KEY  — Your OpenRouter API key (required)
    HTTPS_PROXY         — Proxy URL if behind firewall (optional)
"""

import sys
import os
import json
import base64
import argparse

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

try:
    import requests
    from PIL import Image
    import numpy as np
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install requests Pillow numpy")
    sys.exit(1)


class ScienceFigureGenerator:
    """End-to-end pipeline: AI generate → crop logo → super-resolution.

    Supports two backends:
      - OpenRouter (Gemini/GPT via chat/completions, needs proxy for overseas)
      - MindCraft (gpt-image-2 via images/generations, 国内直连 no proxy)
    """

    MODELS = {
        # OpenRouter models
        "gemini-flash": "google/gemini-3.1-flash-image-preview",
        "gemini-2.5": "google/gemini-2.5-flash-image",
        "gpt-mini": "openai/gpt-5-image-mini",
        "gpt-full": "openai/gpt-5-image",
        # MindCraft models
        "gpt-image-2": "gpt-image-2",
    }

    # Models that use MindCraft /images/generations endpoint
    MINDCRAFT_MODELS = {"gpt-image-2"}

    def __init__(self, api_key=None, proxy=None, model="gemini-flash",
                 output_dir=".", backend=None):
        """Initialize generator.

        Args:
            api_key: API key. Auto-detected from env vars if None.
            proxy: Proxy URL (OpenRouter only). Auto-detected if None.
            model: Model shortname or full ID.
            output_dir: Directory for output files.
            backend: Force backend: "openrouter" or "mindcraft". Auto-detected if None.
        """
        self.model = self.MODELS.get(model, model)
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Determine backend
        if backend:
            self.backend = backend
        elif self.model in self.MINDCRAFT_MODELS:
            self.backend = "mindcraft"
        else:
            self.backend = "openrouter"

        # Set API key and proxy based on backend
        if self.backend == "mindcraft":
            self.api_key = api_key or os.environ.get("MINDCRAFT_API_KEY",
                "MC-72C89F3896E04155B26DD4830FABFA96")
            self.base_url = "https://api.mindcraft.com.cn/v1"
            # MindCraft: disable proxy, disable SSL verify
            self.session = requests.Session()
            self.session.verify = False
            self.session.proxies = {"http": None, "https": None}
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        else:
            self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
            if not self.api_key:
                raise ValueError(
                    "API key required. Set OPENROUTER_API_KEY or MINDCRAFT_API_KEY env var."
                )
            self.base_url = "https://openrouter.ai/api/v1"
            proxy = proxy or os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")
            self.session = requests.Session()
            if proxy:
                self.session.proxies = {"http": proxy, "https": proxy}

    def compress_reference(self, input_path, output_path, max_width=800):
        """Compress reference image for API upload (avoids timeout)."""
        im = Image.open(input_path)
        w, h = im.size
        if w > max_width:
            ratio = max_width / w
            im = im.resize((max_width, int(h * ratio)), Image.LANCZOS)
        im.save(output_path, optimize=True)
        print(f"  Compressed: {w}x{h} -> {im.size}")
        return output_path

    def generate(self, prompt, ref_image=None, output_name="figure.png"):
        """Generate an image. Routes to OpenRouter or MindCraft based on backend."""
        if self.backend == "mindcraft":
            return self._generate_mindcraft(prompt, output_name)
        else:
            return self._generate_openrouter(prompt, ref_image, output_name)

    def _generate_mindcraft(self, prompt, output_name):
        """MindCraft /images/generations endpoint."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "prompt": prompt,
            "size": "1536x1024",
            "quality": "high",
            "return_url": True,  # CRITICAL: must be True
        }
        print(f"  Generating with MindCraft {self.model} ({len(prompt)} chars)...")
        if len(prompt) > 2000:
            print(f"  WARNING: Prompt is {len(prompt)} chars. MindCraft gpt-image-2 "
                  f"may return empty body for prompts >2000 chars.")

        try:
            resp = self.session.post(
                f"{self.base_url}/images/generations",
                headers=headers, json=payload, timeout=(30, 600),
            )
        except Exception as e:
            print(f"  Connection error: {e}")
            return None

        if resp.status_code != 200:
            msg = resp.text[:200] if resp.content else "empty"
            print(f"  API error {resp.status_code}: {msg}")
            return None

        if not resp.content:
            print(f"  Empty response body (prompt too long or service issue)")
            return None

        try:
            data = resp.json()
        except Exception:
            print(f"  Invalid JSON")
            return None

        items = data.get("data", [])
        if not isinstance(items, list) or not items:
            print(f"  No image data in response")
            return None

        file_url = items[0].get("file_url", "")
        if not file_url:
            print(f"  No file_url. Keys: {list(items[0].keys())}")
            return None

        print(f"  Downloading from CDN...")
        try:
            img_resp = self.session.get(file_url, timeout=120)
            if img_resp.status_code == 200 and len(img_resp.content) > 1000:
                out_path = os.path.join(self.output_dir, output_name)
                with open(out_path, "wb") as f:
                    f.write(img_resp.content)
                w = items[0].get("image_width", "?")
                h = items[0].get("image_height", "?")
                print(f"  Saved: {output_name} ({w}x{h}, {len(img_resp.content)//1024} KB)")
                return out_path
        except Exception as e:
            print(f"  Download failed: {e}")

        return None

    def _generate_openrouter(self, prompt, ref_image, output_name):
        """OpenRouter /chat/completions endpoint."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        content = []
        if ref_image:
            with open(ref_image, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            ext = ref_image.rsplit(".", 1)[-1].lower()
            mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}.get(ext, f"image/{ext}")
            content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})
        content.append({"type": "text", "text": prompt})

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "modalities": ["image", "text"],
        }

        print(f"  Generating with OpenRouter {self.model}...")
        try:
            resp = self.session.post(
                f"{self.base_url}/chat/completions",
                headers=headers, json=payload, timeout=300,
            )
        except requests.exceptions.ConnectionError as e:
            print(f"  Connection error (ref image may be too large): {e}")
            return None

        if resp.status_code != 200:
            print(f"  API error {resp.status_code}: {resp.text[:300]}")
            return None

        try:
            data = resp.json()
        except Exception:
            print(f"  Invalid JSON response")
            return None

        msg = data.get("choices", [{}])[0].get("message", {})

        # Check 'images' field (Gemini style)
        for img in msg.get("images", []):
            result = self._extract_image(img, output_name)
            if result:
                return result

        # Check 'content' field (GPT style)
        content_resp = msg.get("content", "")
        if isinstance(content_resp, list):
            for part in content_resp:
                if isinstance(part, dict) and part.get("type") in ("image_url", "output_image"):
                    result = self._extract_image(part, output_name)
                    if result:
                        return result

        print(f"  No image in response. Text: {str(content_resp)[:150]}")
        return None

    def _extract_image(self, img_data, output_name):
        """Extract base64 image data and save to file."""
        if not isinstance(img_data, dict):
            return None
        url = img_data.get("image_url", img_data.get("url", ""))
        if isinstance(url, dict):
            url = url.get("url", "")
        if not url.startswith("data:"):
            return None

        try:
            b64_data = url.split(",", 1)[1]
            img_bytes = base64.b64decode(b64_data)
            out_path = os.path.join(self.output_dir, output_name)
            with open(out_path, "wb") as f:
                f.write(img_bytes)
            im = Image.open(out_path)
            print(f"  Saved: {output_name} ({im.size[0]}x{im.size[1]})")
            return out_path
        except Exception as e:
            print(f"  Failed to decode image: {e}")
            return None

    def smart_crop(self, image_path):
        """Crop bottom blank/logo area by scanning for content boundary."""
        im = Image.open(image_path)
        arr = np.array(im)
        h, w = arr.shape[:2]

        if len(arr.shape) == 3:
            gray = arr.mean(axis=2)
        else:
            gray = arr.astype(float)

        # Scan from bottom up: find last row with >2% dark pixels
        for row in range(h - 1, 0, -1):
            dark_ratio = (gray[row, :] < 200).sum() / w
            if dark_ratio > 0.02:
                margin = int(h * 0.012)
                crop_row = min(row + margin, h)
                if crop_row < h - 5:
                    im_cropped = im.crop((0, 0, w, crop_row))
                    im_cropped.save(image_path)
                    print(f"  Cropped: {h} -> {crop_row} (removed {h - crop_row}px)")
                return image_path

        return image_path

    def upscale(self, input_path, output_path=None, scale=4):
        """AI super-resolution using EDSR model."""
        try:
            from super_image import EdsrModel, ImageLoader
        except ImportError:
            print("  super-image not installed. Install with: pip install super-image")
            print("  Skipping upscale.")
            if output_path and output_path != input_path:
                import shutil
                shutil.copy2(input_path, output_path)
            return output_path or input_path

        if output_path is None:
            base, ext = os.path.splitext(input_path)
            output_path = f"{base}_hr{ext}"

        print(f"  EDSR x{scale} upscaling...")
        model = EdsrModel.from_pretrained("eugenesiow/edsr-base", scale=scale)
        img = Image.open(input_path)
        inputs = ImageLoader.load_image(img)
        preds = model(inputs)
        ImageLoader.save_image(preds, output_path)
        result = Image.open(output_path)
        print(f"  {img.size[0]}x{img.size[1]} -> {result.size[0]}x{result.size[1]}")
        return output_path

    def generate_and_enhance(self, prompt, ref_image=None, output="figure.png", attempts=1):
        """Full pipeline: generate → crop → upscale. Returns list of output paths."""
        results = []
        for i in range(attempts):
            suffix = f"_{chr(97 + i)}" if attempts > 1 else ""
            base, ext = os.path.splitext(output)
            raw_name = f"{base}{suffix}_raw{ext}"
            final_name = f"{base}{suffix}{ext}"

            path = self.generate(prompt, ref_image=ref_image, output_name=raw_name)
            if path:
                self.smart_crop(path)
                final_path = os.path.join(self.output_dir, final_name)
                self.upscale(path, final_path)
                results.append(final_path)
                print(f"  -> {final_name}")

        return results

    def preview(self, image_path, max_size=1500):
        """Create a smaller preview image."""
        im = Image.open(image_path)
        w, h = im.size
        ratio = min(max_size / w, max_size / h)
        if ratio < 1:
            im = im.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
        base, ext = os.path.splitext(image_path)
        preview_path = f"{base}_preview{ext}"
        im.save(preview_path)
        return preview_path


def main():
    parser = argparse.ArgumentParser(description="Generate publication-ready scientific figures")
    parser.add_argument("--prompt", type=str, help="Generation prompt text")
    parser.add_argument("--prompt-file", type=str, help="Read prompt from file")
    parser.add_argument("--ref", type=str, help="Reference image path")
    parser.add_argument("--output", "-o", type=str, default="figure.png", help="Output filename")
    parser.add_argument("--model", type=str, default="gemini-flash",
                        help="Model: gemini-flash, gemini-2.5, gpt-mini, gpt-full, gpt-image-2, or full model ID")
    parser.add_argument("--backend", type=str, default=None, choices=["openrouter", "mindcraft"],
                        help="Force backend (auto-detected from model if not set)")
    parser.add_argument("--attempts", type=int, default=1, help="Number of variants to generate")
    parser.add_argument("--output-dir", type=str, default=".", help="Output directory")
    parser.add_argument("--no-upscale", action="store_true", help="Skip EDSR upscaling")
    parser.add_argument("--no-crop", action="store_true", help="Skip smart cropping")
    parser.add_argument("--compress-ref", type=int, default=800,
                        help="Compress ref image to this max width (0 to disable)")

    args = parser.parse_args()

    if not args.prompt and not args.prompt_file:
        parser.error("Either --prompt or --prompt-file is required")

    prompt = args.prompt
    if args.prompt_file:
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            prompt = f.read().strip()

    gen = ScienceFigureGenerator(model=args.model, output_dir=args.output_dir,
                                  backend=args.backend)

    # Compress reference if provided
    ref = args.ref
    if ref and args.compress_ref > 0:
        compressed = os.path.join(args.output_dir, "_ref_compressed.png")
        gen.compress_reference(ref, compressed, max_width=args.compress_ref)
        ref = compressed

    # Generate
    for i in range(args.attempts):
        suffix = f"_{chr(97 + i)}" if args.attempts > 1 else ""
        base, ext = os.path.splitext(args.output)
        raw_name = f"{base}{suffix}{ext}"

        path = gen.generate(prompt, ref_image=ref, output_name=raw_name)
        if path:
            if not args.no_crop:
                gen.smart_crop(path)
            if not args.no_upscale:
                hr_name = f"{base}{suffix}_hr{ext}"
                gen.upscale(path, os.path.join(args.output_dir, hr_name))

    print("\nDone!")


if __name__ == "__main__":
    main()
