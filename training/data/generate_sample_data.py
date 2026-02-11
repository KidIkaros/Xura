"""Generate a small sample dataset of colored shapes with captions.

Creates image-text pairs suited for Mamba3-JEPA training validation:
  - Simple geometric shapes (circles, rectangles, triangles)
  - Solid color backgrounds with contrasting shapes
  - Short descriptive captions

Usage:
    python data/generate_sample_data.py --n 100 --size 64 --out data/sample
"""

import argparse
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw


COLORS = {
    "red": (220, 50, 50),
    "blue": (50, 50, 220),
    "green": (50, 180, 50),
    "yellow": (220, 220, 50),
    "purple": (150, 50, 200),
    "orange": (230, 130, 30),
    "white": (240, 240, 240),
    "cyan": (50, 200, 200),
}

BG_COLORS = {
    "black": (15, 15, 15),
    "gray": (100, 100, 100),
    "dark blue": (20, 20, 60),
    "dark green": (20, 50, 20),
}

SHAPES = ["circle", "rectangle", "triangle"]

POSITIONS = ["center", "top-left", "top-right", "bottom-left", "bottom-right"]


def draw_shape(draw: ImageDraw.Draw, shape: str, color: tuple, pos: str, size: int):
    """Draw a shape on the image."""
    margin = size // 6
    s = size // 3  # shape size

    cx, cy = size // 2, size // 2
    if "top" in pos:
        cy = margin + s // 2
    if "bottom" in pos:
        cy = size - margin - s // 2
    if "left" in pos:
        cx = margin + s // 2
    if "right" in pos:
        cx = size - margin - s // 2

    if shape == "circle":
        draw.ellipse([cx - s//2, cy - s//2, cx + s//2, cy + s//2], fill=color)
    elif shape == "rectangle":
        draw.rectangle([cx - s//2, cy - s//2, cx + s//2, cy + s//2], fill=color)
    elif shape == "triangle":
        draw.polygon([
            (cx, cy - s//2),
            (cx - s//2, cy + s//2),
            (cx + s//2, cy + s//2),
        ], fill=color)


def generate_sample(idx: int, size: int, out_dir: Path):
    """Generate one image-text pair."""
    shape = random.choice(SHAPES)
    color_name, color_rgb = random.choice(list(COLORS.items()))
    bg_name, bg_rgb = random.choice(list(BG_COLORS.items()))
    pos = random.choice(POSITIONS)

    img = Image.new("RGB", (size, size), bg_rgb)
    draw = ImageDraw.Draw(img)
    draw_shape(draw, shape, color_rgb, pos, size)

    # Save image
    img_path = out_dir / "images" / f"{idx:04d}.png"
    img.save(img_path)

    # Save caption
    caption = f"A {color_name} {shape} on a {bg_name} background, positioned {pos}."
    txt_path = out_dir / "captions" / f"{idx:04d}.txt"
    txt_path.write_text(caption)

    return {
        "image": str(img_path.relative_to(out_dir)),
        "text": str(txt_path.relative_to(out_dir)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100, help="Number of samples")
    parser.add_argument("--size", type=int, default=64, help="Image size (pixels)")
    parser.add_argument("--out", type=str, default="data/sample", help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.out)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "captions").mkdir(parents=True, exist_ok=True)

    random.seed(42)
    manifest = []
    for i in range(args.n):
        entry = generate_sample(i, args.size, out_dir)
        manifest.append(entry)

    # Write manifest
    with open(out_dir / "manifest.jsonl", "w") as f:
        for entry in manifest:
            f.write(json.dumps(entry) + "\n")

    print(f"Generated {args.n} samples in {out_dir}/")
    print(f"  Images:   {out_dir}/images/")
    print(f"  Captions: {out_dir}/captions/")
    print(f"  Manifest: {out_dir}/manifest.jsonl")


if __name__ == "__main__":
    main()
