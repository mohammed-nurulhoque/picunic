#!/usr/bin/env python3
"""Discover visually distinct Unicode characters using learned embeddings."""

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from model import CharEncoder

CELL_W, CELL_H = 8, 16

# Unicode ranges to scan (monospace-friendly)
UNICODE_RANGES = [
    (0x0020, 0x007E),  # Basic ASCII
    (0x00A0, 0x00FF),  # Latin-1 Supplement
    (0x2500, 0x257F),  # Box Drawing
    (0x2580, 0x259F),  # Block Elements
    (0x25A0, 0x25FF),  # Geometric Shapes
    (0x2600, 0x26FF),  # Miscellaneous Symbols
    (0x2700, 0x27BF),  # Dingbats
    (0x2800, 0x28FF),  # Braille
    (0x3000, 0x303F),  # CJK Symbols
]


def render_char(char: str, font: ImageFont.FreeTypeFont) -> np.ndarray:
    """Render char to 8×16 with natural baseline positioning."""
    img = Image.new('L', (CELL_W, CELL_H), 0)
    draw = ImageDraw.Draw(img)
    
    ascent, descent = font.getmetrics()
    baseline_y = int(CELL_H * 0.75)
    y = baseline_y - ascent
    
    bbox = draw.textbbox((0, 0), char, font=font)
    char_w = bbox[2] - bbox[0]
    x = (CELL_W - char_w) // 2 - bbox[0]
    
    draw.text((x, y), char, font=font, fill=255)
    return np.array(img, dtype=np.float32) / 255.0


def get_candidate_chars(font: ImageFont.FreeTypeFont) -> list[str]:
    """Get all renderable chars from Unicode ranges."""
    chars = []
    for start, end in UNICODE_RANGES:
        for cp in range(start, end + 1):
            char = chr(cp)
            try:
                # Check if font can render it (non-empty)
                bbox = font.getbbox(char)
                if bbox and (bbox[2] - bbox[0]) > 0:
                    chars.append(char)
            except:
                pass
    return chars


def compute_embeddings(chars: list[str], font: ImageFont.FreeTypeFont, encoder: CharEncoder) -> np.ndarray:
    """Compute embeddings for all chars."""
    encoder.eval()
    embeddings = []
    with torch.no_grad():
        for char in chars:
            img = render_char(char, font)
            tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
            emb = encoder(tensor).squeeze().numpy()
            embeddings.append(emb)
    return np.array(embeddings)


def select_distinct(chars: list[str], embeddings: np.ndarray, threshold: float = 0.85) -> list[str]:
    """Greedily select chars with pairwise similarity below threshold."""
    n = len(chars)
    # Compute similarity matrix (cosine sim, embeddings already normalized)
    sim = embeddings @ embeddings.T
    
    selected = []
    selected_idx = []
    
    for i in range(n):
        # Check if this char is distinct from all selected
        if not selected_idx:
            selected.append(chars[i])
            selected_idx.append(i)
            continue
        
        max_sim = sim[i, selected_idx].max()
        if max_sim < threshold:
            selected.append(chars[i])
            selected_idx.append(i)
    
    return selected


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--font', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--threshold', type=float, default=0.85, help='Max similarity (lower = more distinct)')
    p.add_argument('--output', default='discovered_charset.py')
    args = p.parse_args()
    
    print(f"Loading font: {args.font}")
    font = ImageFont.truetype(args.font, 14)
    
    print("Finding renderable Unicode chars...")
    chars = get_candidate_chars(font)
    print(f"  Found {len(chars)} candidates")
    
    print(f"Loading model: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    encoder = CharEncoder(ckpt['embedding_dim'])
    encoder.load_state_dict(ckpt['encoder'])
    
    print("Computing embeddings...")
    embeddings = compute_embeddings(chars, font, encoder)
    
    print(f"Selecting distinct chars (threshold={args.threshold})...")
    distinct = select_distinct(chars, embeddings, args.threshold)
    print(f"  Selected {len(distinct)} distinct chars")
    
    # Save as Python file
    with open(args.output, 'w') as f:
        f.write(f'"""Auto-discovered distinct charset ({len(distinct)} chars)."""\n\n')
        f.write(f'DISTINCT = (\n')
        # Group by 40 chars per line
        for i in range(0, len(distinct), 40):
            chunk = ''.join(distinct[i:i+40])
            f.write(f'    "{chunk}"\n')
        f.write(')\n')
    print(f"Saved: {args.output}")
    
    # Also print for inspection
    print(f"\nDistinct chars:\n{''.join(distinct)}")


if __name__ == '__main__':
    main()
