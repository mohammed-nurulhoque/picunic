#!/usr/bin/env python3
"""Export trained model to ONNX with embeddings for all monochrome Unicode."""

import argparse
import json
import numpy as np
import torch
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

from model import CharEncoder

CELL_W, CELL_H = 8, 16

# Monochrome-safe Unicode ranges (no emoji)
MONOCHROME_RANGES = [
    (0x0020, 0x007E),  # ASCII
    (0x00A0, 0x00FF),  # Latin-1 Supplement
    (0x0100, 0x017F),  # Latin Extended-A
    (0x2500, 0x257F),  # Box Drawing
    (0x2580, 0x259F),  # Block Elements
    (0x25A0, 0x25FF),  # Geometric Shapes
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


def get_monochrome_chars(font: ImageFont.FreeTypeFont) -> list[str]:
    """Get all renderable monochrome chars from font."""
    chars = []
    for start, end in MONOCHROME_RANGES:
        for cp in range(start, end + 1):
            char = chr(cp)
            try:
                bbox = font.getbbox(char)
                if bbox and (bbox[2] - bbox[0]) >= 0:  # renderable
                    chars.append(char)
            except:
                pass
    return chars


def export(args):
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    dim = ckpt['embedding_dim']

    encoder = CharEncoder(dim)
    encoder.load_state_dict(ckpt['encoder'])
    encoder.eval()

    out = Path(args.output)

    # ONNX export
    torch.onnx.export(encoder, torch.randn(1, 1, 16, 8), out.with_suffix('.onnx'),
                      input_names=['image'], output_names=['embedding'],
                      dynamic_axes={'image': {0: 'batch'}, 'embedding': {0: 'batch'}}, opset_version=14)
    print(f"Saved {out.with_suffix('.onnx')}")

    # Compute embeddings for ALL monochrome chars (not just training chars)
    if args.font:
        font = ImageFont.truetype(args.font, 14)
        
        if args.all_monochrome:
            chars = get_monochrome_chars(font)
            print(f"Using all {len(chars)} monochrome chars")
        else:
            chars = ckpt['chars']
            print(f"Using {len(chars)} training chars")
        
        emb = []
        with torch.no_grad():
            for c in chars:
                img = render_char(c, font)
                tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
                emb.append(encoder(tensor).squeeze().numpy())
        
        emb = np.array(emb, dtype=np.float32)
        emb.tofile(out.with_suffix('.embeddings.bin'))
        print(f"Saved {out.with_suffix('.embeddings.bin')} {emb.shape}")

        with open(out.with_suffix('.chars.json'), 'w') as f:
            json.dump({'chars': chars, 'embedding_dim': dim}, f)
        print(f"Saved {out.with_suffix('.chars.json')}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', default='model')
    p.add_argument('--font', required=True)
    p.add_argument('--all-monochrome', action='store_true', 
                   help='Export embeddings for all monochrome Unicode (default: training chars only)')
    export(p.parse_args())
