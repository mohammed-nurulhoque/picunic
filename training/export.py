#!/usr/bin/env python3
"""Export trained model to ONNX with embeddings for all monochrome Unicode."""

import argparse
import json
import numpy as np
import torch
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from fontTools.ttLib import TTFont

from model import CharEncoder

CELL_W, CELL_H = 8, 16

# Emoji/colored ranges to EXCLUDE
EMOJI_RANGES = [
    (0x1F300, 0x1F9FF),  # Miscellaneous Symbols and Pictographs, Emoticons, etc.
    (0x2600, 0x26FF),    # Miscellaneous Symbols (many are emoji)
    (0x2700, 0x27BF),    # Dingbats (many are emoji)
    (0xFE00, 0xFE0F),    # Variation Selectors
    (0x1F000, 0x1FFFF),  # All supplementary symbols
]


def is_emoji(cp: int) -> bool:
    """Check if codepoint is in emoji range."""
    for start, end in EMOJI_RANGES:
        if start <= cp <= end:
            return True
    return False


def get_font_chars(font_path: str) -> list[str]:
    """Get all characters available in the font, excluding emoji."""
    tt = TTFont(font_path)
    chars = []
    
    for table in tt['cmap'].tables:
        if hasattr(table, 'cmap'):
            for cp in table.cmap.keys():
                if cp < 0x20:  # Skip control chars
                    continue
                if cp > 0xFFFF:  # Skip supplementary planes (mostly emoji)
                    continue
                if is_emoji(cp):
                    continue
                char = chr(cp)
                if char not in chars:
                    chars.append(char)
    
    tt.close()
    return sorted(chars, key=ord)


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

    # Get all chars from font
    print(f"Scanning font: {args.font}")
    chars = get_font_chars(args.font)
    print(f"Found {len(chars)} characters in font")
    
    # Render and compute embeddings
    font = ImageFont.truetype(args.font, 14)
    emb = []
    valid_chars = []
    
    with torch.no_grad():
        for c in chars:
            try:
                img = render_char(c, font)
                tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
                emb.append(encoder(tensor).squeeze().numpy())
                valid_chars.append(c)
            except:
                pass  # Skip chars that fail to render
    
    emb = np.array(emb, dtype=np.float32)
    emb.tofile(out.with_suffix('.embeddings.bin'))
    print(f"Saved {out.with_suffix('.embeddings.bin')} {emb.shape}")

    with open(out.with_suffix('.chars.json'), 'w') as f:
        json.dump({'chars': valid_chars, 'embedding_dim': dim}, f)
    print(f"Saved {out.with_suffix('.chars.json')} ({len(valid_chars)} chars)")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', default='model')
    p.add_argument('--font', required=True)
    export(p.parse_args())
