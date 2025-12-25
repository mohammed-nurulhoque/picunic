#!/usr/bin/env python3
"""Export trained model to ONNX with embeddings for all monochrome Unicode."""

import argparse
import json
import numpy as np
import torch
import unicodedata
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from fontTools.ttLib import TTFont

from model import CharEncoder

CELL_W, CELL_H = 8, 16

# Ranges to EXCLUDE (emoji + RTL scripts + complex scripts)
EXCLUDE_RANGES = [
    # Emoji
    (0x1F300, 0x1F9FF),  # Miscellaneous Symbols and Pictographs, Emoticons, etc.
    (0x2600, 0x26FF),    # Miscellaneous Symbols (many are emoji)
    (0x2700, 0x27BF),    # Dingbats (many are emoji)
    (0xFE00, 0xFE0F),    # Variation Selectors
    (0x1F000, 0x1FFFF),  # All supplementary symbols
    # RTL scripts (mess up text direction)
    (0x0590, 0x05FF),    # Hebrew
    (0x0600, 0x06FF),    # Arabic
    (0x0700, 0x074F),    # Syriac
    (0x0750, 0x077F),    # Arabic Supplement
    (0x0780, 0x07BF),    # Thaana
    (0x07C0, 0x07FF),    # NKo
    (0x08A0, 0x08FF),    # Arabic Extended-A
    (0xFB50, 0xFDFF),    # Arabic Presentation Forms-A
    (0xFE70, 0xFEFF),    # Arabic Presentation Forms-B
    # Complex scripts with combining marks
    (0x0E00, 0x0E7F),    # Thai
    (0x0E80, 0x0EFF),    # Lao
    (0x0F00, 0x0FFF),    # Tibetan
    (0x1000, 0x109F),    # Myanmar
    # Other problematic
    (0xFFFC, 0xFFFC),    # Object Replacement Character
    (0xFFF9, 0xFFFB),    # Interlinear annotation anchors
]

# Unicode general categories that are zero-width or problematic
EXCLUDE_CATEGORIES = {
    'Mn',  # Mark, Nonspacing (combining marks)
    'Me',  # Mark, Enclosing
    'Mc',  # Mark, Spacing Combining
    'Cf',  # Format, Other (zero-width joiners, direction marks, etc.)
    'Cc',  # Control characters (C0/C1)
    'Lm',  # Modifier letters (often rendered narrow)
    'Sk',  # Modifier symbols (diacritics, often narrow)
}

# Additional ranges to exclude for monospace alignment
ADDITIONAL_EXCLUDE_RANGES = [
    (0x0080, 0x009F),    # C1 control characters
    (0x02B0, 0x02FF),    # Spacing Modifier Letters (often narrow)
    (0x1D00, 0x1DBF),    # Phonetic Extensions (often narrow)
    (0x2000, 0x200A),    # Various-width spaces
    (0x2070, 0x209C),    # Superscripts and Subscripts (narrow)
    (0x2E00, 0x2E7F),    # Supplemental Punctuation (variable width)
    (0xA700, 0xA71F),    # Modifier Tone Letters
]


def is_excluded(cp: int) -> bool:
    """Check if codepoint should be excluded."""
    # Check explicit ranges (emoji, RTL, complex scripts)
    for start, end in EXCLUDE_RANGES:
        if start <= cp <= end:
            return True
    
    # Check additional exclusion ranges (narrow/variable-width chars)
    for start, end in ADDITIONAL_EXCLUDE_RANGES:
        if start <= cp <= end:
            return True
    
    # Check for problematic characters using Unicode category
    try:
        char = chr(cp)
        category = unicodedata.category(char)
        if category in EXCLUDE_CATEGORIES:
            return True
        
        # Also exclude characters with east asian width 'W' or 'F' (double-width)
        # as they break monospace alignment
        ea_width = unicodedata.east_asian_width(char)
        if ea_width in ('W', 'F'):  # Wide or Fullwidth
            return True
            
    except (ValueError, TypeError):
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
                if is_excluded(cp):
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

    # ONNX export (without external data for web compatibility)
    torch.onnx.export(encoder, torch.randn(1, 1, 16, 8), out.with_suffix('.onnx'),
                      input_names=['image'], output_names=['embedding'],
                      dynamic_axes={'image': {0: 'batch'}, 'embedding': {0: 'batch'}}, 
                      opset_version=14,
                      export_params=True,  # Embed parameters in model
                      do_constant_folding=True)  # Fold constants
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
