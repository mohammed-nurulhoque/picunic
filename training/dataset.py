"""Dataset: render characters as they appear in terminal cells (8×16, natural positioning)."""

import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from torch.utils.data import Dataset
from charset import get_charset

# Terminal cell aspect ratio 1:2
CELL_W, CELL_H = 8, 16


class CharacterDataset(Dataset):
    def __init__(self, font_path: str, charset: str = 'distinct', samples_per_char: int = 50):
        self.chars = get_charset(charset)
        self.font_path = font_path
        self.samples_per_char = samples_per_char
        # Font size chosen so typical chars fill the cell naturally
        self.font = ImageFont.truetype(font_path, 14)  # ~14px for 16px cell height
        
    def __len__(self):
        return len(self.chars) * self.samples_per_char

    def __getitem__(self, idx):
        char_idx = idx % len(self.chars)
        anchor = self._render(self.chars[char_idx], augment=True)
        positive = self._render(self.chars[char_idx], augment=True)
        return anchor, positive, char_idx

    def _render(self, char: str, augment: bool = False) -> torch.Tensor:
        """Render char in 8×16 cell at natural terminal position (baseline-aligned)."""
        # Create cell-sized image
        img = Image.new('L', (CELL_W, CELL_H), 0)
        draw = ImageDraw.Draw(img)
        
        # Get font metrics for proper baseline positioning
        # Draw at x=0, y positioned so baseline is ~12px from top (typical for 16px cell)
        ascent, descent = self.font.getmetrics()
        # Position: baseline at ~75% down the cell
        baseline_y = int(CELL_H * 0.75)
        y = baseline_y - ascent
        
        # Center horizontally in cell
        bbox = draw.textbbox((0, 0), char, font=self.font)
        char_w = bbox[2] - bbox[0]
        x = (CELL_W - char_w) // 2 - bbox[0]
        
        draw.text((x, y), char, font=self.font, fill=255)
        
        if augment:
            img = self._augment(img)
        
        arr = np.array(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).unsqueeze(0)

    def _augment(self, img: Image.Image) -> Image.Image:
        arr = np.array(img, dtype=np.float32)
        
        # Foreground/background color variation
        # Map 0 -> random(0-64), 255 -> random(192-255)
        bg = random.randint(0, 64)
        fg = random.randint(192, 255)
        arr = bg + (arr / 255.0) * (fg - bg)
        
        # Noise σ=8
        if random.random() > 0.3:
            arr = arr + np.random.normal(0, 8, arr.shape)
        
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    def render_canonical(self, char_idx: int) -> torch.Tensor:
        return self._render(self.chars[char_idx], augment=False)


if __name__ == '__main__':
    ds = CharacterDataset('../assets/DejaVuSansMono.ttf', 'distinct')
    print(f"Dataset: {len(ds)} samples, {len(ds.chars)} chars")
    print(f"Cell size: {CELL_W}×{CELL_H}")
