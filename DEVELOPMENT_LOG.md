# Development Log

## Architecture

### Training (Python)
- Render Unicode chars to 8×16 bitmaps (terminal 1:2 aspect ratio)
- Natural baseline positioning (not centered/stretched)
- CNN encoder: Conv→BN→ReLU→Pool (×3) → FC → 64-dim L2-normalized embedding
- Classification + contrastive loss
- Character filtering: excludes zero-width, combining marks, RTL scripts, emoji

### Inference (Rust)
- Split image into 8×16 chunks (bilinear sampling)
- ONNX Runtime for CNN inference
- Cosine similarity with precomputed character embeddings
- Optional Atkinson dithering for photos

## Key Numbers
- ~2200 monospace-safe Unicode characters
- 170K model parameters
- 64-dimensional embeddings
- ~100ms inference for 80×30 output

## Character Filtering

Characters excluded from the charset:
- Zero-width and combining marks (Mn, Me, Mc, Cf categories)
- Control characters (Cc)
- Modifier letters and symbols (Lm, Sk)
- RTL scripts (Hebrew, Arabic, etc.)
- Double-width characters (East Asian W/F)
- Emoji and supplementary planes

## Dithering

Atkinson dithering converts grayscale images to binary before matching:
- Scale automatically computed from output resolution
- Creates features at character-cell granularity
- Improves photo rendering by making patterns the CNN can match
