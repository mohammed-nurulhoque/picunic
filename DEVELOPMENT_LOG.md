# Development Log

## Architecture

### Training (Python)
- Render Unicode chars to 8×16 bitmaps (terminal 1:2 aspect ratio)
- Natural baseline positioning (not centered/stretched)
- CNN encoder: Conv→BN→ReLU→Pool (×3) → FC → 64-dim L2-normalized embedding
- Classification + contrastive loss
- Auto-discover distinct chars via pairwise embedding similarity

### Inference (Rust)
- Split image into 8×16 chunks
- ONNX CNN inference → cosine similarity with precomputed embeddings
- Best matching character per chunk

## Key Numbers
- 563 visually distinct characters (from 1215 candidates, threshold 0.80)
- 170K parameters
- 100% training accuracy
- ~100ms inference for 80×30 output

## Character Filtering
- Default: monochrome-safe (ASCII, Latin-1, Box Drawing, Block Elements, Geometric)
- `--ascii`: ASCII only (0x20-0x7E)
- `--all`: all 563 including emoji
