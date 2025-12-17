# picunic

Convert images to Unicode art using CNN embeddings. Includes `bigfoont` for rendering text as large Unicode art.

| Input | Output |
|-------|--------|
| ![Input](line-art.png) | ![Output](line-art-out.png) |

## Usage

### picunic - image to Unicode

```bash
picunic image.png              # 80 chars wide
picunic image.png -w 120       # custom width
picunic image.png -w 120 -H 40 # explicit height
picunic image.png --ascii      # ASCII only
picunic image.png --all        # include emoji chars
picunic image.png -i           # invert colors
```

### bigfoont - text to large Unicode

```bash
bigfoont "HELLO" -w 4 -H 4                                  # 4×4 chars per letter
bigfoont "HELLO" -w 3 -H 3 --font assets/PressStart2P.ttf   # pixel font (best)
bigfoont "HELLO" --ascii                                    # ASCII output only
```

## How it works

1. Train CNN encoder on 563 visually distinct Unicode characters
2. Split input image into 8×16 chunks (terminal cell aspect ratio)
3. Embed each chunk → cosine similarity with precomputed char embeddings
4. Output best matching character per chunk

## Build

```bash
cargo build --release
```

## Training

```bash
cd training
pip install torch pillow numpy tqdm onnx

python train.py --font ../assets/DejaVuSansMono.ttf --epochs 50
python export.py --checkpoint checkpoints/best.pt --output ../assets/encoder --font ../assets/DejaVuSansMono.ttf
```

### Discover distinct characters

Use learned embeddings to find visually distinct Unicode chars:

```bash
python discover_distinct.py --font ../assets/DejaVuSansMono.ttf \
  --checkpoint checkpoints/best.pt --threshold 0.80
```

## Vibe-coded

This project was 100% vibe-coded in a single session with cursor-cli. See [CHAT_TRANSCRIPT.md](CHAT_TRANSCRIPT.md) for the full conversation.
