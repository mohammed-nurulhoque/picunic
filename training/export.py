#!/usr/bin/env python3
"""Export trained model to ONNX."""

import argparse
import json
import numpy as np
import torch
from pathlib import Path

from model import CharEncoder
from dataset import CharacterDataset


def export(args):
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    chars, dim = ckpt['chars'], ckpt['embedding_dim']

    encoder = CharEncoder(dim)
    encoder.load_state_dict(ckpt['encoder'])
    encoder.eval()

    out = Path(args.output)

    # ONNX - input is (B, 1, H=16, W=8)
    torch.onnx.export(encoder, torch.randn(1, 1, 16, 8), out.with_suffix('.onnx'),
                      input_names=['image'], output_names=['embedding'],
                      dynamic_axes={'image': {0: 'batch'}, 'embedding': {0: 'batch'}}, opset_version=14)
    print(f"Saved {out.with_suffix('.onnx')}")

    # Pre-compute embeddings
    if args.font:
        ds = CharacterDataset(args.font)
        emb = []
        with torch.no_grad():
            for c in chars:
                idx = ds.chars.index(c) if c in ds.chars else 0
                emb.append(encoder(ds.render_canonical(idx).unsqueeze(0)).squeeze().numpy())
        emb = np.array(emb, dtype=np.float32)
        emb.tofile(out.with_suffix('.embeddings.bin'))
        print(f"Saved {out.with_suffix('.embeddings.bin')} {emb.shape}")

    # Chars
    with open(out.with_suffix('.chars.json'), 'w') as f:
        json.dump({'chars': chars, 'embedding_dim': dim}, f)
    print(f"Saved {out.with_suffix('.chars.json')}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', default='model')
    p.add_argument('--font')
    export(p.parse_args())
