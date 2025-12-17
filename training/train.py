#!/usr/bin/env python3
"""Train character embeddings with contrastive learning."""

import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import CharacterDataset
from model import CharEncoder, contrastive_loss


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    dataset = CharacterDataset(args.font, args.charset, args.samples_per_char)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    num_classes = len(dataset.chars)
    print(f"Dataset: {len(dataset)} samples, {num_classes} chars")

    encoder = CharEncoder(args.embedding_dim).to(device)
    
    # Load pretrained encoder weights if provided
    if args.from_checkpoint:
        print(f"Loading pretrained weights from {args.from_checkpoint}")
        ckpt = torch.load(args.from_checkpoint, map_location=device, weights_only=False)
        encoder.load_state_dict(ckpt['encoder'])
    
    classifier = nn.Linear(args.embedding_dim, num_classes).to(device)
    params = list(encoder.parameters()) + list(classifier.parameters())
    print(f"Params: {sum(p.numel() for p in params):,}")

    optimizer = optim.AdamW(params, lr=args.lr)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=len(loader))
    ce = nn.CrossEntropyLoss()

    output = Path(args.output)
    output.mkdir(exist_ok=True)
    best_acc = 0

    for epoch in range(args.epochs):
        encoder.train()
        classifier.train()

        for anchor, positive, labels in tqdm(loader, desc=f"Epoch {epoch+1}", leave=False):
            anchor, positive, labels = anchor.to(device), positive.to(device), labels.to(device)
            emb_a, emb_p = encoder(anchor), encoder(positive)

            # Contrastive + classification
            loss = contrastive_loss(emb_a, emb_p)
            loss += (ce(classifier(emb_a), labels) + ce(classifier(emb_p), labels)) / 2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        # Eval: nearest neighbor accuracy
        encoder.eval()
        with torch.no_grad():
            canonical = torch.stack([dataset.render_canonical(i) for i in range(num_classes)]).to(device)
            canonical_emb = encoder(canonical)

            correct = 0
            for i in range(num_classes):
                for _ in range(3):  # 3 augmented tests per char
                    test = dataset._render(dataset.chars[i], augment=True).unsqueeze(0).to(device)
                    pred = (encoder(test) @ canonical_emb.t()).argmax().item()
                    correct += pred == i
            acc = correct / (num_classes * 3)

        print(f"Epoch {epoch+1}: acc={acc:.1%}")

        if acc > best_acc:
            best_acc = acc
            torch.save({'encoder': encoder.state_dict(), 'embedding_dim': args.embedding_dim, 'chars': dataset.chars}, output / 'best.pt')

    print(f"Best: {best_acc:.1%}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--font', required=True)
    p.add_argument('--charset', default='distinct')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--lr', type=float, default=5e-3)
    p.add_argument('--embedding-dim', type=int, default=64)
    p.add_argument('--samples-per-char', type=int, default=100)
    p.add_argument('--output', default='checkpoints')
    p.add_argument('--from-checkpoint', help='Load pretrained encoder weights')
    train(p.parse_args())
