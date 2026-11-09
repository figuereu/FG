"""Sanity checks to run before trusting a training run."""
import math
from itertools import islice
from pathlib import Path

import numpy as np
import option
import pandas as pd
import torch
from loss import mil_loss
from model import Model
from train import build_loader

INITIAL_LOSS_BATCHES = 5
OVERFIT_VIDEOS_PER_GROUP = 2
OVERFIT_STEPS = 500
OVERFIT_LR = 1e-3


def seeded_model(args):
    torch.manual_seed(args.seed)
    return Model(args).to(args.device)


def batch_loss(model, features, classes, lengths, args):
    logits = model(features.to(args.device))
    labels = (classes > 0).float().to(args.device)
    return mil_loss(logits, labels, lengths.to(args.device), args.mil_k_ratio)


@torch.no_grad()
def check_initial_loss(normal_loader, anomal_loader, args):
    model = seeded_model(args).eval()
    losses = []
    for (normal_feat, normal_class, normal_len), (anomal_feat, anomal_class, anomal_len) in islice(
        zip(normal_loader, anomal_loader), INITIAL_LOSS_BATCHES
    ):
        features = torch.cat([normal_feat, anomal_feat])
        classes = torch.cat([normal_class, anomal_class])
        lengths = torch.cat([normal_len, anomal_len])
        losses.append(batch_loss(model, features, classes, lengths, args).item())
    print("\n=== 1. Initial loss, untrained model ===")
    print(f"mean of {len(losses)} batches: {np.mean(losses):.4f} | ln(2) = {math.log(2):.4f}")


def check_overfit(normal_loader, anomal_loader, args):
    normal_feat, normal_class, normal_len = next(iter(normal_loader))
    anomal_feat, anomal_class, anomal_len = next(iter(anomal_loader))
    n = OVERFIT_VIDEOS_PER_GROUP
    features = torch.cat([normal_feat[:n], anomal_feat[:n]])
    classes = torch.cat([normal_class[:n], anomal_class[:n]])
    lengths = torch.cat([normal_len[:n], anomal_len[:n]])

    model = seeded_model(args).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=OVERFIT_LR, weight_decay=0.0)
    print(f"\n=== 2. Overfit {2 * n} rows, labels {classes.tolist()} ===")
    for step in range(OVERFIT_STEPS + 1):
        loss = batch_loss(model, features, classes, lengths, args)
        if step % 100 == 0:
            print(f"step {step:>3}: loss {loss.item():.4f}")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def write_shuffled_list(args):
    """Shuffle labels between videos, keeping all crops of a video on the same label."""
    df = pd.read_csv(args.train_list)
    videos = df["path"].str.replace(r"__\d+\.npy$", "", regex=True)
    video_labels = df.groupby(videos, sort=False)["label"].first()
    shuffled = np.random.default_rng(args.seed).permutation(video_labels.to_numpy())
    new_labels = dict(zip(video_labels.index, shuffled))

    out_path = Path(args.run_root) / "shuffled_train_list.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.assign(label=videos.map(new_labels)).to_csv(out_path, index=False)

    swapped = ((video_labels == "Normal") != (shuffled == "Normal")).mean()
    print("\n=== 3. Shuffled labels ===")
    print(f"{swapped:.1%} of videos switched between normal and anomalous. List saved to {out_path}")
    print(f"Next: python train.py --train-list {out_path} --run-name shuffled_labels")


def main():
    args = option.parser.parse_args()
    normal_loader = build_loader(args, normal=True)
    anomal_loader = build_loader(args, normal=False)

    check_initial_loss(normal_loader, anomal_loader, args)
    check_overfit(normal_loader, anomal_loader, args)
    write_shuffled_list(args)


if __name__ == "__main__":
    main()
