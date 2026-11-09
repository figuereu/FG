import numpy as np
import option
import torch
from dataset import UCFDataset
from model import Model
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader


def build_test_loader(args) -> DataLoader:
    dataset = UCFDataset(
        clip_dim=args.seq_len,
        file_path=args.test_list,
        test_mode=True,
        normal=False,
    )
    # Batch size 1: videos are split into a variable number of chunks.
    return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)

@torch.no_grad()
def predict(model, loader, args) -> np.ndarray:
    model.eval()
    scores = []

    for feature, _, lengths, _ in loader:
        logits = model(feature.squeeze(0).to(args.device))
        # Only the last chunk is padded, so the valid steps come first once flattened.
        snippet_scores = logits.sigmoid().flatten()[:int(lengths.sum())]
        scores.append(np.repeat(snippet_scores.cpu().numpy(), args.frames_per_segment))

    return np.concatenate(scores)

def evaluate(model, loader, frame_labels: np.ndarray, args) -> float:
    return roc_auc_score(frame_labels, predict(model, loader, args))

def main():
    args = option.parser.parse_args()
    if args.checkpoint is None:
        option.parser.error("--checkpoint is required to evaluate a model")

    model = Model(args).to(args.device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))

    auc = evaluate(model, build_test_loader(args), np.load(args.gt_path), args)
    print(f"frame-level AUC: {auc:.4f}")

if __name__ == "__main__":
    main()
