"""Inspect what a trained checkpoint scores, compared with an untrained model."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import option
import pandas as pd
import torch
from dataset import one_crop_per_video, process_feat
from evaluation import build_test_loader, predict
from loss import topk_mean
from model import Model
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from utils.classes import CLASS_TO_INDEX

RANDOM_INITS = 50
NORMAL_CURVES = 3
LENGTH_BINS = [0, 256, 1000, np.inf]
LENGTH_NAMES = ["up to 256", "257-1000", "over 1000"]

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED_INK = "#52514e"
NORMAL_COLOR = "#898781"
UNANNOTATED_COLOR = "#2a78d6"
ANNOTATED_COLOR = "#eb6834"


def video_frames(test_batches, args):
    """Frame ranges of each test video inside the concatenated scores."""
    rows = []
    start = 0
    for _, label, lengths, path in test_batches:
        end = start + int(lengths.sum()) * args.frames_per_segment
        rows.append({"label": label[0], "path": path[0], "start": start, "end": end})
        start = end
    return pd.DataFrame(rows)


def report_random_inits(test_batches, frame_labels, args):
    aucs = []
    for seed in range(RANDOM_INITS):
        torch.manual_seed(seed)
        model = Model(args).to(args.device)
        aucs.append(roc_auc_score(frame_labels, predict(model, test_batches, args)))
    aucs = np.array(aucs)
    print(f"\n=== A. Frame AUC of {RANDOM_INITS} untrained models ===")
    print(f"mean {aucs.mean():.4f}, std {aucs.std():.4f}, min {aucs.min():.4f}, max {aucs.max():.4f}")
    print(f"Models with AUC below 0.5: {(aucs < 0.5).mean():.0%}")


def report_auc_breakdown(frame_labels, scores, normal_frames, args):
    annotated = frame_labels == 1
    unannotated = (frame_labels == 0) & ~normal_frames
    in_anomalous_video = ~normal_frames

    def group_auc(positive, negative):
        keep = positive | negative
        return roc_auc_score(positive[keep], scores[keep])

    rows = {
        "all frames": roc_auc_score(frame_labels, scores),
        "anomalous videos only": roc_auc_score(annotated[in_anomalous_video], scores[in_anomalous_video]),
        "annotated vs normal-video frames": group_auc(annotated, normal_frames),
        "unannotated in anomalous videos vs normal-video frames": group_auc(unannotated, normal_frames),
    }
    print("\n=== B. Frame AUC breakdown ===")
    print(pd.Series(rows, name="auc").round(4).to_string())


def report_per_class(videos, frame_labels, scores, normal_frames):
    annotated = frame_labels == 1
    rows = []
    for label in list(CLASS_TO_INDEX)[1:]:
        class_frames = np.zeros(len(scores), dtype=bool)
        for start, end in zip(*videos.loc[videos["label"] == label, ["start", "end"]].to_numpy().T):
            class_frames[start:end] = True
        positive = class_frames & annotated
        keep = positive | normal_frames
        rows.append({
            "class": label,
            "videos": (videos["label"] == label).sum(),
            "auc_vs_normal_videos": roc_auc_score(positive[keep], scores[keep]),
            "auc_within_own_videos": roc_auc_score(annotated[class_frames], scores[class_frames]),
            "mean_score_annotated": scores[positive].mean(),
        })
    print("\n=== C. Per-class frame AUC ===")
    print(pd.DataFrame(rows).set_index("class").round(4).to_string())


def plot_score_distributions(frame_labels, scores, untrained_scores, normal_frames, out_path):
    groups = [
        ("Frames in normal videos", normal_frames, NORMAL_COLOR),
        ("Unannotated frames in anomalous videos", (frame_labels == 0) & ~normal_frames, UNANNOTATED_COLOR),
        ("Annotated frames", frame_labels == 1, ANNOTATED_COLOR),
    ]
    bins = np.linspace(0, 1, 51)
    print("\n=== D. Score quantiles ===")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharex=True, facecolor=SURFACE)
    for ax, values, title in zip(axes, [untrained_scores, scores], ["Untrained", "Trained"]):
        ax.set_facecolor(SURFACE)
        for name, mask, color in groups:
            ax.hist(values[mask], bins=bins, density=True, histtype="step", linewidth=2, color=color, label=name)
            q10, q50, q90 = np.quantile(values[mask], [0.1, 0.5, 0.9])
            print(f"{title:<10} {name:<40} p10 {q10:.3f} | median {q50:.3f} | p90 {q90:.3f}")
        ax.set_title(title, loc="left", color=INK)
        ax.set_xlabel("Anomaly score", color=MUTED_INK)
        ax.set_yticks([])
        ax.tick_params(colors=MUTED_INK)
        ax.spines[["top", "right", "left"]].set_visible(False)
    axes[1].legend(frameon=False, labelcolor=INK)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def report_length_trend(videos, scores, args):
    normal = videos[videos["label"] == "Normal"]
    rows = pd.DataFrame({
        "snippets": (normal["end"] - normal["start"]).to_numpy() // args.frames_per_segment,
        "mean_score": [scores[start:end].mean() for start, end in zip(normal["start"], normal["end"])],
        "share_above_0.5": [(scores[start:end] > 0.5).mean() for start, end in zip(normal["start"], normal["end"])],
    })
    rho, p_value = spearmanr(rows["snippets"], rows["mean_score"])
    table = rows.groupby(pd.cut(rows["snippets"], LENGTH_BINS, labels=LENGTH_NAMES), observed=False).agg(
        videos=("snippets", "size"), mean_score=("mean_score", "mean"), share_above_05=("share_above_0.5", "mean")
    )
    print("\n=== E. Normal test videos: score vs length ===")
    print(f"Spearman correlation of length and mean score: {rho:.3f} (p = {p_value:.3g})")
    print(table.round(4).to_string())


def plot_score_curves(videos, frame_labels, scores, out_path):
    anomalous = videos[videos["label"] != "Normal"].groupby("label", sort=False).head(1)
    chosen = pd.concat([anomalous, videos[videos["label"] == "Normal"].head(NORMAL_CURVES)])
    columns = 4
    rows = int(np.ceil(len(chosen) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(14, 2.3 * rows), sharey=True, facecolor=SURFACE)
    for ax in axes.flat:
        ax.set_visible(False)
    for ax, video in zip(axes.flat, chosen.itertuples()):
        ax.set_visible(True)
        ax.set_facecolor(SURFACE)
        frames = np.arange(video.end - video.start)
        ax.fill_between(frames, 0, 1, where=frame_labels[video.start:video.end] == 1, color=ANNOTATED_COLOR,
                        alpha=0.22, linewidth=0, label="Annotated")
        ax.plot(frames, scores[video.start:video.end], color=UNANNOTATED_COLOR, linewidth=1.5, label="Score")
        ax.set_title(Path(video.path).stem.split("_x264")[0], loc="left", fontsize=10, color=INK)
        ax.set_ylim(0, 1)
        ax.tick_params(colors=MUTED_INK, labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.supxlabel("Frame", color=MUTED_INK)
    fig.supylabel("Anomaly score", color=MUTED_INK)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", frameon=False, ncols=2)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def video_auc(model, df, args):
    """Video-level AUC, using the same preprocessing and top-k score as training."""
    model.eval()
    scores = []
    for path in df["path"]:
        feature, length = process_feat(np.load(path).astype(np.float32), args.seq_len)
        logits = model(torch.from_numpy(feature)[None].to(args.device))
        lengths = torch.tensor([length]).to(args.device)
        scores.append(topk_mean(logits, lengths, args.mil_k_ratio).item())
    return roc_auc_score(df["label"] != "Normal", scores)


def report_generalization(model, args):
    train_auc = video_auc(model, one_crop_per_video(pd.read_csv(args.train_list)), args)
    test_auc = video_auc(model, one_crop_per_video(pd.read_csv(args.test_list)), args)
    print("\n=== F. Video-level AUC, train preprocessing on both splits ===")
    print(f"train {train_auc:.4f} | test {test_auc:.4f} | gap {train_auc - test_auc:.4f}")


def main():
    args = option.parser.parse_args()
    if args.checkpoint is None:
        option.parser.error("--checkpoint is required")
    out_dir = Path(args.checkpoint).parent / "debug"
    out_dir.mkdir(exist_ok=True)

    test_batches = list(build_test_loader(args))
    videos = video_frames(test_batches, args)
    frame_labels = np.load(args.gt_path)
    if videos["end"].iloc[-1] != len(frame_labels):
        raise ValueError(f"Scores cover {videos['end'].iloc[-1]} frames, ground truth has {len(frame_labels)}")

    normal_frames = np.zeros(len(frame_labels), dtype=bool)
    for video in videos[videos["label"] == "Normal"].itertuples():
        normal_frames[video.start:video.end] = True

    torch.manual_seed(args.seed)
    untrained = Model(args).to(args.device)
    untrained_scores = predict(untrained, test_batches, args)
    model = Model(args).to(args.device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
    scores = predict(model, test_batches, args)

    print(f"Frame AUC with seed {args.seed}: untrained {roc_auc_score(frame_labels, untrained_scores):.4f} | "
          f"trained {roc_auc_score(frame_labels, scores):.4f}")
    report_random_inits(test_batches, frame_labels, args)
    report_auc_breakdown(frame_labels, scores, normal_frames, args)
    report_per_class(videos, frame_labels, scores, normal_frames)
    plot_score_distributions(frame_labels, scores, untrained_scores, normal_frames, out_dir / "score_distributions.png")
    report_length_trend(videos, scores, args)
    report_generalization(model, args)
    plot_score_curves(videos, frame_labels, scores, out_dir / "score_curves.png")
    print(f"\nPlots saved to {out_dir}")


if __name__ == "__main__":
    main()
