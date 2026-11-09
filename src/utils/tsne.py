import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE
from utils.classes import CLASS_TO_INDEX

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED_INK = "#52514e"
SNIPPET_FRAMES = 16

CLASS_COLORS = {
    "Normal": "#898781",
    "Abuse": "#6622bb",
    "Arrest": "#ddaa11",
    "Arson": "#006633",
    "Assault": "#33ccbb",
    "Burglary": "#2288ff",
    "Explosion": "#ee0099",
    "Fighting": "#0000ff",
    "RoadAccidents": "#99aaee",
    "Robbery": "#669900",
    "Shooting": "#bb0000",
    "Shoplifting": "#9966cc",
    "Stealing": "#990077",
    "Vandalism": "#ee7777",
}


def video_features(df):
    return np.stack([np.load(path).mean(axis=0) for path in df["path"]])


def annotated_features(df, gt_segments, gt_labels):
    features = []
    for path, label, segments, segment_labels in zip(df["path"], df["label"], gt_segments, gt_labels):
        if set(segment_labels) != {"A" if label == "Normal" else label}:
            raise ValueError(f"{path} is labeled {label}, but the ground truth says {segment_labels}")
        feature = np.load(path)
        keep = np.zeros(len(feature), dtype=bool)
        for start, end in segments:
            keep[int(start) // SNIPPET_FRAMES : math.ceil(int(end) / SNIPPET_FRAMES)] = True
        features.append(feature[keep].mean(axis=0))
    return np.stack(features)


def plot_tsne(features, labels, title, description, out_path):
    points = TSNE(metric="cosine", init="pca", random_state=0).fit_transform(features)
    labels = np.asarray(labels)

    fig, ax = plt.subplots(figsize=(10, 7), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for label in CLASS_TO_INDEX:
        mask = labels == label
        color = CLASS_COLORS[label]
        ax.scatter(
            points[mask, 0],
            points[mask, 1],
            c=color,
            marker="o",
            s=30,
            alpha=0.75,
            edgecolors=SURFACE,
            linewidths=0.5,
            label=f"{label} ({mask.sum()})",
            zorder=1,
        )
        centroid = points[mask].mean(axis=0)
        ax.scatter(*centroid, c=color, marker="*", s=340, edgecolors=INK, linewidths=0.9, zorder=2)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[:].set_visible(False)

    handles, _ = ax.get_legend_handles_labels()
    centroid_handle = Line2D(
        [], [], linestyle="none", marker="*", markersize=13, markerfacecolor=SURFACE,
        markeredgecolor=INK, label="Class centroid",
    )
    legend = ax.legend(
        handles=[*handles, centroid_handle],
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=False,
        labelcolor=INK,
    )
    for handle in legend.legend_handles:
        handle.set_alpha(1)

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved t-SNE to {out_path}")
