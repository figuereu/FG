import numpy as np
import option
import pandas as pd
import torch
from torch.utils import data
from torch.utils.data import DataLoader
from utils.classes import CLASS_TO_INDEX


def uniform_extract(feat, t_max):
    edges = np.linspace(0, len(feat), t_max + 1, dtype=np.int32)
    sums = np.add.reduceat(feat, edges[:-1], axis=0)
    return (sums / np.diff(edges)[:, None]).astype(np.float32)

def pad(feat, length):
    return np.pad(feat, ((0, length - feat.shape[0]), (0, 0)))

def process_feat(feat, length):
    if feat.shape[0] > length:
        return uniform_extract(feat, length), length
    return pad(feat, length), feat.shape[0]

def process_split(feat, length):
    chunks = [feat[i:i + length] for i in range(0, feat.shape[0], length)]
    lengths = np.array([len(chunk) for chunk in chunks])
    chunks[-1] = pad(chunks[-1], length)
    return np.stack(chunks), lengths

class UCFDataset(data.Dataset):

    def __init__(self, clip_dim: int, file_path: str, test_mode: bool, normal: bool = False):
        df = pd.read_csv(file_path)
        if not test_mode:
            is_normal = df["label"] == "Normal"
            df = df[is_normal if normal else ~is_normal]

        self.df = df.reset_index(drop=True)
        self.clip_dim = clip_dim
        self.test_mode = test_mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        feature = np.load(row["path"]).astype(np.float32)

        if self.test_mode:
            feature, lengths = process_split(feature, self.clip_dim)
            return torch.from_numpy(feature), row["label"], torch.from_numpy(lengths), row["path"]

        feature, length = process_feat(feature, self.clip_dim)
        return torch.from_numpy(feature), CLASS_TO_INDEX[row["label"]], length

def one_crop_per_video(df):
    videos = df["path"].str.replace(r"__\d+\.npy$", "", regex=True)
    return df[~videos.duplicated()]

def summarize_split(name, df, seq_len):
    videos = one_crop_per_video(df)
    shapes = [np.load(path, mmap_mode="r").shape for path in videos["path"]]
    lengths = np.array([shape[0] for shape in shapes])
    table = (
        videos.assign(snippets=lengths)
        .groupby("label")
        .agg(videos=("snippets", "size"), snippets=("snippets", "sum"), mean_snippets=("snippets", "mean"))
        .join(df.groupby("label").size().rename("rows"))
        .reindex(list(CLASS_TO_INDEX))
        .round({"mean_snippets": 1})
    )
    table.insert(0, "index", list(CLASS_TO_INDEX.values()))

    print(f"\n=== {name} data ===")
    print(f"Rows: {len(df)} | Videos: {len(videos)} | Rows per video: {len(df) / len(videos):g}")
    print(f"Feature dims: {sorted({shape[1] for shape in shapes})}")
    print(f"Snippets per video: min {lengths.min()}, mean {lengths.mean():.1f}, "
          f"median {np.median(lengths):.0f}, max {lengths.max()}, total {lengths.sum()}")
    print(f"Videos longer than seq_len={seq_len}: {(lengths > seq_len).sum()}")
    print(table.to_string())
    return int(lengths.sum())

if __name__ == '__main__':

    from utils.tsne import annotated_features, plot_tsne, video_features

    args = option.parser.parse_args()

    normal_dataset = UCFDataset(clip_dim=args.seq_len, file_path=args.train_list, test_mode=False, normal=True)
    normal_loader = DataLoader(normal_dataset, batch_size=64, shuffle=True, drop_last=True)
    anomal_dataset = UCFDataset(clip_dim=args.seq_len, file_path=args.train_list, test_mode=False, normal=False)
    anomal_loader = DataLoader(anomal_dataset, batch_size=64, shuffle=True, drop_last=True)
    test_dataset = UCFDataset(clip_dim=args.seq_len, file_path=args.test_list, test_mode=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    for i, (feat, label, length) in enumerate(normal_loader):
        print(f"Normal sample {i}: feat shape: {feat.shape}, label: {label}, length: {length}")
    for i, (feat, label, length) in enumerate(anomal_loader):
        print(f"Anomal sample {i}: feat shape: {feat.shape}, label: {label}, length: {length}")

    feat, label, lengths, path = next(iter(test_loader))
    print(f"Test sample 0: feat shape: {feat.shape}, label: {label}, lengths: {lengths}, path: {path}")

    train_df = pd.concat([normal_dataset.df, anomal_dataset.df])
    summarize_split("Train", train_df, args.seq_len)
    test_snippets = summarize_split("Test", test_dataset.df, args.seq_len)
    print(f"Frame-level GT: {len(np.load(args.gt_path))} frames | Test snippets x 16: {test_snippets * 16}")

    train_videos = one_crop_per_video(train_df)
    test_videos = one_crop_per_video(test_dataset.df)
    plot_tsne(video_features(train_videos), train_videos["label"], "Train data: t-SNE by class",
              "All frames averaged.", "tsne_train.png")
    plot_tsne(video_features(test_videos), test_videos["label"], "Test data: t-SNE by class",
              "All frames averaged.", "tsne_test.png")

    gt_segments = np.load(args.gt_segment_path, allow_pickle=True)
    gt_labels = np.load(args.gt_label_path, allow_pickle=True)
    plot_tsne(annotated_features(test_videos, gt_segments, gt_labels), test_videos["label"],
              "Test data: t-SNE by class, annotated frames only",
              "Anomalous videos average annotated frames; normal videos average all frames.",
              "tsne_test_annotated.png")
