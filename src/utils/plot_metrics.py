import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot AUC and training loss from epoch 1, step 10 onward."
    )
    parser.add_argument("csv_path", type=Path, help="Path to the metrics CSV file")
    parser.add_argument(
        "-o", "--output", type=Path,
        help="Optional path where the plot should be saved (for example, metrics.png)",
    )
    parser.add_argument("--epoch-col", default="epoch", help="Epoch column name (default: epoch)")
    parser.add_argument("--step-col", default="step", help="Step column name (default: step)")
    parser.add_argument("--auc-col", default="auc", help="AUC column name (default: auc)")
    parser.add_argument(
        "--loss-col", default="train_loss",
        help="Loss column name (default: train_loss)",
    )
    args = parser.parse_args()

    if not args.csv_path.is_file():
        parser.error(f"CSV file not found: {args.csv_path}")

    df = pd.read_csv(args.csv_path)
    # Ignore accidental whitespace around CSV header names.
    df.columns = df.columns.astype(str).str.strip()

    required = [args.epoch_col, args.step_col, args.auc_col, args.loss_col]
    missing = [col for col in required if col not in df.columns]
    if missing:
        parser.error(
            f"Missing column(s): {', '.join(missing)}. "
            f"Available columns: {', '.join(df.columns)}"
        )

    # Convert the relevant columns to numeric values, then keep the requested range.
    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    start_mask = (df[args.epoch_col] > 1) | (
        (df[args.epoch_col] == 1) & (df[args.step_col] >= 10)
    )
    df = df.loc[start_mask].dropna(subset=required)
    df = df.sort_values([args.epoch_col, args.step_col], kind="stable")

    if df.empty:
        parser.error("No valid rows found from epoch 1, step 10 onward.")

    # The step column resets each epoch in the supplied CSV. Position each point
    # within its epoch so that the x-axis shows continuous training progress.
    max_step = df[args.step_col].max()
    step_scale = max_step + 1 if max_step >= 0 else 1
    x = df[args.epoch_col] + df[args.step_col] / step_scale

    fig, (ax_auc, ax_loss) = plt.subplots(
        2, 1, figsize=(11, 7), sharex=True, constrained_layout=True
    )

    ax_auc.plot(x, df[args.auc_col], marker="o", markersize=3, linewidth=1.5)
    ax_auc.set_title("AUC")
    ax_auc.set_ylabel("AUC")
    ax_auc.grid(True, alpha=0.3)

    ax_loss.plot(x, df[args.loss_col], marker="o", markersize=3, linewidth=1.5)
    ax_loss.set_title("Training loss")
    ax_loss.set_ylabel("Loss")
    ax_loss.grid(True, alpha=0.3)

    # Put one tick near the middle of each epoch represented in the data.
    epoch_ticks = []
    epoch_labels = []
    for epoch, group in df.groupby(args.epoch_col, sort=True):
        middle_step = (group[args.step_col].min() + group[args.step_col].max()) / 2
        epoch_ticks.append(epoch + middle_step / step_scale)
        epoch_labels.append(str(int(epoch)) if float(epoch).is_integer() else str(epoch))

    ax_loss.set_xticks(epoch_ticks, epoch_labels)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=200, bbox_inches="tight")
        print(f"Plot saved to: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
