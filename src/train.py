import csv
import json
import logging
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import option
import torch
from dataset import UCFDataset
from evaluation import build_test_loader, evaluate
from loss import mil_loss
from model import Model
from torch.utils.data import DataLoader
from tqdm import tqdm


def build_loader(args, normal: bool) -> DataLoader:
    dataset = UCFDataset(
        clip_dim=args.seq_len,
        file_path=args.train_list,
        test_mode=False,
        normal=normal,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )

def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class TqdmHandler(logging.StreamHandler):

    def emit(self, record):
        tqdm.write(self.format(record))

class Run:

    def __init__(self, args, model):
        name = args.run_name or datetime.now().strftime("%Y%m%d-%H%M%S")
        self.dir = Path(args.run_root) / name
        self.dir.mkdir(parents=True, exist_ok=False)
        self.best_checkpoint = self.dir / "best.pth"
        self.metrics_path = self.dir / "metrics.csv"

        self.logger = logging.getLogger(f"run.{name}")
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers.clear()

        console = TqdmHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(logging.Formatter("%(message)s"))

        log_file = logging.FileHandler(self.dir / "train.log")
        log_file.setLevel(logging.DEBUG)
        log_file.setFormatter(logging.Formatter("%(asctime)s | %(message)s", "%H:%M:%S"))

        self.logger.addHandler(console)
        self.logger.addHandler(log_file)

        with self.metrics_path.open("w", newline="") as f:
            csv.writer(f).writerow(["epoch", "step", "train_loss", "auc", "best"])

        (self.dir / "config.json").write_text(json.dumps(vars(args), indent=2))
        (self.dir / "model.txt").write_text(f"{model}\n\nparameters: {count_params(model)}\n")

        self.log(f"{model}\ntrainable parameters: {count_params(model)}")
        self.log(f"run: {self.dir}\n")
        self.log_file(f"config {json.dumps(vars(args))}")

    def log(self, message: str):
        self.logger.info(message)

    def log_file(self, message: str):
        self.logger.debug(message)

    def record(self, epoch: int, step: int, loss: float, auc: float, best: bool):
        with self.metrics_path.open("a", newline="") as f:
            csv.writer(f).writerow([epoch, step, f"{loss:.6f}", f"{auc:.6f}", int(best)])

class Trainer:

    def __init__(self, model, optimizer, loaders, frame_labels, run, args):
        self.model = model
        self.optimizer = optimizer
        self.normal_loader, self.anomal_loader, self.test_loader = loaders
        self.frame_labels = frame_labels
        self.run = run
        self.args = args

        self.steps_per_epoch = min(len(self.normal_loader), len(self.anomal_loader))
        self.eval_steps = {
            round(self.steps_per_epoch * i / args.evals_per_epoch)
            for i in range(1, args.evals_per_epoch + 1)
        }
        self.best_auc = 0.0

    def training_step(self, batch) -> float:
        (normal_feat, normal_class, normal_len), (anomal_feat, anomal_class, anomal_len) = batch
        features = torch.cat([normal_feat, anomal_feat]).to(self.args.device)
        lengths = torch.cat([normal_len, anomal_len]).to(self.args.device)
        classes = torch.cat([normal_class, anomal_class]).to(self.args.device)

        logits = self.model(features)
        loss = mil_loss(logits, (classes > 0).float(), lengths, self.args.mil_k_ratio)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def validate(self, epoch: int, step: int, loss: float):
        auc = evaluate(self.model, self.test_loader, self.frame_labels, self.args)
        improved = auc > self.best_auc
        if improved:
            self.best_auc = auc
            torch.save(self.model.state_dict(), self.run.best_checkpoint)

        self.run.record(epoch, step, loss, auc, improved)
        self.run.log(
            f"epoch {epoch:>3}/{self.args.epochs} | step {step:>3}/{self.steps_per_epoch} | "
            f"loss {loss:.4f} | auc {auc:.4f}"
            + ("  *" if improved else "")
        )

    def train_epoch(self, epoch: int):
        self.model.train()
        losses = []
        batches = tqdm(
            zip(self.normal_loader, self.anomal_loader),
            total=self.steps_per_epoch,
            desc=f"epoch {epoch}/{self.args.epochs}",
            leave=False,
        )

        for step, batch in enumerate(batches, start=1):
            losses.append(self.training_step(batch))
            batches.set_postfix(loss=f"{losses[-1]:.4f}")

            if step in self.eval_steps:
                self.validate(epoch, step, float(np.mean(losses)))
                self.model.train()

    def fit(self):
        self.run.log(
            f"{self.steps_per_epoch} steps per epoch, {self.args.epochs} epochs, "
            f"evaluating at steps {sorted(self.eval_steps)}\n"
        )
        self.validate(epoch=0, step=0, loss=float("nan"))  # before any training

        for epoch in range(1, self.args.epochs + 1):
            self.train_epoch(epoch)
            if self.args.restore_best:
                state = torch.load(self.run.best_checkpoint, map_location=self.args.device)
                self.model.load_state_dict(state)
                self.run.log(f"restored best checkpoint (auc {self.best_auc:.4f})")

        self.run.log(f"\nbest auc {self.best_auc:.4f} saved to {self.run.best_checkpoint}")

def main():
    args = option.parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    model = Model(args).to(args.device)

    run = Run(args, model)
    loaders = (
        build_loader(args, normal=True),
        build_loader(args, normal=False),
        build_test_loader(args),
    )

    trainer = Trainer(
        model=model,
        optimizer=torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay),
        loaders=loaders,
        frame_labels=np.load(args.gt_path),
        run=run,
        args=args,
    )
    trainer.fit()

if __name__ == "__main__":
    main()
