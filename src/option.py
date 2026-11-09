import argparse

parser = argparse.ArgumentParser(description='FGVAD')

parser.add_argument('--seed', default=234, type=int)
parser.add_argument('--device', default='cuda:0', type=str)
parser.add_argument('--num-workers', dest='num_workers', default=4, type=int)

# Dataset
parser.add_argument('--train-list', dest='train_list', default='../list/ucf_CLIP_rgb.csv')
parser.add_argument('--test-list', dest='test_list', default='../list/ucf_CLIP_rgbtest.csv')
parser.add_argument('--gt-path', dest='gt_path', default='../list/gt_ucf.npy')
parser.add_argument('--gt-segment-path', dest='gt_segment_path', default='../list/gt_segment_ucf.npy')
parser.add_argument('--gt-label-path', dest='gt_label_path', default='../list/gt_label_ucf.npy')
parser.add_argument('--seq-len', dest='seq_len', default=256, type=int)
parser.add_argument('--frames-per-segment', dest='frames_per_segment', default=16, type=int)

# Model
parser.add_argument('--video-feature-dim', dest='video_feature_dim', default=512, type=int)

# Loss
parser.add_argument('--mil-k-ratio', dest='mil_k_ratio', default=16, type=int)

# Training
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--batch-size', dest='batch_size', default=64, type=int)
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--weight-decay', dest='weight_decay', default=1e-2, type=float)
parser.add_argument('--evals-per-epoch', dest='evals_per_epoch', default=12, type=int)
parser.add_argument('--restore-best', dest='restore_best', default=False, type=bool)
parser.add_argument('--run-root', dest='run_root', default='../runs')
parser.add_argument('--run-name', dest='run_name', default=None)

# Evaluation
parser.add_argument('--checkpoint', default=None)
