import random

import numpy as np
import option
import torch
from torch import nn


class Model(nn.Module):

    def __init__(self, args):

        super().__init__()

        self.binary_classifier = nn.Sequential(
            nn.Linear(args.video_feature_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:

        return self.binary_classifier(features).squeeze(-1)

if __name__ == "__main__":

    args = option.parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    model = Model(args).to(args.device)
    print(model)

    print(f"params: {sum(p.numel() for p in model.parameters())}")
