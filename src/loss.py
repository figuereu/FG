import torch
import torch.nn.functional as F


def topk_mean(logits: torch.Tensor, lengths: torch.Tensor, k_ratio: int) -> torch.Tensor:

    steps = torch.arange(logits.shape[1], device=logits.device)
    valid = steps[None, :] < lengths[:, None]
    logits = logits.masked_fill(~valid, torch.finfo(logits.dtype).min)

    k = lengths // k_ratio + 1
    top, _ = logits.topk(int(k.max()), dim=1)

    selected = torch.arange(top.shape[1], device=logits.device)[None, :] < k[:, None]
    return (top * selected).sum(dim=1) / k

def mil_loss(logits: torch.Tensor, labels: torch.Tensor, lengths: torch.Tensor, k_ratio: int = 16,) -> torch.Tensor:

    video_logits = topk_mean(logits, lengths, k_ratio)
    return F.binary_cross_entropy_with_logits(video_logits, labels)

if __name__ == "__main__":

    batch, clip_dim = 4, 256
    lengths = torch.tensor([256, 200, 64, 16])
    labels = torch.tensor([1.0, 1.0, 0.0, 0.0])

    logits = torch.randn(batch, clip_dim, requires_grad=True)
    print(f"random logits -> loss: {mil_loss(logits, labels, lengths).item():.4f}")

    perfect = torch.full((batch, clip_dim), -10.0)
    perfect[:2, :10] = 10.0
    print(f"perfect logits -> loss: {mil_loss(perfect, labels, lengths).item():.4f}")

    padded = perfect.clone()
    padded[3, 16:] = 10.0
    print(f"padded frames  -> loss: {mil_loss(padded, labels, lengths).item():.4f}")
