from __future__ import annotations

import torch


@torch.no_grad()
def recall_at_k(logits: torch.Tensor, k: int) -> float:
    """
    logits: [B, B], где правильная пара i соответствует цели i (диагональ).
    Recall@k = доля примеров, где правильный индекс попал в top-k.
    """
    b = logits.size(0)
    target = torch.arange(b, device=logits.device)
    topk = logits.topk(k, dim=1).indices  # [B, k]
    ok = (topk == target.unsqueeze(1)).any(dim=1).float().mean()
    return float(ok.item())


@torch.no_grad()
def retrieval_metrics(logits_per_image: torch.Tensor, logits_per_text: torch.Tensor, ks=(1, 5, 10)) -> dict:
    out = {}
    for k in ks:
        out[f"i2t_R@{k}"] = recall_at_k(logits_per_image, k=min(k, logits_per_image.size(0)))
        out[f"t2i_R@{k}"] = recall_at_k(logits_per_text,  k=min(k, logits_per_text.size(0)))
    return out
