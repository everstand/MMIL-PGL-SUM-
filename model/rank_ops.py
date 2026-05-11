import torch


def soft_rank(x, tau=0.1):
    x = x.view(-1)
    diff = x.unsqueeze(0) - x.unsqueeze(1)
    P = torch.sigmoid(-diff / tau)
    r = 1.0 + P.sum(dim=1) - 0.5
    return r


def spearman_rank_loss(pred, target, mask, tau=0.1, eps=1e-6):
    pred = pred.view(-1)
    target = target.view(-1)
    mask = mask.view(-1).bool()

    pred = pred[mask]
    target = target[mask]

    if pred.numel() < 2:
        return pred.new_tensor(0.0)

    r_pred = soft_rank(pred, tau=tau)
    r_tgt = soft_rank(target.detach(), tau=tau)

    r_pred = (r_pred - r_pred.mean()) / (r_pred.std(unbiased=False) + eps)
    r_tgt = (r_tgt - r_tgt.mean()) / (r_tgt.std(unbiased=False) + eps)

    return ((r_pred - r_tgt) ** 2).mean()
