"""OGM-GE 在线梯度调制（CVPR 2022，检测任务适配版）。

原论文按模态得分比 rho 计算调制系数 k = 1 - tanh(alpha * relu(rho))，
强势模态衰减梯度（OGM），弱势模态叠加自适应高斯噪声（GE）。
检测任务没有单模态分类头，这里用双分支梯度 L2 范数比作为
模态收敛速度的代理：rho = ||grad_rgb|| / ||grad_aux||。
只调制 4D 卷积层梯度（与原论文一致）。
"""

import torch


def _branch_grad_norm(model, prefix, eps=1e-8):
    total = 0.0
    for name, p in model.named_parameters():
        if name.startswith(prefix) and p.grad is not None:
            total += p.grad.detach().float().pow(2).sum().item()
    return total ** 0.5 + eps


def modulate_gradients_ogm_ge(model, alpha=1.0, use_ge=True):
    grad_rgb = _branch_grad_norm(model, "backbone_rgb")
    grad_aux = _branch_grad_norm(model, "backbone_aux")
    ratio = grad_rgb / grad_aux

    if ratio > 1.0:
        coeff_rgb = 1.0 - torch.tanh(torch.tensor(alpha * ratio, dtype=torch.float32))
        coeff_aux = 1.0
    else:
        coeff_aux = 1.0 - torch.tanh(torch.tensor(alpha / ratio, dtype=torch.float32))
        coeff_rgb = 1.0
    coeff_rgb = float(coeff_rgb)
    coeff_aux = float(coeff_aux)

    for name, p in model.named_parameters():
        if p.grad is None or p.dim() != 4:
            continue
        if name.startswith("backbone_rgb"):
            coeff = coeff_rgb
        elif name.startswith("backbone_aux"):
            coeff = coeff_aux
        else:
            continue
        if use_ge:
            noise = torch.zeros_like(p.grad).normal_(0.0, p.grad.std().item() + 1e-8)
            p.grad = p.grad * coeff + noise
        else:
            p.grad.mul_(coeff)
    return ratio, coeff_rgb, coeff_aux
