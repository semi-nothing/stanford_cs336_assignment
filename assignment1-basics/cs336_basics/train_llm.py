
import math
import torch
import torch.nn as nn
from jaxtyping import Float, Int, Bool

from typing import Optional
from collections.abc import Callable, Iterable


def cross_entropy(logits: Float[torch.Tensor, "... S V"], 
                    labels: Int[torch.Tensor, "... S"]) -> Float[torch.Tensor, ""]:
    """
    Compute the cross-entropy loss between the logits and the labels.
    
    Args:
        logits: The logits of the model.
        labels: The labels of the model.

    Returns:
        The cross-entropy loss.
    """
    logits = logits - torch.max(logits, dim=-1, keepdim=True).values
    log_softmax = logits - torch.log(torch.sum(torch.exp(logits), dim=-1, keepdim=True)) # B S V
    log_prob= torch.gather(log_softmax, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    loss = -log_prob.mean()
    return loss


def perplexity(loss: Float[torch.Tensor, ""]) -> Float[torch.Tensor, ""]:
    """
    Compute the perplexity from the cross-entropy loss.

    Args:
        loss: The cross-entropy loss.

    Returns:
        The perplexity.
    """
    return torch.exp(loss)



class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr: float=1e-3) -> None:
        if lr < 0:
            raise ValueError("Learning rate must be positive.")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable]=None) -> Float[torch.Tensor, ""]:
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                state = self.state[p]
                t = state.get("t", 0)
                grad = p.grad.data
                p.data -= (lr / math.sqrt(t+1)) * grad
                state["t"] = t+1
        return loss

class AdamW(torch.optim.Optimizer):

    def __init__(self, 
                params,
                lr: float=1e-3,
                beta1: float=0.9,
                beta2: float=0.999,
                weight_decay: float=0,
                eps: float=1e-8) -> None:
        if lr <= 0:
            raise ValueError("Learning rate must be positive.")
        if beta1 <= 0 or beta1 >= 1:
            raise ValueError("Beta1 must be between 0 and 1.")
        if beta2 <= 0 or beta2 >= 1:
            raise ValueError("Beta2 must be between 0 and 1.")
        if weight_decay < 0:
            raise ValueError("Weight decay must be non-negative.")
        if eps <= 0:
            raise ValueError("Epsilon must be positive.")
        defaults = {"lr": lr, "beta1": beta1, "beta2": beta2, "weight_decay": weight_decay, "eps": eps}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable]=None) -> Float[torch.Tensor, ""]:
        loss = None 
        if closure is not None:
            with torch.enable_grad():
                loss = closure() # recalculate the loss if needed

        for group in self.param_groups:
            lr = group["lr"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise ValueError("Sparse gradients are not supported.")
                
                # Calculate in float32 in case of overflow
                in_p_dtype = p.data.dtype
                grad_fp32 = p.grad.detach().to(torch.float32)
                p_fp32 = p.data.detach().to(torch.float32)
                
                state = self.state[p]
                t = state.get("t", 0) + 1
                m = state.get("m", torch.zeros_like(p_fp32, dtype=torch.float32, device=p.device))
                v = state.get("v", torch.zeros_like(p_fp32, dtype=torch.float32, device=p.device))

                # replace the m directly, without creating new tensor
                m.mul_(beta1).add_(grad_fp32, alpha=1-beta1)
                v.mul_(beta2).addcmul_(grad_fp32, grad_fp32, value=1-beta2)

                cur_lr = lr * math.sqrt(1-beta2**t) / (1-beta1**t)
                
                demon = v.sqrt().add_(eps)
                p_fp32.addcdiv_(m, demon, value=-cur_lr) # update the parameter
                # p_fp32 -= cur_lr * m/(torch.sqrt(v) + eps) # update the parameter
                p_fp32.mul_(1 - lr * weight_decay) # Apply weight decay
                state["t"] = t
                state["m"] = m
                state["v"] = v
                p.copy_(p_fp32.to(in_p_dtype))
        return loss


def get_lr_cosine_schedule(t: int,
                            lr_max: float,
                            lr_min: float,
                            warm_ups: int,
                            tc: int) -> float:
    """
    Implementation of the learning rate scheduler.

    Args:
        t: The current step.
        lr_max: The maximum learning rate.
        lr_min: The minimum learning rate.
        warm_ups: The number of warmup steps.
        tc: The number of cosine annealing iterations.

    Returns:
        The learning rate.
    """
    if lr_min < 0 or lr_max < 0:
        raise ValueError("Learning rate must be non-negative.")
    if lr_max < lr_min:
        raise ValueError("Maximum learning rate must be greater than minimum learning rate.")
    if warm_ups < 0 or tc < 0:
        raise ValueError("Warmup steps and total steps must be non-negative.")
    if warm_ups > tc:
        raise ValueError("Warmup steps must be less than total steps.")
    
    if t < warm_ups:
        cur_lr = (t/warm_ups) * lr_max
    elif t >= warm_ups and t <= tc:
        p1 = 1 + math.cos((t-warm_ups)*math.pi/(tc-warm_ups))
        p2 = lr_max - lr_min
        cur_lr = lr_min + 0.5 * p1 * p2
    else:
        cur_lr = lr_min
    return cur_lr

@torch.no_grad()
def gradient_clipping(params: Iterable[torch.nn.Parameter], 
                    max_l2_norm: float,
                    eps: float=1e-6) -> float:
    """
    Implementation of the gradient clipping. The gradient will be scaled inplace.

    Args:
        params: model parameters.
        max_l2_norm: max l2-norm value for the gradient of all parameters.
        eps: constant for numeric stability.

    Returns:
        The l2_norm for gradient of all parameters.
    """
    total_norm = 0.0
    for p in params:
        if p.grad is None:
            continue
        grad = p.grad
        total_norm += grad.pow(2).sum().item()

    total_norm = total_norm ** 0.5

    if total_norm > max_l2_norm:
        scale = max_l2_norm / (total_norm + eps)
        for p in params:
            if p.grad is None:
                continue
            p.grad.mul_(scale)

    return total_norm
    


if __name__ == "__main__":
    # test SGD
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = AdamW([weights], lr=1)

    for t in range(100):
        opt.zero_grad()
        loss = (weights**2).mean()
        print(loss.cpu().item())
        loss.backward()
        opt.step()
    