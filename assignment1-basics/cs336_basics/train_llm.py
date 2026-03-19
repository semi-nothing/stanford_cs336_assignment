
import os
import math
import wandb
import torch
import torch.nn as nn
import numpy as np
from jaxtyping import Float, Int, Bool

from typing import Optional
from collections.abc import Callable, Iterable

from data_utils import get_batch
# from transformer import TransformerLLM
from transformer_norope import TransformerLLM


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

    @torch.no_grad()
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
                betas: list[float, float]=[0.9, 0.999], 
                weight_decay: float=0,
                eps: float=1e-8) -> None:
        if lr <= 0:
            raise ValueError("Learning rate must be positive.")
        if betas[0] <= 0 or betas[0] >= 1:
            raise ValueError("Beta1 must be between 0 and 1.")
        if betas[1] <= 0 or betas[1] >= 1:
            raise ValueError("Beta2 must be between 0 and 1.")
        if weight_decay < 0:
            raise ValueError("Weight decay must be non-negative.")
        if eps <= 0:
            raise ValueError("Epsilon must be positive.")
        defaults = {"lr": lr, "betas": betas, "weight_decay": weight_decay, "eps": eps}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable]=None) -> Float[torch.Tensor, ""]:
        loss = None 
        if closure is not None:
            with torch.enable_grad():
                loss = closure() # recalculate the loss if needed

        for group in self.param_groups:
            lr = group["lr"]
            beta1 = group["betas"][0]
            beta2 = group["betas"][1]
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

@torch.no_grad()
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

def save_checkpoint(model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer,
                    iteration: int,
                    out: str) -> None:
    """
    Save the model and optimizer state to a file.

    Args:
        model: model to save.
        optimizer: optimizer to save.
        iteration: iteration number.
        out: output file path.
    """
    if not os.path.exists(os.path.dirname(out)):
        os.makedirs(os.path.dirname(out), exist_ok=True)

    out_path = out.format(iteration)
    torch.save({"model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "iteration": iteration}, 
                out_path)

def load_checkpoint(src: str,
                    model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer=None,
                    device: str="cpu") -> int:
    """
    Load the model and optimizer state from a file. Also the iteration number.

    Args:
        src: source file path.
        model: model to load.
        optimizer: optimizer to load.
    """
    res = torch.load(src, map_location=device)

    model.load_state_dict(res["model"])
    if optimizer is not None:
        optimizer.load_state_dict(res["optimizer"])

    return res["iteration"]


def init_wandb(lr: float,
                lr_max: float,
                lr_min: float,
                warm_ups: int,
                tc: int,
                betas: tuple[float, float],
                weight_decay: float,
                max_l2_norm: float,
                batch_size: int,
                seq_length: int,
                model_params: Optional[dict]=None) -> wandb.sdk.wandb_run.Run:
    config = {
        "architecture": "transformer_postnorm",
        "dataset": "tinystory",
        "epochs": 1,
        "lr": lr,
        "lr_max": lr_max,
        "lr_min": lr_min,
        "warm_ups": warm_ups,
        "tc": tc,
        "betas": betas,
        "weight_decay": weight_decay,
        "max_l2_norm": max_l2_norm,
        "batch_size": batch_size,
        "seq_length": seq_length,
    }
    if model_params is not None:
        config.update(model_params)
    run = wandb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
    entity="liangyuan-edin-queen-mary-university-of-london",
    # Set the wandb project where this run will be logged.
    project="Stanford_cs336",
    # Track hyperparameters and run metadata.
    config=config
    )

    return run

def build_model(model_params: Optional[dict]=None) -> torch.nn.Module:
    model = TransformerLLM(**model_params)
    return model

def train(model: torch.nn.Module,
          train_data_file: str,
          val_data_file: str,
          src: str,
          out: str,
          lr: float,
          lr_max: float,
          lr_min: float,
          warm_ups: int,
          tc: int,
          steps: int,
          save_steps: int,
          eval_steps: int,
          log_steps: int,
          betas: tuple[float, float],
          weight_decay: float,
          max_l2_norm: float,
          batch_size: int,
          seq_length: int,
          eps: float=1e-8,
          device: str="cpu",
          model_params: Optional[dict]=None
          ):

    # check the parameters
    if lr < 0 or lr_max < 0 or lr_min < 0:
        raise ValueError("Learning rate must be non-negative.")
    if lr_max < lr_min:
        raise ValueError("Maximum learning rate must be greater than minimum learning rate.")
    if warm_ups < 0 or tc < 0 or steps < 0 or save_steps < 0 or eval_steps < 0 or log_steps < 0:
        raise ValueError("Warmup steps, total steps, save steps, eval steps and log steps must be non-negative.")
    if warm_ups > tc:
        raise ValueError("Warmup steps must be less than total steps.")
    if betas[0] <= 0 or betas[0] >= 1:
        raise ValueError("Beta1 must be between 0 and 1.")
    if betas[1] <= 0 or betas[1] >= 1:
        raise ValueError("Beta2 must be between 0 and 1.")
    if weight_decay < 0:
        raise ValueError("Weight decay must be non-negative.")
    if max_l2_norm < 0:
        raise ValueError("Max l2 norm must be non-negative.")

    # move model to device
    model.to(device)

    # initialize the optimizer
    opt = AdamW(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay, eps=eps)

    # recover from checkpoint if exists
    if os.path.exists(src):
        start_t = load_checkpoint(src, model, opt, device)
    else:
        start_t = 0
    
    # data
    train_data = np.memmap(train_data_file, dtype=np.uint16, mode="r")
    if val_data_file is not None:
        val_data = np.memmap(val_data_file, dtype=np.uint16, mode="r")

    # initialize wandb
    wandb_run = init_wandb(lr, lr_max, lr_min, warm_ups, tc, betas, weight_decay, max_l2_norm, batch_size, seq_length, model_params)

    for t in range(start_t, steps):
        model.train()
        opt.zero_grad()

        cur_lr = get_lr_cosine_schedule(t, lr_max, lr_min, warm_ups, tc)
        for g in opt.param_groups:
            g["lr"] = cur_lr

        ipt, tgt = get_batch(train_data, batch_size, seq_length, device)
        logits = model(ipt)
        loss = cross_entropy(logits, tgt)
        loss.backward()
        gradient_clipping(model.parameters(), max_l2_norm, eps)
        opt.step()

        if t % log_steps == 0:
            wandb_run.log({"train_loss": loss.item(), "learning_rate": cur_lr, "train_perplexity": perplexity(loss).item()}, step=t)
            print(f"Iteration {t}, loss {loss.item()}")
        if t != 0 and t % save_steps == 0:
            save_checkpoint(model, opt, t, out)
        if t != 0 and t % eval_steps == 0 and val_data_file is not None:
            model.eval()
            with torch.no_grad():
                ipt, tgt = get_batch(val_data, batch_size, seq_length, device)
                logits = model(ipt)
                val_loss = cross_entropy(logits, tgt)
                val_ppl = perplexity(val_loss)
                wandb_run.log({"validation_loss": val_loss.item(), "validation_perplexity": val_ppl.item()}, step=t)
                print(f"Iteration {t}, validation loss {val_loss.item()}, perplexity {val_ppl.item()}")
            model.train()
    wandb_run.finish()


if __name__ == "__main__":
    # # test SGD
    # weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    # opt = AdamW([weights], lr=1)

    # for t in range(100):
    #     opt.zero_grad()
    #     loss = (weights**2).mean()
    #     print(loss.cpu().item())
    #     loss.backward()
    #     opt.step()

    # build model
    model_params = {"vocab_size": 10257, 
                    "d_model": 768, 
                    "num_heads": 12, 
                    "num_layer": 12,
                    "d_ff": 3072,
                    "theta": 10000,
                    "max_seq_len": 1024,
                    "eps": 1e-6,
                    "device": "cuda", 
                    "dtype": torch.float32}
    model = build_model(model_params)

    # train
    train_params = {
        "train_data_file": "../data/TinyStoriesV2-GPT4-train_encoded.bin",
        "val_data_file": "../data/TinyStoriesV2-GPT4-valid_encoded.bin",
        "src": "../models/tinystory_transformer_basics/checkpoint_postnorm.pt",
        "out": "../models/tinystory_transformer_basics/checkpoint_postnorm_{}.pt",
        "lr": 5e-4,
        "lr_max": 5e-4,
        "lr_min": 1e-6,
        "warm_ups": 500,
        "tc": 10000,
        "steps": 10000,
        "save_steps": 1000,
        "eval_steps": 1000,
        "log_steps": 100,
        "betas": (0.9, 0.999),
        "weight_decay": 0.01,
        "max_l2_norm": 1.0,
        "batch_size": 8,
        "seq_length": 1024,
        "eps": 1e-8,
        "device": "cuda",
        "model_params": model_params
    }
    train(model=model, **train_params)
    