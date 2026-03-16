
import torch
from einops import rearrange
from jaxtyping import Int, Float

from transformer import TransformerLLM, softmax


@torch.inference_mode()
def top_p_sampling(logits: Float[torch.Tensor, "batch_size, vocab_size"],
                    p: float=0.9) -> Int[torch.Tensor, "batch_size"]:
    """
    Perform nucleus (top-p) sampling on the given logits.

    Args:
        logits: The logits from the model, of shape [batch_size, vocab_size].
        p: The cumulative probability threshold for nucleus sampling.

    Returns:
        A tensor containing the sampled token IDs, of shape [batch_size].
    """
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)

    # prefix sum
    cumsum_prob = torch.cumsum(sorted_logits, dim=-1)

    # find the indices to remove (The first index where the cumulative probability exceeds p should be kept, and the rest should be removed)
    sorted_indices_to_remove = cumsum_prob > p
    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone() # Shift the indices to the right to keep the first index where the cumulative probability exceeds p
    sorted_indices_to_remove[:, 0] = False # Keep the first token

    # Replace the low probability tokens with 0
    sorted_logits[sorted_indices_to_remove] = 0.0

    # Renormalize the probabilities
    softmax_sorted_logits = softmax(sorted_logits, dim=-1)

    # Sample from the distribution
    sampled_next_token = torch.multinomial(softmax_sorted_logits, num_samples=1)

    # Map back to the original indices
    next_token = torch.gather(sorted_indices, dim=-1, index=sampled_next_token)

    return rearrange(next_token, "batch_size 1-> batch_size")
    

@torch.inference_mode()
def decode(model: torch.nn.Module,
           input_ids: Int[torch.Tensor, "batch_size seq_len"],
           temperature: float=1.0,
           p: float=0.9,
           max_length: int=256,
           end_token: int=256) -> Int[torch.Tensor, "batch_size"]:
    """
    Generate a sequence of tokens from the given model and input.

    Args:
        model: The language model to use for generation.
        input_ids: The input token IDs to start the generation from.
        temperature: The sampling temperature to control randomness.
        p: The cumulative probability threshold for nucleus sampling.
        max_length: The maximum length of the generated sequence.
        end_token: The token ID that indicates the end of the sequence.

    Returns:
        A tensor containing the generated token IDs.
    """
    finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
    for t in range(max_length):
        logits = model(input_ids)
        softmax_logits = softmax(logits[:, -1, :] / temperature, dim=-1) # [batch_size, vocab_size]]
        next_token = top_p_sampling(softmax_logits, p) # [batch_size]

        finished = finished | (next_token == end_token)
        next_token = next_token.masked_fill(finished, end_token) # If finished, set the next token to end_token to prevent further generation

        input_ids = torch.cat([input_ids, rearrange(next_token, "batch_size -> batch_size 1")], dim=-1) # [batch_size, seq_len + 1]
        if finished.all():
            break

    return input_ids
