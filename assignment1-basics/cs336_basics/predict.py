
import torch
from einops import rearrange
from jaxtyping import Int, Float

from transformer import TransformerLLM, SoftMax


softmax = SoftMax()

@torch.inference_mode()
def top_p_sampling(logits: Float[torch.Tensor, "batch_size vocab_size"],
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


# test decode function
if __name__ == "__main__":
    from train_llm import build_model, AdamW, load_checkpoint
    device = "cuda" if torch.cuda.is_available() else "cpu"
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
    model.to(device)
    load_checkpoint("../models/tinystory_transformer_basics/checkpoint_9000.pt", model, None, device)
    model.eval()

    from tokenizer_parall import BytePairEncodeToken
    data_name = "tinystory"
    tokcli = BytePairEncodeToken()
    print("Loading vocab from ", f"./{data_name}_bpe_parall.pkl")
    tokcli.load_vocab(f"./{data_name}_bpe_parall.pkl")
    input_text = '"Hey, Polly! Let’s go out!” said Tim. Sue looked at the sky and saw that it was difficult to find a way to dance shining. She smiled and agreed to help the talking!” As Sue watched the sky moved, what it was. She'
    input_ids = torch.tensor(tokcli.encode(input_text), dtype=torch.long, device=device).unsqueeze(0) # [1, seq_len]

    generated_ids = decode(model, input_ids, temperature=0.5, p=0.8, max_length=256, end_token=256)
    print("Generated token IDs: ", generated_ids.squeeze().tolist())
    print("Generated text: ", tokcli.decode(generated_ids.squeeze().tolist(), errors="replace"))

