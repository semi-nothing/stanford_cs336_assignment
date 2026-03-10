
import random
import torch
import numpy as np
from jaxtyping import Float, Int, Bool

from tokenizer_parall import BytePairEncodeToken


def tokenize_data(in_file_path: str, data_name: str, out_file_path: str) -> None:
    """
    Use tokenizer to tokenize the data and save the tokenized data to a file.

    Args:
        in_file_path: input file path
        data_name: data name
        out_file_path: output file path

    Returns:
        None
    """
    tokcli = BytePairEncodeToken()
    print("Loading vocab from ", f"./{data_name}_bpe_parall.pkl")
    tokcli.load_vocab(f"./{data_name}_bpe_parall.pkl")
    count = 1
    # buffer = []
    out_file = open(out_file_path, "wb")
    with open(in_file_path, "rb") as in_file:
        for line in in_file:
            # if count > 100000: break
            # if len(buffer) > 1000000:
                # print(count, len(tokcli.vocab), len(getattr(tokcli, "cache", {})), len(getattr(tokcli, "byte_token", {})))
                # np.asarray(buffer, dtype=np.uint16).tofile(out_file)
                # buffer.clear()
            count += 1
            data = line.decode("utf-8", errors="ignore").strip()
            if data == "": continue
            encoded_data = tokcli.encode(data)
            # buffer.extend(encoded_data)
            np.asarray(encoded_data, dtype=np.uint16).tofile(out_file)
            if count % 1000000 == 0:
                print(count)
    
    # if buffer:
    #     np.asarray(buffer, dtype=np.uint16).tofile(out_file)
    out_file.close()    
    print("Tokenized data saved to ", out_file_path)
    

def get_batch(x: np.ndarray, 
                batch_size: int, 
                seq_len: int,
                device: str = "cpu") -> tuple[Int[torch.LongTensor, "batch_size seq_len"], Int[torch.LongTensor, "batch_size seq_len"]]:
    """
    Sample batch_size number of samples from the tokenized data file with lenth seq_len.

    Args:
        x: numpy array of tokenized data
        batch_size: batch size
        seq_len: sequence length
        device: device to use, cuda or cpu

    Returns:
        A tuple of (input, target) torch tensor. Both tensor have shape (batch_size, seq_len).
    """
    token_num = len(x) # np.memmap("train.bin", dtype=np.uint16)
    if token_num < seq_len+1:
        raise ValueError("Token num is not enough to sample seq_len inp/tgt.")

    starts = np.random.randint(0, token_num-seq_len, size=batch_size)

    inp = np.stack([x[s: (s+seq_len)] for s in starts]).astype(np.int64)
    tgt = np.stack([x[(s+1):(s+seq_len+1)] for s in starts]).astype(np.int64)

    inp = torch.from_numpy(inp).to(device)
    tgt = torch.from_numpy(tgt).to(device)

    return inp, tgt


# if __name__ == "__main__":
#     in_file_path = "../data/TinyStoriesV2-GPT4-valid.txt"
#     data_name = "tinystory"
#     out_file_path = "../data/TinyStoriesV2-GPT4-valid_encoded.bin"

#     tokenize_data(in_file_path, data_name, out_file_path)
