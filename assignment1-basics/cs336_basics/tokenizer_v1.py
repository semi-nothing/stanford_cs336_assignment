

"""
Unicode:
ord(str) -> int
chr(int) -> str 
Also can work on chr(byte_int) -> str

Unicode encoding: encode a string into bytes
str.encode("utf-8") -> bytes
bytes.decode("utf-8") -> str

"""

from collections import defaultdict
from functools import wraps

import math
import time
import pickle
import pstats
import cProfile
import regex as re
import matplotlib.pyplot as plt

# from pretokenization_example import find_chunk_boundaries


def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__} took {end-start:.6f} seconds.")
        return result
    return wrapper


class BytePairEncodeToken(object):

    """
    Implementation of the Byte Pair Encoding (BPE) token. This class represents a token and its count in the BPE algorithm.
    """

    def __init__(self):
        self._init_vocab()
        self.pre_token_regex = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        self.special_tokens =  ["<|endoftext|>"]
        self.special_token_regex = re.compile("(" + "|".join(map(re.escape, self.special_tokens)) + ")")

    @timeit
    def build_reverse_vocab(self) -> dict[int, tuple[int]]:
        return {v: k for k, v in self.vocab.items()}

    def _init_vocab(self) -> None:
        """
        Initilize the vocabulary with 256 byte tokens and a special end of text token. 
        This will be used to encode the pre-tokenized corpus into byte pairs.
        """
        self.vocab = {i: i for i in range(257)} # 0-255 UTF-8 encoding code, 256 special token
        # self.vocab[b"<|endoftext|>"] = 256
    
    def _count_pairs(self, tokens_freq: dict[tuple[int], int]) -> dict[tuple[int], int]:
        """
        Count frequency of adjacent byte pairs based on the pre-tokenized tokens frequency.
        
        :param tokens_freq: pre-tokenized tokens with their frequences in the corpus.
        :type tokens_freq: dict[tuple[int], int]
        :return: list of byte pairs and their frequencies, sorted by frequency in frequence descending order and lexicographically descending order.
        :rtype: dict[tuple[int], int]
        """
        
        pairs = defaultdict(int)
        for token_bs, freq in tokens_freq.items():
            for pair in zip(token_bs, token_bs[1:]):
                pairs[pair] += freq
        return pairs
    
    def pre_tokenize(self, text: str) -> list[tuple[int]]:
        """
        Pre-tokenize the corpus using a regex pattern that captures words, numbers, punctuation, and whitespace. 
        The regex pattern is designed to match common tokenization rules, including contractions and special characters.
        which with be used to accelerate the pairs counting.

        :param text: Description
        :return: tokens
        :rtype: list[tuple[int]]
        """
        # process special token
        parts = self.special_token_regex.split(text)
        res = []
        for part in parts:
            if part not in self.special_tokens:
                tokens = self.pre_token_regex.finditer(part)
                for token in tokens:
                    bs = tuple(token.group().encode("utf-8")) # UTF-8 byte coding list
                    res.append(bs)
            else:
                res.append((256,))
        return res

    @timeit
    def get_stats(self, corpus: list[str]) -> dict[tuple[int], int]:
        """
        get_stats is to calculate tokens frequency based on the output of pre_tokenize.

        :param corpus: Description
        :return: tokens with frequency
        :rtype: dict[tuple[int], int]
        """
        tokens_freq = defaultdict(int)
        token_num = 0
        for text in corpus:
            bss = self.pre_tokenize(text)
            for bs in bss:
                if bs == (256,): continue # pass special token
                tokens_freq[bs] += 1
                token_num += len(bs)
            else:
                continue
        return tokens_freq, token_num
    
    def merge_pair(self, token: tuple[int], a: int, b: int, z: int) -> tuple[int]:
        """
        Merge a pair of bytes (a, b) into a new token in the given token tuple.
        
        :param token: tuple of bytes to be merged
        :type token: tuple[bytes]
        :param a: first byte in the pair to be merged
        :type a: bytes
        :param b: second byte in the pair to be merged
        :type b: bytes
        :return: merged tuple of bytes with the pair (a, b) merged into a new token
        :rtype: tuple[bytes]
        """
        new_token = []
        i = 0
        while i < len(token):
            if i < len(token) - 1 and token[i] == a and token[i + 1] == b:
                new_token.append(z)
                i += 2
            else:
                new_token.append(token[i])
                i += 1
        return tuple(new_token)

    def merge(self, best_pair: tuple[int], tokens_freq: dict[tuple[int], int]) -> dict[tuple[int], int]:
        """
        Update the tokens_frequency by merging the best pair into a new token.
        Update the vocabulary including the new token.
        
        :param best_pair: most frequent byte pair to be merged into a new token, represented as a tuple of two bytes.
        :type best_pair: tuple[int]
        :param tokens_freq: pre-tokenized tokens with their frequences in the corpus, represented as a dictionary where 
        keys are tuples of bytes and values are their corresponding frequencies.
        :type tokens_freq: dict[tuple[int], int]
        :return: updated tokens frequency dictionary with the best pair merged into a new token.
        :rtype: dict[tuple[int], int]
        """
        # update the vocabulary with the new token
        new_tokens_freq = defaultdict(int)
        new_token = (best_pair[0], best_pair[1])
        self.vocab[new_token] = len(self.vocab)
         # update tokens frequence by merging the best pair into a new token and updating the frequencies accordingly.
        token_num = 0
        for token_bs, freq in tokens_freq.items():
            new_token_bs = self.merge_pair(token_bs, best_pair[0], best_pair[1], self.vocab[new_token])
            new_tokens_freq[new_token_bs] = new_tokens_freq[new_token_bs] + freq
            token_num += len(new_token_bs) * freq
        return new_tokens_freq, token_num
    
    @timeit
    def train(self, corpus: list[str], num_merges: int) -> list[float]:
        tokens_freq, init_token_num = self.get_stats(corpus)
        compression_ratio = []
        token_num = init_token_num
        for _ in range(num_merges):
            pairs_freq = self._count_pairs(tokens_freq)
            if not pairs_freq:
                break
            best_pair = max(pairs_freq, key=pairs_freq.get)
            best_pair_freq = pairs_freq[best_pair]
            # best_pair, freq = max(pairs_freq.items(), key=lambda item: (item[1], item[0][0], item[0][1]))
            print("Current are merging:", best_pair, best_pair_freq)
            tokens_freq, new_token_num = self.merge(best_pair, tokens_freq)
            # print("Calculated token reduced: ", best_pair_freq)
            # print("Real token reduced: ", token_num - new_token_num)
            compression_ratio.append((init_token_num*1.0)/new_token_num)
            # print("Compression ratio: ", (init_token_num*1.0)/new_token_num)
            token_num = new_token_num
        self.id_to_token = self.build_reverse_vocab()
        return compression_ratio

    def save_vocab(self, file_path: str) -> None:
        """
        Save the trained BPE vocabulary to a file. The vocabulary is saved in a simple text format where each line 
        contains a token and its corresponding id, separated by a tab character.

        :param file_path: path to the file where the vocabulary will be saved.
        :type file_path: str
        """
        with open(file_path, "wb") as f:
            pickle.dump(self.vocab, f)

    def load_vocab(self, file_path: str) -> None:
        """
        Load a BPE vocabulary from a file. The file should be in the same format as the one produced by the save_vocab 
        method, where each line contains a token and its corresponding id, separated by a tab character.

        :param file_path: path to the file from which the vocabulary will be loaded.
        :type file_path: str
        """
        self.vocab = {}
        with open(file_path, "rb") as f:
            self.vocab = pickle.load(f)
        self.id_to_token = self.build_reverse_vocab()
    
    def encode(self, text: str) -> list[int]:
        """
        Encode str into a list of token ids using the trained BPE vocabulary.
        
        :param text: text to be encoded into token ids.
        :type text: str
        :return: token ids corresponding to the input text based on the trained BPE vocabulary.
        :rtype: list[int]
        """

        tokens = self.pre_tokenize(text)
        ids = []
        for bs in tokens:
            if bs != (256,):
                while True:
                    pairs = [pair for pair in zip(bs, bs[1:])]
                    candidates = [(self.vocab.get(pair), pair) for pair in pairs if pair in self.vocab]
                    if not candidates:
                        break
                    idx, (a,b) = min(candidates)
                    bs = self.merge_pair(bs, a, b, idx)
            ids.extend(bs)
        return ids

    def decode_id(self, id_):
        res = []

        def dfs(id_):
            bs = self.id_to_token[id_]
            if isinstance(bs, int):
                res.append(bs)
                return
            dfs(bs[0])
            dfs(bs[1])

        dfs(id_)

        return res

    def decode(self, ids: list[int]) -> str:
        """
        Decode a list of token ids back into a string using the trained BPE vocabulary.
        
        :param ids: list of token ids to be decoded back into a string.
        :type ids: list[int]
        :return: decoded string from the input token ids.
        :rtype: str
        """
        bs = []
        for id_ in ids:
            bs.extend(self.decode_id(id_))
        return "".join([chr(b) if b != 256 else "<|endoftext|>" for b in bs])


def plot_fig(X, Y):
    plt.figure()
    plt.plot(X, Y)
    plt.xlabel("Number of merges")
    plt.ylabel("Compression ratio")
    plt.title("Compression Ratio vs Number of merges")
    plt.show()
    plt.savefig("./compression_ratio_vs_merge_v1.png")

@timeit
def load_data(file_path:str) -> list[str]:
    count = 0
    data = []
    with open(file_path, "rb") as in_file:
        for line in in_file:
            # if count > 100: break
            data.extend(line.decode("utf-8", errors="ignore").split("\n"))
            count += 1
    return data

# Test
if __name__ == "__main__":
    # monitor the function cost
    profiler = cProfile.Profile()
    profiler.enable()

    num_merge = 10000
    data = load_data("../data/TinyStoriesV2-GPT4-train.txt")

    tokcli = BytePairEncodeToken()
    tokens_freq, init_token_num = tokcli.get_stats(data)
    with open("serialize_token_freq.txt", "w+") as out_file:
        for k, v in tokens_freq.items():
            out_file.write(f"{k}\t{v}\n")
    print(init_token_num)
    exit(0)
    # unit test
    # print(list("Hello World! ".encode("utf-8")) + [256] + list(" a good day!".encode("utf-8")))
    print("Before training: ", tokcli.encode("Hello World! <|endoftext|> a good day!"))
    tokcli.id_to_token = tokcli.build_reverse_vocab()
    print("Decoding result: ", tokcli.decode(tokcli.encode("Hello World! <|endoftext|> a good day!")))
    compression_ratio = tokcli.train(data, num_merge)
    print("After training: ", tokcli.encode("Hello World! a good day!"))
    print("Decoding result: ", tokcli.decode(tokcli.encode("Hello World! <|endoftext|> a good day!")))
    tokcli.save_vocab("./test_train_v1.txt")

    # plot the compression ratio with merge
    real_num_merge = len(compression_ratio)
    print(f"Expect {num_merge} merges, but actually did {real_num_merge} merges.")
    num_merges = list(range(1, real_num_merge+1, 1))
    plot_fig(num_merges, compression_ratio)

    profiler.disable()
    profiler.dump_stats("profile.prof")

        