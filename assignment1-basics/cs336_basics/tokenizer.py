

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

import regex as re


class BytePairEncodeToken(object):

    """
    Implementation of the Byte Pair Encoding (BPE) token. This class represents a token and its count in the BPE algorithm.
    """

    def __init__(self):
        self._init_vocab()
        self.pre_token_regex = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    def build_reverse_vocab(self) -> dict[int, str]:
        return {v: k for k, v in self.vocab.items()}

    def _init_vocab(self) -> None:
        """
        Initilize the vocabulary with 256 byte tokens and a special end of text token. 
        This will be used to encode the pre-tokenized corpus into byte pairs.
        """
        self.vocab = {bytes([i]): i for i in range(256)}
        self.vocab[b"<|endoftext|>"] = 256
    
    def _count_pairs(self, tokens_freq: dict[str, int]) -> dict[tuple[bytes], int]:
        """
        Count frequency of adjacent byte pairs based on the pre-tokenized tokens frequency.
        
        :param tokens_freq: pre-tokenized tokens with their frequences in the corpus.
        :type tokens_freq: dict[str, int]
        :return: list of byte pairs and their frequencies, sorted by frequency in frequence descending order and lexicographically descending order.
        :rtype: list[tuple[str, int]]
        """
        
        pairs = defaultdict(int)
        for token_bt, freq in tokens_freq.items():
            for pair in zip(token_bt, token_bt[1:]):
                pairs[pair] += freq
        return pairs
    
    def pre_tokenize(self, corpus: list[str]) -> dict[tuple[bytes], int]:
        """
        Pre-tokenize the corpus using a regex pattern that captures words, numbers, punctuation, and whitespace. The regex pattern is designed to match common tokenization rules, including contractions and special characters.
        which with be used to accelerate the pairs counting.

        :param corpus: Description
        :return: tokens with their frequencies in the corpus
        :rtype: dict
        """
        tokens_freq = defaultdict(int)
        token_num = 0
        for text in corpus:
            tokens = self.pre_token_regex.finditer(text)
            for token in tokens:
                bs = token.group().encode("utf-8")
                token_bytes = tuple(bs[i:i+1] for i in range(len(bs)))
                tokens_freq[token_bytes] += 1
                token_num += len(token_bytes)
        return tokens_freq, token_num
    
    def merge_pair(self, token: tuple[bytes], a: bytes, b: bytes) -> tuple[bytes]:
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
                new_token.append(a+b)
                i += 2
            else:
                new_token.append(token[i])
                i += 1
        return tuple(new_token)

    def merge(self, best_pair: tuple[bytes], tokens_freq: dict[tuple[bytes], int]) -> dict[tuple[bytes], int]:
        """
        Update the tokens_frequency by merging the best pair into a new token.
        Update the vocabulary including the new token.
        
        :param best_pair: most frequent byte pair to be merged into a new token, represented as a tuple of two bytes.
        :type best_pair: tuple[bytes]
        :param tokens_freq: pre-tokenized tokens with their frequences in the corpus, represented as a dictionary where keys are tuples of bytes and values are their corresponding frequencies.
        :type tokens_freq: dict[tuple[bytes], int]
        :return: updated tokens frequency dictionary with the best pair merged into a new token.
        :rtype: dict[tuple[bytes], int]
        """
        # update tokens frequence by merging the best pair into a new token and updating the frequencies accordingly.
        new_tokens_freq = defaultdict(int)
        token_num = 0
        for token_bt, freq in tokens_freq.items():
            new_token = self.merge_pair(token_bt, best_pair[0], best_pair[1])
            new_tokens_freq[new_token] = new_tokens_freq[new_token] + freq
            token_num += len(new_token)
        # update the vocabulary with the new token
        new_token_str = best_pair[0] + best_pair[1]
        self.vocab[new_token_str] = len(self.vocab)
        return new_tokens_freq, token_num
    
    def train(self, corpus: list[str], num_merges: int) -> None:
        tokens_freq, init_token_num = self.pre_tokenize(corpus)
        token_num = init_token_num.copy()
        for _ in range(num_merges):
            pairs_freq = self._count_pairs(tokens_freq)
            if not pairs_freq:
                break
            best_pair = max(pairs_freq, key=pairs_freq.get)
            best_pair_freq = pairs_freq[best_pair]
            # best_pair, freq = max(pairs_freq.items(), key=lambda item: (item[1], item[0][0], item[0][1]))
            print("Current are merging:", best_pair, best_pair_freq)
            tokens_freq, new_token_num = self.merge(best_pair, tokens_freq)
            print("Calculated token reduced: ", best_pair_freq)
            print("Real token reduced: ", token_num - new_token_num)
            print("Compression ratio: ", (init_token_num*1.0)/new_token_num)
            token_num = new_token_num
        self.id_to_token = self.build_reverse_vocab()

    def save_vocab(self, file_path: str) -> None:
        """
        Save the trained BPE vocabulary to a file. The vocabulary is saved in a simple text format where each line contains a token and its corresponding id, separated by a tab character.

        :param file_path: path to the file where the vocabulary will be saved.
        :type file_path: str
        """
        with open(file_path, "w+", encoding="utf-8") as f:
            for token, id in self.vocab.items():
                f.write(f"{token.decode('utf-8', errors='ignore')}\t{id}\n")

    def load_vocab(self, file_path: str) -> None:
        """
        Load a BPE vocabulary from a file. The file should be in the same format as the one produced by the save_vocab method, where each line contains a token and its corresponding id, separated by a tab character.

        :param file_path: path to the file from which the vocabulary will be loaded.
        :type file_path: str
        """
        self.vocab = {}
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                token, id = line.strip().split("\t")
                self.vocab[token.encode("utf-8")] = int(id)
        self.id_to_token = self.build_reverse_vocab()
    
    def encode(self, text: str) -> list[int]:
        """
        Encode str into a list of token ids using the trained BPE vocabulary.
        
        :param text: text to be encoded into token ids.
        :type text: str
        :return: token ids corresponding to the input text based on the trained BPE vocabulary.
        :rtype: list[int]
        """

        tokens = self.pre_token_regex.finditer(text)
        ids = []
        for token in tokens:
            bs = token.group().encode("utf-8")
            cur_bytes = tuple(bs[i:i+1] for i in range(len(bs)))
            while True:
                pairs = [pair for pair in zip(cur_bytes, cur_bytes[1:])]
                candidates = [(self.vocab.get(pair[0]+pair[1]), pair) for pair in pairs if pair[0]+pair[1] in self.vocab]
                if not candidates:
                    break
                _, (a,b) = min(candidates)
                cur_bytes = self.merge_pair(cur_bytes, a, b)
            ids.extend([self.vocab[b] for b in cur_bytes])
        return ids

    def decode(self, ids: list[int]) -> str:
        """
        Decode a list of token ids back into a string using the trained BPE vocabulary.
        
        :param ids: list of token ids to be decoded back into a string.
        :type ids: list[int]
        :return: decoded string from the input token ids.
        :rtype: str
        """

        bytes_list = [self.id_to_token[id] for id in ids]
        return b"".join(bytes_list).decode("utf-8", errors="strict")
    

# Test
if __name__ == "__main__":
    data = []
    with open("../data/TinyStoriesV2-GPT4-train.txt", "r") as in_file:
        for line in in_file:
            data.append(line.strip())

    tokcli = BytePairEncodeToken()
    tokcli.train(data, 10)
    tokcli.save_vocab("./test_train.txt")
        