import re
from collections import Counter

import torch
from torch.utils.data import Dataset


ANCHOR_NAMES = ["Raskolnikov", "Sonia", "Razumihin", "Dounia", "Katerina", "Porfiry"]


def build_vocab(train_text: str, min_count: int = 10):
    """
    Builds char2idx / idx2char mappings from TRAIN TEXT ONLY. Any character
    appearing fewer than min_count times is collapsed into a single <UNK>
    token so rare/foreign characters don't bloat the vocabulary. Sorting
    kept_chars ensures deterministic ordering across runs. This is critical 
    for reproducibility when saving/loading model checkpoints.

    Returns:
        char2idx: dict mapping character -> integer index
        idx2char: list mapping integer index -> character (<UNK> at index 0)
    """
    char_counts = Counter(train_text)
    kept_chars = sorted([ch for ch, count in char_counts.items()
                         if count >= min_count])
    idx2char = ["<UNK>"] + kept_chars
    char2idx = {ch: i for i, ch in enumerate(idx2char)}
    return char2idx, idx2char


def encode(text: str, char2idx: dict) -> list:
    """
    Converts a string into a list of vocabulary indices using an already-built
    char2idx mapping. Any character not in char2idx maps to <UNK> at index 0.
    """
    unk_idx = char2idx["<UNK>"]
    return [char2idx.get(ch, unk_idx) for ch in text]


def make_chunks(encoded: list, seq_len: int = 100):
    """
    Splits encoded text into non-overlapping (input, target) chunk pairs for
    truncated BPTT training. Each chunk reads seq_len + 1 characters: the
    first seq_len form the input, the last seq_len (shifted by one) form the
    target. Any leftover characters that don't fill a complete chunk are dropped.

    Returns:
        list of (input_seq, target_seq) pairs, each a list of integer indices
    """
    chunks = []
    num_chunks = (len(encoded) - 1) // seq_len
    for i in range(num_chunks):
        start = i * seq_len
        chunk = encoded[start: start + seq_len + 1]
        if len(chunk) < seq_len + 1:
            break
        chunks.append((chunk[:-1], chunk[1:]))
    return chunks


class CharDataset(Dataset):
    """
    Wraps pre-chunked (input, target) pairs as a PyTorch Dataset so DataLoader
    can batch and shuffle them during training. Returns LongTensors since
    nn.Embedding and CrossEntropyLoss both expect integer index inputs.
    """
    def __init__(self, chunks):
        self.chunks = chunks

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        input_seq, target_seq = self.chunks[idx]
        return (torch.tensor(input_seq,  dtype=torch.long),
                torch.tensor(target_seq, dtype=torch.long))


def find_anchor_positions(text: str, anchor_names: list = ANCHOR_NAMES) -> dict:
    """
    Finds every occurrence of each anchor name in text, recording the character
    position where each occurrence starts in raw coordinates within the given text.
    Used by H3's distance-vs-loss analysis to compute distance-since-last-
    occurrence at every position in the test set.

    Returns:
        dict mapping name -> sorted list of start positions
    """
    positions = {}
    for name in anchor_names:
        matches = [m.start() for m in re.finditer(rf"\b{name}\b", text)]
        positions[name] = matches
    return positions