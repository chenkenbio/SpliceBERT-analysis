#!/usr/bin/env python3
"""
Author: Ken Chen (https://github.com/chenkenbio)
Date: 2024-05-18
"""

import argparse
import os
import sys
import pandas as pd
import math
import h5py
import numpy as np
from tqdm import tqdm
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple, Union
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, Subset, RandomSampler, DistributedSampler
from torch.cuda.amp import autocast, GradScaler

NT_TABLE = {
    "A": 0, "a": 0,
    "C": 1, "c": 1,
    "G": 2, "g": 2,
    "T": 3, "t": 3,
    "N": 4, "n": 4
}
def encode_sequence(seq: str) -> np.ndarray:
    return np.array([NT_TABLE[nt] for nt in seq], dtype=np.int8)
    


BASE_DIR = os.path.dirname(os.path.realpath(__file__))

class DeepstarrData(Dataset):
    def __init__(self, group: Literal["train", "val", "test"], encoding_type: Literal["splicebert", "dna5"]):
        super().__init__()
        assert group in ["train", "val", "test"]
        group = group[0].upper() + group[1:]

        df = pd.read_table(f"{BASE_DIR}/Sequences_activity_{group}.txt")

        self.y_dev = df.Dev_log2_enrichment.astype(np.float32)
        self.y_hk = df.Hk_log2_enrichment.astype(np.float32)

        fasta = f"{BASE_DIR}/Sequences_{group}.fa"
        processed = fasta + ".h5"

        if not os.path.exists(processed):
            seqs = list()
            with open(fasta) as infile, h5py.File(processed, 'w') as fp:
                for l in tqdm(infile, desc="Processing sequences: " + fasta):
                    if l.startswith(">"):
                        continue
                    seqs.append(encode_sequence(l.strip()))
                seqs = np.stack(seqs)
                print(seqs.shape)
                fp.create_dataset(name="sequence", data=seqs, compression="lzf")
        
        self.seqs = h5py.File(processed, 'r')['sequence']

        assert encoding_type in ["splicebert", "dna5"]
        self.to_splicebert = encoding_type == "splicebert"
    
    def __len__(self) -> int:
        return len(self.y_dev)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor]:
        seq = self.seqs[idx]
        if self.to_splicebert:
            seq = np.concatenate(([2], seq + 5, [3]))
        return torch.tensor(seq), self.y_dev[idx], self.y_hk[idx]
            




