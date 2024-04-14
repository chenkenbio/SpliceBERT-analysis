#!/usr/bin/env python3
r"""
Author: Ken Chen
Email: chenkenbio@gmail.com
Date: 2024-01-16


SpliceAI dataset: Canonical and GTEx splice sites
Version: 0.0.1

Source files are from Illumina's SpliceAI

Note: In gtex dataset, the number of splice donors does not equal to that of acceptors

"""

import argparse
import os
import sys
import numpy as np
from tqdm import tqdm
import pickle
import warnings
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple, Union
from scipy.sparse import csr_matrix

from ..scripts.genome import EasyGenome, get_reverse_strand
import logging
logger = logging.getLogger(__name__)

def convert_to_bert_encodings(seq: np.ndarray, cls_token: int=None, sep_token: int=None, target: Literal["dna-10"]="dna-10") -> np.ndarray:
    r"""
    Convert a sequence to BERT encodings, default target is "dna-10" (5 bases + 5 special tokens)
    dna-10: [PAD], [UNK], [CLS], [SEP], [MASK], N, A, C, G, T
    """
    assert seq.ndim == 1
    if target == "dna-10":
        seq += 5
    if cls_token is not None:
        seq = np.concatenate([[cls_token], seq])
    if sep_token is not None:
        seq = np.concatenate([seq, [sep_token]])
    return seq


class SpliceaiDataset(Dataset):
    __version__ = "0.0.1"

    def __init__(self, 
                 encoding_type: Literal["dna5", "splicebert"],
                 genome: str,
                 dataset: Literal["spliceai-gtex", "spliceai-canonical", "canonical", "gtex", "gtex-test"],
                 window_size: int=5000,
                 flanking_size: int=None, # if None, flanking_size = window_size
                 shift: int=0,
                 exclude_paralogs: bool=False,
                ):
        super().__init__()
        self.window_size = window_size
        self.flanking_size = window_size if flanking_size is None else flanking_size
        assert encoding_type in ["dna5", "splicebert"]
        self.to_bert_encoding = encoding_type == "splicebert"
        self.shift = shift

        self.chroms = list()
        self.starts = list()
        self.genes = list()
        self.gene_info = dict()
        self.paralogs = list()
        self.ss_data = dict()
        self.samples = list()

        self.train_genes = set()
        self.val_genes = set()
        self.test_genes = set()

        self.genome = EasyGenome(genome, in_memory=False)

        self.exclude_paralogs = exclude_paralogs
    
        if dataset == "spliceai-gtex" or dataset == "gtex":
            self._load_dataset("gtex")
        elif dataset == "spliceai-canonical" or dataset == "canonical":
            # raise NotImplementedError("canonical splice sites not implemented yet")
            self._load_dataset("canonical")
        else:
            self._load_dataset(dataset=dataset)
            # raise ValueError(f"dataset={dataset} not recognized, must be one of ['spliceai-gtex', 'spliceai-canonical', 'canonical', 'gtex']")
    
    def __len__(self):
        return len(self.chroms)
    
    def __getitem__(self, index):
        if len(self.train_genes) == 0 and self.shift > 0:
            warnings.warn(f"train_genes is empty, shift(={self.shift}) will not be applied")
        gene = self.genes[index]
        start = self.starts[index]
        if self.shift > 0 and gene in self.train_genes:
            shift = np.random.randint(-self.shift, self.shift + 1)
            start += shift

        chrom, tx_start, tx_end, strand, _ = self.gene_info[gene]

        start = max(min(start, tx_end - 1), tx_start - self.flanking_size + 1)
        end = start + self.window_size
        left, right = start - self.flanking_size, end + self.flanking_size

        # fetch ss data
        start_pad, end_pad = 0, 0
        if start < tx_start:
            start_pad = tx_start - start
            start = tx_start
        if end > tx_end:
            end_pad = end - tx_end
            end = tx_end
        ss = self.ss_data[gene][0, start - tx_start:end - tx_start].toarray().flatten().astype(np.int8)

        assert len(ss) > 0 #, "{}".format((gene, self.gene_info[gene], self.starts[index], self.starts[index] + self.window_size, start_pad, end_pad, left_pad, right_pad, len(seq), len(ss)))

        ss = np.concatenate([
            np.full((start_pad, ), -100, dtype=np.int8), 
            ss, 
            np.full((end_pad, ), -100, dtype=np.int8)
        ])

        # fetch sequence
        left_pad, right_pad = 0, 0
        if left < tx_start:
            left_pad = tx_start - left
            left = tx_start
        if right > tx_end:
            right_pad = right - tx_end
            right = tx_end
        seq = self.genome.fetch_sequence(chrom, left, right)
        seq = np.concatenate([np.zeros(left_pad, dtype=np.int8), seq, np.zeros(right_pad, dtype=np.int8)])

        if strand == "-":
            seq = get_reverse_strand(seq, integer=True)
            ss = ss[::-1].copy()

        assert len(seq) == 2 * self.flanking_size + self.window_size and len(ss) == self.window_size, "{}".format((
            gene, self.gene_info[gene], self.starts[index], self.starts[index] + self.window_size, start_pad, end_pad, left_pad, right_pad, len(seq), len(ss)
        ))

        if self.to_bert_encoding:
            seq = np.concatenate(([2], seq + 5, [3]))
        
        seq = torch.from_numpy(seq)
        ss = torch.from_numpy(ss)
        return seq, ss, index
    
    def _load_dataset(self, dataset: Literal["gtex", "canonical"]):
        data_file = f"../data/{dataset}_dataset.txt"
        if not os.path.exists(data_file):
            assert os.path.exists(dataset), f"file {data_file} or {dataset} does not exists"
            data_file = dataset
        if self.exclude_paralogs:
            processed = "{}.{}nt.v{}.no-paralog.pkl".format(data_file, self.window_size, self.__version__)
        else:
            processed = "{}.{}nt.v{}.pkl".format(data_file, self.window_size, self.__version__)
        if os.path.exists(processed):
            logger.warning(f"Loading processed dataset from {processed}")
            with open(processed, "rb") as infile:
                self.chroms, self.starts, self.genes, self.paralogs, self.ss_data, self.gene_info = pickle.load(infile)
        else:
            with open(data_file) as infile:
                for l in tqdm(infile, desc=f"Loading SpliceAI-{dataset} dataset"):
                    if l.startswith("#"):
                        continue
                    gene, paralog, chrom, strand, tx_start, tx_end, jn_start, jn_end = l.strip().split("\t")
                    if self.exclude_paralogs and int(paralog) != 0:
                        continue
                    tx_start, tx_end = int(tx_start) - 1, int(tx_end)
                    self.gene_info[gene] = (chrom, int(tx_start), int(tx_end), strand, int(paralog))
                    jn_start = [int(x) - 1 for x in jn_start.split(",")[:-1]]
                    jn_end = [int(x) - 1 for x in jn_end.split(",")[:-1]]
                    self.ss_data[gene] = np.zeros(shape=(tx_end - tx_start, ), dtype=np.int8)
                    skip = set(jn_start).intersection(set(jn_end))
                    jn_start = sorted(set(jn_start).difference(skip))
                    jn_end = sorted(set(jn_end).difference(skip))
                    if strand == "-":
                        jn_start, jn_end = jn_end, jn_start
                    jn_start = np.array(jn_start, dtype=np.int32)
                    jn_end = np.array(jn_end, dtype=np.int32)
                    skip = np.array(list(skip), dtype=np.int32)
                    jn_start = jn_start[(jn_start >= tx_start) & (jn_start < tx_end)]
                    jn_end = jn_end[(jn_end >= tx_start) & (jn_end < tx_end)]
                    skip = skip[(skip >= tx_start) & (skip < tx_end)]
                    self.ss_data[gene][jn_start - tx_start] = 1
                    self.ss_data[gene][jn_end - tx_start] = 2
                    self.ss_data[gene][skip - tx_start] = -100
                    self.ss_data[gene] = csr_matrix(self.ss_data[gene])
                    for start in range(tx_start, tx_end, self.window_size):
                        self.starts.append(start)
                        self.chroms.append(chrom)
                        self.genes.append(gene)
                        self.paralogs.append(int(paralog))

            self.starts = np.array(self.starts, dtype=np.int32)
            self.chroms = np.array(self.chroms)
            self.genes = np.array(self.genes)
            self.paralogs = np.array(self.paralogs, dtype=np.int8)
            with open(processed, "wb") as outfile:
                pickle.dump((self.chroms, self.starts, self.genes, self.paralogs, self.ss_data, self.gene_info), outfile)
    
    @classmethod
    def check_dataset(cls, dataset: Literal["gtex", "canonical"]="gtex"):
        ds = cls(dataset)
        from collections import defaultdict
        donor_count, acceptor_count = defaultdict(int), defaultdict(int)
        loader = DataLoader(ds, batch_size=16, shuffle=True, num_workers=16)
        n_donors, n_acceptors, n_unknown = 0, 0, 0
        for seq, ss, index in tqdm(loader, desc="checking dataset"):
            for i in range(len(seq)):
                seq_i = seq[i].numpy()
                ss_i = ss[i].numpy()
                if ds.to_bert_encoding:
                    seq_i = seq_i[1:-1]
                    ss_i = ss_i[1:-1]

                seq_center = seq_i[ds.window_size:2*ds.window_size]
                donors = np.where(ss_i == 1)[0] + ds.window_size
                acceptors = np.where(ss_i == 2)[0] + ds.window_size
                for d in donors:
                    donor_count["{}{}".format(*seq_i[d + 1:d + 3])] += 1
                    n_donors += 1
                for a in acceptors:
                    acceptor_count["{}{}".format(*seq_i[a-2:a])] += 1
                    n_acceptors += 1
                n_unknown += np.where((ss_i == -100) & (seq_center != 0))[0].shape[0]
        return donor_count, acceptor_count, n_donors, n_acceptors, n_unknown
    
    def get_split(self, seed: int=0, num_val_genes=1000, test_chroms: List[str]=["chr1", "chr3", "chr5", "chr7", "chr9"]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""
        Returns:
            train_inds: indices of training samples
            val_inds: indices of validation samples
            test_inds: indices of test samples
        """
        original_state = np.random.get_state()
        np.random.seed(seed)
        indices = np.arange(len(self))
        is_test = np.isin(self.chroms, test_chroms)
        test_inds = indices[is_test & (self.paralogs == 0)]

        val_genes = np.random.choice(np.unique(self.genes[(~is_test) & (self.paralogs == 0)]), size=num_val_genes, replace=False)
        is_valid = np.isin(self.genes, val_genes)
        val_inds = indices[is_valid]
        train_inds = indices[(~is_test) & (~is_valid)]
        self.train_genes = set(self.genes[train_inds])
        self.val_genes = set(self.genes[is_valid])
        self.test_genes = set(self.genes[test_inds])
        assert len(self.train_genes.intersection(self.val_genes)) == 0
        assert len(self.train_genes.intersection(self.test_genes)) == 0
        assert len(self.val_genes.intersection(self.test_genes)) == 0

        np.random.set_state(original_state)
        return train_inds, val_inds, test_inds