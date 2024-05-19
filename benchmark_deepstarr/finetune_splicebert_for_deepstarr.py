#!/usr/bin/env python3
"""
Author: Ken Chen (https://github.com/chenkenbio)
Date: 2024-05-19
"""

import argparse
import os
import sys
import math
import numpy as np
from tqdm import tqdm
from splicebert_flash import BertForSequenceClassification
import pandas as pd
import torch
from torch import Tensor
from scipy.stats import pearsonr, spearmanr
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, Subset, RandomSampler, DistributedSampler
from torch.cuda.amp import autocast, GradScaler
from deepstarr_data import DeepstarrData
# from biock2.utils import make_directory


def get_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--batch-size", "-b", type=int, default=128)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument('-w', "--weight-decay", type=float, default=0.0)
    p.add_argument('-n', "--num-workers", type=int, default=4)
    p.add_argument("-o", "--outdir", required=True)
    p.add_argument("-m", "--model", default="../../../SpliceBERT.1024nt")
    p.add_argument("--debug", action="store_true")
    p.add_argument('--seed', type=int, default=0)
    return p

def test_model(model, loader):
    model.eval()

    all_dev = list()
    all_hk = list()
    all_dev_preds = list()
    all_hk_preds = list()

    with torch.no_grad():
        for seq, y_dev, y_hk in tqdm(loader, desc="Testing"):
            seq = seq.cuda().long()
            p_dev, p_hk = model(seq).logits.chunk(2, dim=1)

            all_dev.append(y_dev)
            all_hk.append(y_hk)
            all_dev_preds.append(p_dev.detach().cpu().squeeze())
            all_hk_preds.append(p_hk.detach().cpu().squeeze())
    
    all_dev = torch.cat(all_dev).numpy()
    all_hk = torch.cat(all_hk).numpy()
    all_dev_preds = torch.cat(all_dev_preds).numpy()
    all_hk_preds = torch.cat(all_hk_preds).numpy()

    return all_dev, all_hk, all_dev_preds, all_hk_preds


if __name__ == "__main__":
    args = get_args().parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    ds_train = DeepstarrData("train", "splicebert")
    ds_val = DeepstarrData("val", "splicebert")
    ds_test = DeepstarrData("test", "splicebert")

    if not os.path.isdir(args.outdir):
        os.makedirs(args.outdir)

    train_loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = BertForSequenceClassification.from_pretrained(args.model, num_labels=2).cuda()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print(optimizer)

    scaler = GradScaler()

    loss_pool = list()

    best_pcc = 0
    wait = 0

    for epoch in range(300):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for it, (seq, y_dev, y_hk) in enumerate(pbar):
            if args.debug and it > 100:
                break
            seq = seq.cuda().long()

            with autocast():
                p_dev, p_hk = model(seq).logits.chunk(2, dim=1)

                loss_dev = F.mse_loss(p_dev.squeeze(), y_dev.cuda())
                loss_hk = F.mse_loss(p_hk.squeeze(), y_hk.cuda())
                loss = loss_dev + loss_hk

            if len(loss_pool) < 100:
                loss_pool.append(loss.item())
            else:
                loss_pool[it % 100] = loss.item()

            optimizer.zero_grad()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()


            pbar.set_postfix(loss=np.mean(loss_pool))
        

        all_dev, all_hk, all_dev_preds, all_hk_preds = test_model(model, val_loader)

        val_dev_pcc = pearsonr(all_dev, all_dev_preds)[0]
        val_dev_scc = spearmanr(all_dev, all_dev_preds)[0]
        val_hk_pcc = pearsonr(all_hk, all_hk_preds)[0]
        val_hk_scc = spearmanr(all_hk, all_hk_preds)[0]

        print(f"Val Dev PCC: {val_dev_pcc:.4f}, Val Dev SCC: {val_dev_scc:.4f}, Val HK PCC: {val_hk_pcc:.4f}, Val HK SCC: {val_hk_scc:.4f}")

        all_dev, all_hk, all_dev_preds, all_hk_preds = test_model(model, test_loader)

        if val_dev_pcc + val_hk_pcc > best_pcc:
            best_pcc = val_dev_pcc + val_hk_pcc
            wait = 0
            pd.DataFrame({"dev": all_dev, "dev_preds": all_dev_preds, "hk": all_hk, "hk_preds": all_hk_preds}).to_csv(f"{args.outdir}/deepstarr_preds_epoch_{epoch}.csv", index=False)
        else:
            wait += 1


        test_dev_pcc = pearsonr(all_dev, all_dev_preds)[0]
        test_dev_scc = spearmanr(all_dev, all_dev_preds)[0]
        test_hk_pcc = pearsonr(all_hk, all_hk_preds)[0]
        test_hk_scc = spearmanr(all_hk, all_hk_preds)[0]

        print(f"Test Dev PCC: {test_dev_pcc:.4f}, Test Dev SCC: {test_dev_scc:.4f}, Test HK PCC: {test_hk_pcc:.4f}, Test HK SCC: {test_hk_scc:.4f}\n")
        
        if wait > 0:
            print(f"wait: {wait}")
            if wait > args.patience:
                print("Early stopping")
                break