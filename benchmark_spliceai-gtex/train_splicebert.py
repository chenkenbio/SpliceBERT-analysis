#!/usr/bin/env python3
"""
Author: Ken Chen
Email: chenkenbio@gmail.com
Date: 2024-01-16
"""

import argparse
import os
import sys
import warnings
import numpy as np
from tqdm import tqdm
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
sys.path.append("../scripts")
from utils import make_logger, get_run_info, make_directory, set_seed, model_summary
from splicebert_model import BertForTokenClassification
from spliceai_data import SpliceaiDataset
from transformers import get_polynomial_decay_schedule_with_warmup
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple, Union
import logging
logger = logging.getLogger(__name__)


torch.set_float32_matmul_precision("medium")

def topk_acc_1d(labels, scores):
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    assert labels.ndim == 1 and scores.ndim == 1 and labels.shape[0] == scores.shape[0], "{}".format((labels.shape, scores.shape))
    assert np.issubdtype(labels.dtype, np.bool8) or (np.issubdtype(labels.dtype, np.integer) and labels.max() <= 1 and labels.min() >= 0), "{}".format((labels.dtype, labels.min(), labels.max()))
    labels = labels.astype(np.int8)
    inds = np.where(labels == 1)[0]
    if len(inds) == 0:
        warnings.warn("number of positive sample is 0, return np.nan")
        return np.nan
    else:
        q = np.quantile(scores, q=1 - len(inds) / len(labels))
        return np.where(scores[inds] >= q)[0].shape[0] / max(1, np.where(scores >= q)[0].shape[0])


def get_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data", default="gtex")
    p.add_argument("--genome", '-g', type=str, required=True, help="genome in hdf5 format")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--encoding-type", choices=["splicebert", "dna5"], default="splicebert")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--model", '-m', default="../../models/SpliceBERT.1024nt")
    p.add_argument("--num-workers", type=int, default=8, help="number of workers for dataloader")
    p.add_argument('-a', "--rdrop-weight", type=float, default=1, help="rdrop weight")
    p.add_argument("--prefetch-factor", type=int, default=4, help="prefetch factor for dataloader")
    p.add_argument("--resume")
    p.add_argument("--init-weights", )
    p.add_argument('-o', "--outdir", required=True, help="output directory")
    p.add_argument('--seed', type=int, default=2020)
    return p

def main(args: argparse.Namespace) -> None:
    # set seed
    set_seed(args.seed)
    outdir = make_directory(args.outdir)
    logger = make_logger(title="", filename=os.path.join(outdir, "train.log"), no_prefix=True)

    logger.info(get_run_info(sys.argv, args=args))

    ds = SpliceaiDataset(
        genome=args.genome, 
        dataset=args.data, 
        encoding_type="splicebert", 
        shift=23, 
        window_size=300)
    train_inds, val_inds, test_inds = ds.get_split()

    # logger.info("train ({} genes): {}".format(len(np.unique(train_ds.genes)), np.unique(train_ds.chroms, return_counts=True)))
    # logger.info("valid ({} genes): {}".format(len(np.unique(valid_ds.genes)), np.unique(valid_ds.chroms, return_counts=True)))
    # logger.info("test ({} genes): {}".format(len(np.unique(test_ds.genes)), np.unique(test_ds.chroms, return_counts=True)))
    dl_args = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
    }
    if args.num_workers > 0 and args.prefetch_factor > 0:
        dl_args["prefetch_factor"] = args.prefetch_factor

    train_loader = DataLoader(Subset(ds, indices=train_inds), shuffle=True, **dl_args)
    valid_loader = DataLoader(Subset(ds, indices=val_inds), shuffle=False, **dl_args)
    test_loader = DataLoader(Subset(ds, indices=test_inds), shuffle=False, **dl_args)
    torch.save(next(iter(train_loader)), f"{args.outdir}/demo_data.pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertForTokenClassification.from_pretrained(args.model, num_labels=3).to(device)
    # model = SpliceaiForTokenClassification(
    #     encoding_type="dna5", 
    #     vocab_size=5,
    #     num_classes=3, 
    #     flanking_size=5001 if args.encoding_type == "splicebert" else 5000, 
    #     kernel_size=KERNEL_SIZE, 
    #     atrous_rate=ATROUS_RATE,
    # ).to(device)
    if args.init_weights:
        logger.info(f"load weights from {args.init_weights}")
        w = torch.load(args.init_weights, map_location="cpu")
        model.load_state_dict(w)
    logger.info("{}\n{}".format(model, model_summary(model)))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-6)
    logger.info(f"optimizer: {optimizer}")
    K = max(1, 128 // args.batch_size)

    scheduler = get_polynomial_decay_schedule_with_warmup(
        optimizer,
        num_warmup_steps=3000 // K,
        num_training_steps=100 * len(train_loader) // K,
        lr_end=1e-6,
        power=1.0,
    )
    scaler = GradScaler()
    writer = SummaryWriter(log_dir=outdir + "/tensorboard")

    if args.resume is not None:
        logger.warning(f"resume from {args.resume}")
        d = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(d["model"])
        optimizer.load_state_dict(d["optimizer"])
        scaler.load_state_dict(d["scaler"])
        scheduler.load_state_dict(d["scheduler"])
    
    best_score = 0
    num_steps = 0
    wait = 0
    for epoch in range(100):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} train")

        for it, (seq, label, _) in enumerate(pbar):
            seq, label = seq.to(device).long(), label.to(device).long()
            with autocast():
                logits = model(seq).logits[:, 301:601]
                dim = logits.shape[-1]
                logits = logits.reshape(-1, dim)
                label = label.reshape(-1)
                ss_loss = F.cross_entropy(logits, label)
                # if args.rdrop_weight > 0:
                if args.rdrop_weight > 0:
                    logits_rdrop = model(seq).logits[:, 301:601]
                    logits_rdrop = logits_rdrop.reshape(-1, dim)
                    k = (label >= 0)
                    n = k.sum()
                    p = torch.log_softmax(logits[k, :], dim=-1)
                    p_tec = torch.softmax(logits[k, :], dim=-1)
                    q = torch.log_softmax(logits_rdrop[k, :], dim=-1)
                    q_tec = torch.softmax(logits_rdrop[k, :], dim=-1)
                    # kl_loss = F.kl_div(p, q_tec, reduction="none").sum() / n
                    # kl_loss += F.kl_div(q, p_tec, reduction="none").sum() / n
                    kl_loss = F.kl_div(p, q_tec, reduction="batchmean")
                    kl_loss += F.kl_div(q, p_tec, reduction="batchmean")

                    del logits_rdrop, p, p_tec, q, q_tec
                else:
                    kl_loss = torch.tensor(0.0, device=device)
                loss = ss_loss + kl_loss * args.rdrop_weight
                del logits

            scaler.scale(loss).backward()
            pbar.set_postfix_str("loss(ss/kl)={:.3g}({:.3g},{:.3g}), lr={:.3g}".format(
                loss.item(), 
                ss_loss.item(), 
                kl_loss.item(), 
                optimizer.param_groups[-1]["lr"]
            ), refresh=False)
            writer.add_scalars(
                "train", 
                {
                    "loss": loss.item(), 
                    "ss_loss": ss_loss.item(), 
                    "kl_loss": kl_loss.item(), 
                },
                global_step=num_steps
            )
            num_steps += 1
            del loss, ss_loss, kl_loss

            if (it + 1) % K == 0 or it == len(pbar) - 1:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
        
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "scheduler": scheduler.state_dict(),
            }, f=f"{args.outdir}/checkpoint.pt")
        
        model.eval()
        val_d_acc, val_a_acc = test_model(model, valid_loader)
        writer.add_scalar("valid/d_acc", val_d_acc, global_step=epoch)
        writer.add_scalar("valid/a_acc", val_a_acc, global_step=epoch)
        logger.info("Epoch {} valid: d_acc={:.4g}, a_acc={:.4g}".format(epoch, val_d_acc, val_a_acc))

        test_d_acc, test_a_acc = test_model(model, test_loader)
        writer.add_scalar("test/d_acc", test_d_acc, global_step=epoch)
        writer.add_scalar("test/a_acc", test_a_acc, global_step=epoch)

        logger.info("Epoch {} test: d_acc={:.4g}, a_acc={:.4g}".format(epoch, test_d_acc, test_a_acc))

        val_score = (val_d_acc + val_a_acc) / 2

        torch.save(model.state_dict() , f="{}/model_state_dict.epoch{}.pt".format(args.outdir, epoch))
        if val_score > best_score:
            wait = 0
            best_score = val_score
            logger.info("save model\n")
            # model.save_pretrained(args.outdir + "/best_model")
            torch.save(model.state_dict() , f="{}/best_model.pt".format(args.outdir))
        else:
            wait += 1
            logger.info("wait: {}\n".format(wait))
            if wait >= 20:
                break


@torch.no_grad()
@autocast()
def test_model(model: nn.Module, loader: DataLoader):
    model.eval()
    device = next(model.parameters()).device
    all_scores, all_labels = [], []
    for it, (seq, label, _) in enumerate(tqdm(loader, desc="predicting")):
        seq, label = seq.to(device), label.to(device)
        score = model(seq).logits[:, 301:601]
        dim = score.shape[-1]
        score = score.reshape(-1, dim)
        label = label.reshape(-1)
        keep = torch.where(label != -100)[0]
        score = torch.softmax(score[keep], dim=-1)[:, 1:].cpu().numpy().astype(np.float16)
        label = label[keep] #.cpu().numpy().astype(np.int8)
        donor_label = (label == 1).cpu().numpy().astype(np.int8)
        acceptor_label = (label == 2).cpu().numpy().astype(np.int8)
        del label
        all_scores.append(score)
        all_labels.append(np.stack([donor_label, acceptor_label], axis=1))
    all_scores = np.concatenate(all_scores)
    all_labels = np.concatenate(all_labels)

    donor_acc = topk_acc_1d(all_labels[:, 0], all_scores[:, 0])
    acceptor_acc = topk_acc_1d(all_labels[:, 1], all_scores[:, 1])
    return donor_acc, acceptor_acc


if __name__ == "__main__":
    main(get_args().parse_args())
