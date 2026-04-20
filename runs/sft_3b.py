#!/usr/bin/env python3
"""
Supervised fine-tuning (SFT) for OpenMythos-3B.

Loads the latest pretrain checkpoint, fine-tunes on Alpaca-style instruction
data with loss masked to assistant tokens only. FSDP-ready.

Launch:
    torchrun --standalone --nproc_per_node=8 runs/sft_3b.py

Env overrides:
    MYTHOS_PRETRAIN_DIR=checkpoints/pretrain
    MYTHOS_SFT_DIR=checkpoints/sft
    MYTHOS_SFT_DATASET=yahma/alpaca-cleaned
    MYTHOS_SFT_EPOCHS=2
    MYTHOS_SFT_LR=2e-5
"""

import os
import sys
import math
import time
import random
from pathlib import Path
from contextlib import nullcontext

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import torch
import torch.nn as nn
import torch.distributed as dist
from loguru import logger
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    MixedPrecision,
    FullStateDictConfig,
    StateDictType,
)
from torch.distributed.fsdp.wrap import ModuleWrapPolicy

from datasets import load_dataset

from open_mythos import OpenMythos
from open_mythos.main import TransformerBlock, RecurrentBlock
from open_mythos.variants import mythos_3b
from open_mythos.tokenizer import MythosTokenizer


IGNORE_INDEX = -100

PROMPT_TEMPLATE = (
    "### Instruction:\n{instruction}\n\n"
    "{maybe_input}"
    "### Response:\n"
)
PROMPT_WITH_INPUT = "### Input:\n{input}\n\n"
EOS_TEXT = "<|endoftext|>"


def format_alpaca(example):
    instr = example.get("instruction", "") or ""
    inp = example.get("input", "") or ""
    out = example.get("output", "") or ""
    maybe_input = PROMPT_WITH_INPUT.format(input=inp) if inp.strip() else ""
    prompt = PROMPT_TEMPLATE.format(instruction=instr, maybe_input=maybe_input)
    return prompt, out + EOS_TEXT


class AlpacaIterable(IterableDataset):
    def __init__(self, name, tok, seq_len, rank, world, epochs):
        self.name = name
        self.tok = tok
        self.seq_len = seq_len
        self.rank = rank
        self.world = world
        self.epochs = epochs

    def _iter_once(self):
        w = get_worker_info()
        nw = w.num_workers if w else 1
        wid = w.id if w else 0
        total = self.world * nw
        idx = self.rank * nw + wid
        ds = load_dataset(self.name, split="train")
        n = len(ds)
        for i in range(idx, n, total):
            ex = ds[i]
            p, r = format_alpaca(ex)
            p_ids = self.tok.encode(p)
            r_ids = self.tok.encode(r)
            ids = (p_ids + r_ids)[: self.seq_len + 1]
            if len(ids) < 2:
                continue
            labels = [IGNORE_INDEX] * min(len(p_ids), len(ids)) + ids[len(p_ids):]
            labels = labels[: self.seq_len + 1]
            pad = self.seq_len + 1 - len(ids)
            if pad > 0:
                ids = ids + [0] * pad
                labels = labels + [IGNORE_INDEX] * pad
            x = torch.tensor(ids[:-1], dtype=torch.long)
            y = torch.tensor(labels[1:], dtype=torch.long)
            yield x, y

    def __iter__(self):
        for _ in range(self.epochs):
            yield from self._iter_once()


def _list_ckpts(d):
    if not os.path.isdir(d):
        return []
    return sorted(
        os.path.join(d, f)
        for f in os.listdir(d)
        if f.startswith("step_") and f.endswith(".pt")
    )


def load_model_only(model, path, ddp):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if ddp:
        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
        ):
            model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt["model"])
    return int(ckpt.get("step", 0))


def save_ckpt(model, opt, step, cfg, vocab_size, ckpt_dir, ddp, master):
    if ddp:
        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        ):
            ms = model.state_dict()
            os_ = FSDP.optim_state_dict(model, opt)
    else:
        ms = model.state_dict()
        os_ = opt.state_dict()
    if not master:
        return
    os.makedirs(ckpt_dir, exist_ok=True)
    p = os.path.join(ckpt_dir, f"step_{step:07d}.pt")
    tmp = p + ".tmp"
    torch.save({"step": step, "model": ms, "optimizer": os_, "cfg": cfg,
                "vocab_size": vocab_size}, tmp)
    os.replace(tmp, p)
    for old in _list_ckpts(ckpt_dir)[:-3]:
        try:
            os.remove(old)
        except OSError:
            pass
    logger.success(f"[sft] checkpoint → {p}")


def main():
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        dist.init_process_group("nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(device)
    else:
        rank = local_rank = 0
        world = 1
        device = "cuda" if torch.cuda.is_available() else "cpu"
    master = rank == 0

    tb_writer = None
    if master:
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_dir = os.environ.get(
                "MYTHOS_TB_DIR", str(REPO / "logs" / "tb" / "sft")
            )
            os.makedirs(tb_dir, exist_ok=True)
            tb_writer = SummaryWriter(tb_dir)
            logger.info(f"[sft] tensorboard → {tb_dir}")
        except ImportError:
            logger.warning("[sft] tensorboard not installed; skipping TB logs")

    pretrain_dir = os.environ.get("MYTHOS_PRETRAIN_DIR",
                                  str(REPO / "checkpoints" / "pretrain"))
    sft_dir = os.environ.get("MYTHOS_SFT_DIR",
                             str(REPO / "checkpoints" / "sft"))
    dataset_name = os.environ.get("MYTHOS_SFT_DATASET", "yahma/alpaca-cleaned")
    epochs = int(os.environ.get("MYTHOS_SFT_EPOCHS", "2"))
    lr = float(os.environ.get("MYTHOS_SFT_LR", "2e-5"))
    seq_len = int(os.environ.get("MYTHOS_SEQ_LEN", "2048"))
    micro_batch = int(os.environ.get("MYTHOS_MICRO_BATCH", "2"))
    grad_accum = max(1, 64 // (world * micro_batch))
    warmup = 100
    log_every = 10
    ckpt_every = int(os.environ.get("MYTHOS_SFT_CKPT_EVERY", "500"))

    tok = MythosTokenizer()
    vocab_size = tok.vocab_size

    cfg = mythos_3b()
    cfg.vocab_size = vocab_size
    cfg.max_seq_len = seq_len

    bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if bf16_ok else torch.float16
    model = OpenMythos(cfg)

    if ddp:
        mp = MixedPrecision(
            param_dtype=amp_dtype, reduce_dtype=amp_dtype, buffer_dtype=amp_dtype
        )
        model = FSDP(
            model,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            mixed_precision=mp,
            auto_wrap_policy=ModuleWrapPolicy({TransformerBlock, RecurrentBlock}),
            device_id=local_rank,
        )
        amp_ctx = nullcontext()
    else:
        model = model.to(device)
        amp_ctx = (
            torch.amp.autocast(device_type="cuda", dtype=amp_dtype)
            if "cuda" in device
            else nullcontext()
        )

    pre = _list_ckpts(pretrain_dir)
    if not pre:
        if master:
            logger.warning(f"[sft] no pretrain checkpoint at {pretrain_dir}; "
                           "training from scratch is a bad idea for SFT.")
    else:
        if master:
            logger.info(f"[sft] loading pretrain weights from {pre[-1]}")
        load_model_only(model, pre[-1], ddp)

    opt = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=0.0, betas=(0.9, 0.95), fused=True
    )

    start_step = 0
    existing = _list_ckpts(sft_dir)
    if existing:
        if master:
            logger.info(f"[sft] resuming sft from {existing[-1]}")
        ckpt = torch.load(existing[-1], map_location="cpu", weights_only=False)
        if ddp:
            with FSDP.state_dict_type(
                model,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
            ):
                model.load_state_dict(ckpt["model"])
                ost = FSDP.optim_state_dict_to_load(
                    model=model, optim=opt, optim_state_dict=ckpt["optimizer"]
                )
                opt.load_state_dict(ost)
        else:
            model.load_state_dict(ckpt["model"])
            opt.load_state_dict(ckpt["optimizer"])
        start_step = int(ckpt["step"])

    ds = AlpacaIterable(dataset_name, tok, seq_len, rank, world, epochs)
    loader = DataLoader(ds, batch_size=micro_batch, num_workers=2, pin_memory=True)

    if master:
        os.makedirs(sft_dir, exist_ok=True)
        logger.info(f"[sft] dataset={dataset_name} epochs={epochs} lr={lr} "
                    f"seq_len={seq_len} micro={micro_batch} grad_accum={grad_accum}")

    model.train()
    step = start_step
    t0 = time.perf_counter()
    it = iter(loader)
    opt.zero_grad()
    loss_acc = 0.0
    micro = 0
    done = False
    while not done:
        try:
            x, y = next(it)
        except StopIteration:
            done = True
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        sync = nullcontext() if (not ddp or micro == grad_accum - 1) else model.no_sync()
        if step < warmup:
            cur_lr = lr * (step + 1) / warmup
            for g in opt.param_groups:
                g["lr"] = cur_lr
        with sync, amp_ctx:
            logits = model(x)
            loss = nn.functional.cross_entropy(
                logits.view(-1, vocab_size), y.view(-1),
                ignore_index=IGNORE_INDEX,
            ) / grad_accum
        loss.backward()
        loss_acc += loss.item()
        micro += 1
        if micro == grad_accum:
            if ddp:
                gn = model.clip_grad_norm_(1.0)
            else:
                gn = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad()
            step += 1
            if master and step % log_every == 0:
                dt = time.perf_counter() - t0
                cur_lr = opt.param_groups[0]["lr"]
                logger.info(
                    f"[sft] step {step} loss {loss_acc:.4f} "
                    f"gnorm {float(gn):.2f} lr {cur_lr:.2e} dt {dt:.1f}s"
                )
                if tb_writer:
                    tb_writer.add_scalar("sft/loss", loss_acc, step)
                    tb_writer.add_scalar("sft/grad_norm", float(gn), step)
                    tb_writer.add_scalar("sft/lr", cur_lr, step)
                    tb_writer.flush()
                t0 = time.perf_counter()
            if step % ckpt_every == 0:
                save_ckpt(model, opt, step, cfg, vocab_size, sft_dir, ddp, master)
            loss_acc = 0.0
            micro = 0

    save_ckpt(model, opt, step, cfg, vocab_size, sft_dir, ddp, master)

    if tb_writer:
        tb_writer.close()
    if ddp:
        dist.barrier()
        dist.destroy_process_group()
    if master:
        logger.success("[sft] done.")


if __name__ == "__main__":
    main()
