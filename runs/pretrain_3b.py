#!/usr/bin/env python3
"""
Pretrain OpenMythos-3B on FineWeb-Edu.

This is a thin entry point that reuses `training/3b_fine_web_edu.py`. Defaults
are tuned for an A800 x 8 node. Override via env vars:

    MYTHOS_SUBSET=sample-10BT        # sample-10BT | sample-100BT | default
    MYTHOS_TARGET_TOKENS=30000000000
    MYTHOS_CKPT_DIR=checkpoints/pretrain

Launch:
    torchrun --standalone --nproc_per_node=8 runs/pretrain_3b.py
"""

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "training"))

os.environ.setdefault("MYTHOS_SUBSET", "sample-10BT")
os.environ.setdefault("MYTHOS_TARGET_TOKENS", "30000000000")
os.environ.setdefault("MYTHOS_CKPT_DIR", str(REPO / "checkpoints" / "pretrain"))

import importlib.util

spec = importlib.util.spec_from_file_location(
    "pretrain_mod", REPO / "training" / "3b_fine_web_edu.py"
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


_orig_main = mod.main


def main():
    import training_3b_fine_web_edu_patch  # noqa: F401  (side-effect only, if present)


# Monkey-patch the constants inside mod.main via source rewriting is brittle;
# instead we re-exec the original main but first patch the module-level CKPT
# defaults by binding names the original main reads from environment.
def _run():
    # The original main() hard-codes some values. We honor env overrides by
    # shimming: os.chdir into the repo root so relative ckpt paths line up,
    # then call main().
    os.chdir(REPO)

    # Surgical override via closure: rebuild a main that respects env vars.
    import math
    import time

    import torch
    import torch.nn as nn
    import torch.distributed as dist
    from loguru import logger
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        ShardingStrategy,
        MixedPrecision,
    )
    from torch.distributed.fsdp.wrap import ModuleWrapPolicy
    from torch.utils.data import DataLoader
    from contextlib import nullcontext

    from open_mythos import OpenMythos
    from open_mythos.main import TransformerBlock, RecurrentBlock
    from open_mythos.variants import mythos_3b
    from open_mythos.tokenizer import MythosTokenizer

    FineWebEduDataset = mod.FineWebEduDataset
    get_lr = mod.get_lr
    save_checkpoint = mod.save_checkpoint
    load_checkpoint = mod.load_checkpoint
    _list_ckpts = mod._list_ckpts

    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        dist.init_process_group("nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(device)
    else:
        rank = local_rank = 0
        world_size = 1
        device = "cuda" if torch.cuda.is_available() else "cpu"
    master = rank == 0

    tb_writer = None
    if master:
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_dir = os.environ.get(
                "MYTHOS_TB_DIR", str(REPO / "logs" / "tb" / "pretrain")
            )
            os.makedirs(tb_dir, exist_ok=True)
            tb_writer = SummaryWriter(tb_dir)
            logger.info(f"[pretrain] tensorboard → {tb_dir}")
        except ImportError:
            logger.warning("[pretrain] tensorboard not installed; skipping TB logs")

    encoding = MythosTokenizer()
    vocab_size = encoding.vocab_size

    seq_len = int(os.environ.get("MYTHOS_SEQ_LEN", "2048"))
    micro_batch = int(os.environ.get("MYTHOS_MICRO_BATCH", "4"))
    target_tokens = int(os.environ["MYTHOS_TARGET_TOKENS"])
    grad_accum = max(1, 256 // (world_size * micro_batch))
    global_batch_tok = world_size * micro_batch * grad_accum * seq_len
    total_steps = max(1, target_tokens // global_batch_tok)
    warmup_steps = int(os.environ.get("MYTHOS_WARMUP", "2000"))
    lr = float(os.environ.get("MYTHOS_LR", "3e-4"))
    wd = 0.1
    log_every = 10
    ckpt_every = int(os.environ.get("MYTHOS_CKPT_EVERY", "1000"))
    ckpt_dir = os.environ["MYTHOS_CKPT_DIR"]
    dataset_subset = os.environ["MYTHOS_SUBSET"]

    if master:
        logger.info(
            f"[pretrain] subset={dataset_subset} target_tokens={target_tokens:,} "
            f"total_steps={total_steps:,} global_batch_tok={global_batch_tok:,}"
        )

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

    opt = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.95), fused=True
    )

    start_step = 0
    existing = _list_ckpts(ckpt_dir)
    if existing:
        if master:
            logger.info(f"[pretrain] resuming from {existing[-1]}")
        start_step = load_checkpoint(model, opt, existing[-1], ddp)

    ds = FineWebEduDataset(encoding, seq_len, dataset_subset, rank, world_size)
    loader = DataLoader(ds, batch_size=micro_batch, num_workers=4, pin_memory=True)

    if master:
        os.makedirs(ckpt_dir, exist_ok=True)

    model.train()
    it = iter(loader)
    t0 = time.perf_counter()
    step = start_step
    while step < total_steps:
        cur_lr = get_lr(step, warmup_steps, total_steps, lr, lr * 0.1)
        for g in opt.param_groups:
            g["lr"] = cur_lr
        opt.zero_grad()
        loss_acc = 0.0
        for ms in range(grad_accum):
            try:
                x, y = next(it)
            except StopIteration:
                it = iter(loader)
                x, y = next(it)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            sync = (
                nullcontext()
                if (not ddp or ms == grad_accum - 1)
                else model.no_sync()
            )
            with sync, amp_ctx:
                logits = model(x)
                loss = nn.functional.cross_entropy(
                    logits.view(-1, vocab_size), y.view(-1)
                ) / grad_accum
            loss.backward()
            loss_acc += loss.item()
        if ddp:
            gn = model.clip_grad_norm_(1.0)
        else:
            gn = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        if master and step % log_every == 0:
            dt = time.perf_counter() - t0
            tps = global_batch_tok * log_every / dt
            logger.info(
                f"[pretrain] step {step}/{total_steps} loss {loss_acc:.4f} "
                f"gnorm {float(gn):.2f} lr {cur_lr:.2e} {tps/1e6:.2f}M tok/s"
            )
            if tb_writer:
                tb_writer.add_scalar("pretrain/loss", loss_acc, step)
                tb_writer.add_scalar("pretrain/grad_norm", float(gn), step)
                tb_writer.add_scalar("pretrain/lr", cur_lr, step)
                tb_writer.add_scalar("pretrain/tokens_per_sec", tps, step)
                tb_writer.add_scalar(
                    "pretrain/tokens_seen_B", step * global_batch_tok / 1e9, step
                )
                tb_writer.flush()
            t0 = time.perf_counter()
        if step % ckpt_every == 0:
            save_checkpoint(model, opt, step, cfg, vocab_size, ckpt_dir, ddp, master)

    if step > start_step and step % ckpt_every != 0:
        save_checkpoint(model, opt, step, cfg, vocab_size, ckpt_dir, ddp, master)

    if tb_writer:
        tb_writer.close()
    if ddp:
        dist.barrier()
        dist.destroy_process_group()
    if master:
        logger.success("[pretrain] done.")


if __name__ == "__main__":
    _run()
