#!/usr/bin/env python3
"""
Direct Preference Optimization (DPO) for OpenMythos-3B.

Loss:
    L = -log sigmoid( beta * ( log pi(y_w|x)/pi_ref(y_w|x)
                              - log pi(y_l|x)/pi_ref(y_l|x) ) )

Policy starts from the SFT checkpoint. Reference is a frozen copy of the same
SFT checkpoint (no grad). FSDP-ready on both.

Launch:
    torchrun --standalone --nproc_per_node=8 runs/dpo_3b.py

Env overrides:
    MYTHOS_SFT_DIR=checkpoints/sft
    MYTHOS_DPO_DIR=checkpoints/dpo
    MYTHOS_DPO_DATASET=Intel/orca_dpo_pairs
    MYTHOS_DPO_EPOCHS=1
    MYTHOS_DPO_LR=5e-7
    MYTHOS_DPO_BETA=0.1
"""

import os
import sys
import time
from pathlib import Path
from contextlib import nullcontext

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import torch
import torch.nn as nn
import torch.nn.functional as F
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


def save_policy(model, opt, step, cfg, vocab_size, ckpt_dir, ddp, master):
    if ddp:
        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        ):
            ms = model.state_dict()
            ostate = FSDP.optim_state_dict(model, opt)
    else:
        ms = model.state_dict()
        ostate = opt.state_dict()
    if not master:
        return
    os.makedirs(ckpt_dir, exist_ok=True)
    p = os.path.join(ckpt_dir, f"step_{step:07d}.pt")
    tmp = p + ".tmp"
    torch.save({"step": step, "model": ms, "optimizer": ostate, "cfg": cfg,
                "vocab_size": vocab_size}, tmp)
    os.replace(tmp, p)
    for old in _list_ckpts(ckpt_dir)[:-3]:
        try:
            os.remove(old)
        except OSError:
            pass
    logger.success(f"[dpo] checkpoint → {p}")


def build_prompt(example):
    """Support multiple dataset schemas. Returns (prompt, chosen, rejected)."""
    # Intel/orca_dpo_pairs: system, question, chosen, rejected
    if "question" in example and "chosen" in example and "rejected" in example:
        sys_ = example.get("system", "") or ""
        q = example["question"]
        instr = f"{sys_}\n\n{q}".strip() if sys_ else q
        prompt = PROMPT_TEMPLATE.format(instruction=instr, maybe_input="")
        return prompt, example["chosen"] + EOS_TEXT, example["rejected"] + EOS_TEXT
    # HuggingFaceH4/ultrafeedback_binarized: prompt, chosen(list), rejected(list)
    if "prompt" in example and "chosen" in example:
        p = example["prompt"]
        prompt = PROMPT_TEMPLATE.format(instruction=p, maybe_input="")
        c = example["chosen"]
        r = example["rejected"]
        c_text = c[-1]["content"] if isinstance(c, list) else c
        r_text = r[-1]["content"] if isinstance(r, list) else r
        return prompt, c_text + EOS_TEXT, r_text + EOS_TEXT
    raise ValueError(f"Unsupported DPO schema: {list(example.keys())}")


class DPOIterable(IterableDataset):
    def __init__(self, name, tok, seq_len, rank, world, epochs):
        self.name = name
        self.tok = tok
        self.seq_len = seq_len
        self.rank = rank
        self.world = world
        self.epochs = epochs

    def _one_side(self, prompt_ids, resp_ids):
        ids = (prompt_ids + resp_ids)[: self.seq_len + 1]
        labels = ([IGNORE_INDEX] * min(len(prompt_ids), len(ids))
                  + ids[len(prompt_ids):])
        labels = labels[: self.seq_len + 1]
        pad = self.seq_len + 1 - len(ids)
        if pad > 0:
            ids = ids + [0] * pad
            labels = labels + [IGNORE_INDEX] * pad
        return (
            torch.tensor(ids[:-1], dtype=torch.long),
            torch.tensor(labels[1:], dtype=torch.long),
        )

    def _iter_once(self):
        w = get_worker_info()
        nw = w.num_workers if w else 1
        wid = w.id if w else 0
        total = self.world * nw
        idx = self.rank * nw + wid
        ds = load_dataset(self.name, split="train")
        n = len(ds)
        for i in range(idx, n, total):
            try:
                p, c, r = build_prompt(ds[i])
            except Exception:
                continue
            p_ids = self.tok.encode(p)
            if len(p_ids) >= self.seq_len:
                continue
            c_ids = self.tok.encode(c)
            r_ids = self.tok.encode(r)
            xc, yc = self._one_side(p_ids, c_ids)
            xr, yr = self._one_side(p_ids, r_ids)
            yield xc, yc, xr, yr

    def __iter__(self):
        for _ in range(self.epochs):
            yield from self._iter_once()


def sum_logprobs(logits, labels):
    """Sum log-probs of `labels` under `logits`, ignoring IGNORE_INDEX positions.

    logits: (B, T, V), labels: (B, T). Returns (B,).
    """
    lp = F.log_softmax(logits.float(), dim=-1)
    mask = labels != IGNORE_INDEX
    safe_labels = labels.masked_fill(~mask, 0).unsqueeze(-1)
    gathered = lp.gather(-1, safe_labels).squeeze(-1)
    gathered = gathered * mask
    return gathered.sum(dim=-1)


def build_model(cfg, ddp, local_rank, amp_dtype):
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
    return model


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
                "MYTHOS_TB_DIR", str(REPO / "logs" / "tb" / "dpo")
            )
            os.makedirs(tb_dir, exist_ok=True)
            tb_writer = SummaryWriter(tb_dir)
            logger.info(f"[dpo] tensorboard → {tb_dir}")
        except ImportError:
            logger.warning("[dpo] tensorboard not installed; skipping TB logs")

    sft_dir = os.environ.get("MYTHOS_SFT_DIR", str(REPO / "checkpoints" / "sft"))
    dpo_dir = os.environ.get("MYTHOS_DPO_DIR", str(REPO / "checkpoints" / "dpo"))
    dataset_name = os.environ.get("MYTHOS_DPO_DATASET", "Intel/orca_dpo_pairs")
    epochs = int(os.environ.get("MYTHOS_DPO_EPOCHS", "1"))
    lr = float(os.environ.get("MYTHOS_DPO_LR", "5e-7"))
    beta = float(os.environ.get("MYTHOS_DPO_BETA", "0.1"))
    seq_len = int(os.environ.get("MYTHOS_SEQ_LEN", "1024"))
    micro_batch = int(os.environ.get("MYTHOS_MICRO_BATCH", "1"))
    grad_accum = max(1, 32 // (world * micro_batch))
    log_every = 10
    ckpt_every = int(os.environ.get("MYTHOS_DPO_CKPT_EVERY", "500"))

    tok = MythosTokenizer()
    vocab_size = tok.vocab_size
    cfg = mythos_3b()
    cfg.vocab_size = vocab_size
    cfg.max_seq_len = seq_len

    bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if bf16_ok else torch.float16

    policy = build_model(cfg, ddp, local_rank, amp_dtype)
    ref = build_model(cfg, ddp, local_rank, amp_dtype)
    if not ddp:
        policy = policy.to(device)
        ref = ref.to(device)

    sft = _list_ckpts(sft_dir)
    if not sft:
        if master:
            logger.error(f"[dpo] no SFT checkpoint in {sft_dir}; aborting.")
        if ddp:
            dist.destroy_process_group()
        return
    if master:
        logger.info(f"[dpo] loading SFT weights into policy + ref: {sft[-1]}")
    load_model_only(policy, sft[-1], ddp)
    load_model_only(ref, sft[-1], ddp)

    for p in ref.parameters():
        p.requires_grad_(False)
    ref.eval()

    opt = torch.optim.AdamW(
        policy.parameters(), lr=lr, weight_decay=0.0, betas=(0.9, 0.95), fused=True
    )

    ds = DPOIterable(dataset_name, tok, seq_len, rank, world, epochs)
    loader = DataLoader(ds, batch_size=micro_batch, num_workers=2, pin_memory=True)

    amp_ctx = (
        nullcontext() if ddp
        else (torch.amp.autocast(device_type="cuda", dtype=amp_dtype)
              if "cuda" in device else nullcontext())
    )

    if master:
        os.makedirs(dpo_dir, exist_ok=True)
        logger.info(f"[dpo] dataset={dataset_name} beta={beta} lr={lr} "
                    f"seq_len={seq_len} micro={micro_batch} grad_accum={grad_accum}")

    policy.train()
    opt.zero_grad()
    step = 0
    micro = 0
    loss_acc = 0.0
    acc_acc = 0.0
    t0 = time.perf_counter()

    for xc, yc, xr, yr in loader:
        xc = xc.to(device, non_blocking=True)
        yc = yc.to(device, non_blocking=True)
        xr = xr.to(device, non_blocking=True)
        yr = yr.to(device, non_blocking=True)

        sync = nullcontext() if (not ddp or micro == grad_accum - 1) else policy.no_sync()
        with sync, amp_ctx:
            lc = policy(xc)
            lr_ = policy(xr)
            pol_c = sum_logprobs(lc, yc)
            pol_r = sum_logprobs(lr_, yr)
            with torch.no_grad():
                rc = ref(xc)
                rr = ref(xr)
                ref_c = sum_logprobs(rc, yc)
                ref_r = sum_logprobs(rr, yr)
            logits = beta * ((pol_c - ref_c) - (pol_r - ref_r))
            loss = -F.logsigmoid(logits).mean() / grad_accum
            acc = (logits > 0).float().mean().item()
        loss.backward()
        loss_acc += loss.item()
        acc_acc += acc
        micro += 1
        if micro == grad_accum:
            if ddp:
                gn = policy.clip_grad_norm_(1.0)
            else:
                gn = nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
            opt.zero_grad()
            step += 1
            if master and step % log_every == 0:
                dt = time.perf_counter() - t0
                acc_mean = acc_acc / grad_accum
                logger.info(
                    f"[dpo] step {step} loss {loss_acc:.4f} "
                    f"acc {acc_mean:.2%} gnorm {float(gn):.2f} dt {dt:.1f}s"
                )
                if tb_writer:
                    tb_writer.add_scalar("dpo/loss", loss_acc, step)
                    tb_writer.add_scalar("dpo/accuracy", acc_mean, step)
                    tb_writer.add_scalar("dpo/grad_norm", float(gn), step)
                    tb_writer.add_scalar("dpo/lr", opt.param_groups[0]["lr"], step)
                    tb_writer.flush()
                t0 = time.perf_counter()
            if step % ckpt_every == 0:
                save_policy(policy, opt, step, cfg, vocab_size, dpo_dir, ddp, master)
            loss_acc = 0.0
            acc_acc = 0.0
            micro = 0

    save_policy(policy, opt, max(step, 1), cfg, vocab_size, dpo_dir, ddp, master)

    if tb_writer:
        tb_writer.close()
    if ddp:
        dist.barrier()
        dist.destroy_process_group()
    if master:
        logger.success("[dpo] done.")


if __name__ == "__main__":
    main()
