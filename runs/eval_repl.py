#!/usr/bin/env python3
"""
Interactive REPL to chat with a trained OpenMythos checkpoint.

Usage:
    python runs/eval_repl.py                          # loads latest DPO ckpt
    python runs/eval_repl.py --stage sft              # use latest SFT ckpt
    python runs/eval_repl.py --stage pretrain         # raw pretrain ckpt
    python runs/eval_repl.py --ckpt path/to/step.pt   # explicit file

At the `>>>` prompt:
    :n 24         set inference loop count (depth extrapolation)
    :t 0.8        set sampling temperature
    :k 50         set top-k
    :m 256        set max_new_tokens
    :reset        clear the session
    :quit         exit
"""

import os
import sys
import argparse
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import torch

from open_mythos import OpenMythos
from open_mythos.variants import mythos_3b
from open_mythos.tokenizer import MythosTokenizer


PROMPT_TEMPLATE = (
    "### Instruction:\n{instruction}\n\n"
    "### Response:\n"
)
EOS_TEXT = "<|endoftext|>"


def _list_ckpts(d):
    if not os.path.isdir(d):
        return []
    return sorted(
        os.path.join(d, f)
        for f in os.listdir(d)
        if f.startswith("step_") and f.endswith(".pt")
    )


def find_ckpt(stage: str, explicit: str | None) -> str:
    if explicit:
        if not os.path.isfile(explicit):
            raise SystemExit(f"checkpoint not found: {explicit}")
        return explicit
    for s in ([stage] if stage else ["dpo", "sft", "pretrain"]):
        d = REPO / "checkpoints" / s
        xs = _list_ckpts(str(d))
        if xs:
            return xs[-1]
    raise SystemExit("no checkpoints found in checkpoints/{dpo,sft,pretrain}")


def load_model(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("cfg") or mythos_3b()
    model = OpenMythos(cfg)
    model.load_state_dict(ckpt["model"])
    bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16_ok else torch.float32
    model = model.to(device=device, dtype=dtype)
    model.eval()
    return model, cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["pretrain", "sft", "dpo"], default=None)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--n_loops", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    path = find_ckpt(args.stage, args.ckpt)
    print(f"[repl] loading {path} on {device}")
    model, cfg = load_model(path, device)
    tok = MythosTokenizer()

    n_loops = args.n_loops
    temperature = args.temperature
    top_k = args.top_k
    max_new = args.max_new_tokens

    print(f"[repl] ready. cfg.vocab={cfg.vocab_size} n_loops={n_loops} "
          f"temp={temperature} top_k={top_k} max_new={max_new}")
    print("       :n N | :t F | :k N | :m N | :reset | :quit")

    while True:
        try:
            line = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line == ":quit" or line == ":q":
            break
        if line == ":reset":
            print("[repl] session reset.")
            continue
        if line.startswith(":n "):
            n_loops = int(line.split()[1])
            print(f"[repl] n_loops={n_loops}")
            continue
        if line.startswith(":t "):
            temperature = float(line.split()[1])
            print(f"[repl] temperature={temperature}")
            continue
        if line.startswith(":k "):
            top_k = int(line.split()[1])
            print(f"[repl] top_k={top_k}")
            continue
        if line.startswith(":m "):
            max_new = int(line.split()[1])
            print(f"[repl] max_new_tokens={max_new}")
            continue

        prompt = PROMPT_TEMPLATE.format(instruction=line)
        ids = tok.encode(prompt)
        x = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad():
            out = model.generate(
                x,
                max_new_tokens=max_new,
                n_loops=n_loops,
                temperature=max(temperature, 1e-5),
                top_k=top_k,
            )
        gen_ids = out[0, len(ids):].tolist()
        text = tok.decode(gen_ids)
        if EOS_TEXT in text:
            text = text.split(EOS_TEXT)[0]
        print(text.strip())


if __name__ == "__main__":
    main()
