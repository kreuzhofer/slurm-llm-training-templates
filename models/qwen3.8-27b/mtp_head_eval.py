"""
mtp_head_eval.py -- score MTP heads on the SAME rows against the SAME trunk.

    python mtp_head_eval.py <trunk_dir> <dataset_dir> --head <name>=<path> ...

Prints each head's +2 loss over the drift-guard rows (and optionally a training
sample), so "did alignment help" is answered by comparison on identical inputs
rather than by reading a training curve.

WHY THIS EXISTS

rc1's head run reported its train loss falling to 0.164 while the drift-guard
loss rose from 0.601 to 0.618 at the last epoch -- the shape of overfitting. But
the run never scored the BASE head on those 30 rows, so there was no way to
tell whether the trained head was better than the one it replaces, worse, or
the same within noise. A head that overfits the training rows and loses on
held-out ones must not ship, and the only way to know is this comparison.

`base` as a head path means: load the base checkpoint's own mtp.* tensors.
"""

import argparse
import os
import sys

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

import torch

from model import load_vision_model
from mtp import MTPTrainer, Qwen3_5MTPHead, load_base_mtp_weights
from tasks.judge import dataset as judge_dataset
from tasks.judge.collator import JudgeCollator, build_feature


def load_head_from_file(head, path):
    from safetensors import safe_open

    with safe_open(path, framework="pt") as f:
        state = {k[len("mtp."):] if k.startswith("mtp.") else k: f.get_tensor(k)
                 for k in f.keys()}
    missing, unexpected = head.load_state_dict(state, strict=False)
    assert not missing and not unexpected, (missing, unexpected)


@torch.no_grad()
def mean_loss(trainer, feats, collator):
    trainer.eval()
    total = 0.0
    for f in feats:
        b = collator([f])
        total += float(trainer(**{k: v.cuda() for k, v in b.items()})["loss"])
    return total / max(len(feats), 1)


def main():
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("trunk_dir")
    p.add_argument("dataset_dir")
    p.add_argument("--head", action="append", required=True,
                   help="name=path, or name=base for the base checkpoint's head")
    p.add_argument("--base", default=None)
    p.add_argument("--n-eval", type=int, default=judge_dataset.DEFAULT_N_EVAL)
    p.add_argument("--train-sample", type=int, default=30,
                   help="also score this many training rows, to show the gap")
    p.add_argument("--max-seq-len", type=int, default=10240)
    args = p.parse_args()
    templates = (os.environ.get("TEMPLATES_DIR") or os.environ.get("DEMO_DIR")
                 or "/mnt/data/slurm-llm-templates")
    base = args.base or f"{templates}/models/Qwen3.8-27B"

    rows = judge_dataset.load_rows(args.dataset_dir)
    train_rows, eval_rows = judge_dataset.train_eval_split(rows, args.dataset_dir, args.n_eval)
    processor = judge_dataset.load_processor(args.trunk_dir)
    collator = JudgeCollator(pad_token_id=processor.tokenizer.pad_token_id,
                             max_seq_len=args.max_seq_len)
    print(f"trunk   : {args.trunk_dir}")
    print(f"rows    : {len(eval_rows)} drift-guard + {args.train_sample} training sample")
    eval_feats = [build_feature(processor, r, args.dataset_dir) for r in eval_rows]
    train_feats = [build_feature(processor, r, args.dataset_dir)
                   for r in train_rows[: args.train_sample]]

    model = load_vision_model(args.trunk_dir).cuda()
    results = {}
    for spec in args.head:
        name, path = spec.split("=", 1)
        head = Qwen3_5MTPHead(model.config.text_config).to(dtype=torch.bfloat16)
        if path == "base":
            load_base_mtp_weights(head, base, strict=True)
        else:
            load_head_from_file(head, path)
        head.cuda()
        trainer = MTPTrainer(model, head, train_trunk=False)
        g = mean_loss(trainer, eval_feats, collator)
        t = mean_loss(trainer, train_feats, collator) if train_feats else float("nan")
        trainer.close()
        results[name] = (g, t)
        print(f"  {name:14} drift-guard +2 loss {g:.4f}   training-sample {t:.4f}")
        del head, trainer
        torch.cuda.empty_cache()

    names = list(results)
    if len(names) >= 2:
        a, b = names[0], names[1]
        d = results[b][0] - results[a][0]
        print(f"\n{b} vs {a} on the drift guard: {d:+.4f} "
              f"({'better' if d < 0 else 'WORSE'} by {abs(d) / results[a][0] * 100:.1f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
