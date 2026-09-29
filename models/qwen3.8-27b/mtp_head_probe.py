"""
mtp_head_probe.py -- prove the trainable MTP head is correct before it trains.

One GPU. Loads the base checkpoint's own MTP weights into the ported module and
asks four questions in order, each of which can fail:

  1. Do all 15 `mtp.*` tensors land, with nothing missing or unexpected?
  2. Is the +2 objective ALIGNED? A pretrained head on its own base trunk must
     score far below chance, and must beat the +1 and +3 shifts clearly. This is
     the check that matters: an off-by-one does not crash, it trains to a
     plausible loss on the wrong target and produces a confidently wrong drafter.
  3. Does a gradient reach the head, with the trunk frozen?
  4. Does it overfit one batch?

Run:
    srun --nodes=1 --gpus-per-node=1 --partition=main --time=30 \\
        "$TEMPLATES_DIR/venv/bin/python" \\
        "$TEMPLATES_DIR/repo/models/qwen3.8-27b/mtp_head_probe.py"
"""

import os
import sys
import time

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

import torch

from model import load_vision_model
from mtp import IGNORE_INDEX, MTPTrainer, Qwen3_5MTPHead, load_base_mtp_weights, verify_alignment
from tasks.judge import dataset as judge_dataset
from tasks.judge.collator import JudgeCollator, build_feature


def main():
    templates = (
        os.environ.get("TEMPLATES_DIR")
        or os.environ.get("DEMO_DIR")
        or "/mnt/data/slurm-llm-templates"
    )
    base = os.environ.get("MODEL_PATH", f"{templates}/models/Qwen3.8-27B")
    dataset_dir = os.environ.get(
        "JUDGE_DATASET_DIR", f"{templates}/datasets/judge-sft-rc0-54df8b59"
    )
    steps = int(os.environ.get("PROBE_STEPS", "10"))
    rows_n = int(os.environ.get("PROBE_ROWS", "2"))

    print(f"Base    : {base}")
    print(f"Dataset : {dataset_dir}")

    rows = judge_dataset.load_rows(dataset_dir)[:rows_n]
    processor = judge_dataset.load_processor(base)
    collator = JudgeCollator(pad_token_id=processor.tokenizer.pad_token_id,
                             max_seq_len=10240)
    batch = collator([build_feature(processor, r, dataset_dir) for r in rows])
    print(f"Batch   : {tuple(batch['input_ids'].shape)}, "
          f"{int((batch['labels'] != IGNORE_INDEX).sum())} supervised positions")

    model = load_vision_model(base)
    text_config = model.config.text_config
    head = Qwen3_5MTPHead(text_config).to(dtype=torch.bfloat16)

    print("\n== 1. weights ==")
    loaded = load_base_mtp_weights(head, base, strict=True)
    n_params = sum(p.numel() for p in head.parameters())
    print(f"  {len(loaded)} tensors loaded, none missing or unexpected")
    print(f"  head is {n_params:,} params ({100 * n_params / 27_436_420_336:.2f}% of the model)")
    print(f"  block type: {head.layers[0].block_type}")

    model.cuda()
    head.cuda()
    trainer = MTPTrainer(model, head, train_trunk=False)
    trainer.eval()
    gpu = {k: v.cuda() for k, v in batch.items()}

    print("\n== 2. alignment ==")
    res = verify_alignment(trainer, gpu, text_config.vocab_size)
    chance = res.pop("chance")
    for offset in sorted(res):
        tag = "  <-- the +2 objective" if offset == 2 else ""
        print(f"  shift +{offset}: loss {res[offset]:7.4f}{tag}")
    print(f"  chance (ln vocab): {chance:.4f}")
    best = min(res, key=res.get)
    ok_align = best == 2 and res[2] < chance / 2
    print(f"  best shift is +{best}; +2 is {'well below' if res[2] < chance / 2 else 'NOT below'} half of chance")
    if not ok_align:
        print("\n  ALIGNMENT FAILED. A pretrained head on its own trunk cannot score")
        print("  near chance unless it is being asked for the wrong token. Do not")
        print("  train on this objective.")
        return 1
    print("  ALIGNED -- the pretrained head already solves this objective.")

    print("\n== 3. gradient reaches the head, trunk frozen ==")
    trainer.train()
    trainable = [p for p in trainer.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    print(f"  {n_train:,} trainable params")
    assert n_train == n_params, "trunk parameters are trainable; they should be frozen"
    out = trainer(**gpu)
    out["loss"].backward()
    got_grad = [n for n, p in trainer.head.named_parameters()
                if p.grad is not None and p.grad.abs().sum() > 0]
    print(f"  {len(got_grad)} of {len(list(trainer.head.named_parameters()))} "
          f"head tensors received a non-zero gradient")
    assert got_grad, "no gradient reached the head"
    trainer.zero_grad(set_to_none=True)

    print(f"\n== 4. overfit one batch, {steps} steps ==")
    optimizer = torch.optim.AdamW(trainable, lr=1e-4)
    torch.cuda.reset_peak_memory_stats()
    losses, times = [], []
    for step in range(steps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        loss = trainer(**gpu)["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        losses.append(loss.detach().item())
        print(f"  step {step + 1:>2}  loss {losses[-1]:8.4f}  {times[-1]:6.2f} s")

    steady = times[1:] or times
    print(f"\n  loss {losses[0]:.4f} -> {losses[-1]:.4f}")
    print(f"  step {sum(steady) / len(steady):.2f} s mean over steps 2-{steps} "
          f"(first {times[0]:.2f} s)")
    print(f"  peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB allocated, "
          f"{torch.cuda.max_memory_reserved() / 2**30:.2f} GiB reserved")

    trainer.close()
    if losses[-1] >= losses[0]:
        print("\nFAIL: loss did not fall on a repeated batch.")
        return 1
    print("\nPASS: head loads, objective is aligned, gradients flow, and it learns.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
