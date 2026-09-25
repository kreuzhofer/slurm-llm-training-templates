"""
mtp_drift_probe.py -- how stale is the transplanted MTP drafter?

rc0's merged checkpoint carries the BASE multi-token-prediction head, copied
verbatim, because the module is never instantiated during training (transformers
has no MTP implementation for this architecture at all -- the only mentions of
`mtp` in modeling_qwen3_5.py are two ignore-regexes). So the drafter learned
"given BASE trunk hidden state h, the token two ahead is X" and is now fed
hidden states from a LoRA-merged trunk. The mapping is stale.

The question is by how much, and whether it is worth building an MTP-aware
training path to fix at the root.

WHAT THIS MEASURES, AND WHAT IT DOES NOT

The drafter's input is the trunk's FINAL hidden state -- the same tensor that
feeds lm_head. vLLM's Qwen3_5MultiTokenPredictor.forward takes it as
`hidden_states`, norms it, concatenates the token embedding and projects
through `mtp.fc`. So the drift the drafter actually experiences is the drift in
that tensor, and that is what this compares between base and merged.

This is a PROXY for acceptance rate, not a substitute. Acceptance depends on
argmax agreement after the head and lm_head, which a small hidden-state change
can still flip near a decision boundary. What it can do is bound the upside: if
the final hidden states are all but identical, there is nothing for co-training
or head-only retraining to recover, and the question is closed cheaply. If they
have moved materially, the serving-side acceptance measurement is worth the
window.

The tensor is captured with a forward hook on the trunk's final norm rather
than from output_hidden_states, because whether that tuple's last entry is pre-
or post-norm is an implementation detail that has changed between versions.
The hook is unambiguous: it is the module whose output lm_head consumes.
"""

import argparse
import gc
import os
import sys

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

import torch

from tasks.judge import dataset as judge_dataset
from tasks.judge.collator import JudgeCollator, build_feature
from tasks.judge.masking import IGNORE_INDEX


def find_final_norm(model):
    """
    The RMSNorm whose output feeds lm_head, located by structure not by name.

    Verified by asserting lm_head(hook_output) == logits, so a layout change
    fails here instead of silently measuring the wrong tensor.
    """
    text = model.model.language_model
    assert hasattr(text, "norm"), f"no final norm on {type(text).__name__}"
    return text.norm


@torch.no_grad()
def final_hidden(model_path, batches, device="cuda"):
    """Run each batch and return the trunk's final hidden state, on CPU."""
    from transformers import Qwen3_5ForConditionalGeneration

    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_path, dtype=torch.bfloat16, attn_implementation="sdpa", use_kernels=True
    ).to(device)
    model.eval()

    captured = {}
    handle = find_final_norm(model).register_forward_hook(
        lambda _m, _i, out: captured.__setitem__(
            "h", out[0] if isinstance(out, tuple) else out
        )
    )

    out_states = []
    for i, batch in enumerate(batches):
        gpu = {k: v.to(device) for k, v in batch.items()}
        result = model(**{k: v for k, v in gpu.items() if k != "labels"})
        h = captured["h"]

        # Prove the hook caught the tensor lm_head actually consumes.
        if i == 0:
            probe = model.lm_head(h[:, -3:, :])
            assert torch.allclose(probe, result.logits[:, -3:, :], atol=1e-2), (
                "hook output does not reproduce the logits; the captured tensor "
                "is not what lm_head consumes"
            )
        out_states.append(h.float().cpu())
        del gpu, result
        torch.cuda.empty_cache()

    handle.remove()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return out_states


def compare(a, b, mask):
    """Cosine similarity and relative L2 at each real position."""
    a, b = a[mask], b[mask]
    cos = torch.nn.functional.cosine_similarity(a, b, dim=-1)
    rel = (a - b).norm(dim=-1) / a.norm(dim=-1).clamp_min(1e-9)
    return cos, rel


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    demo = os.environ.get("DEMO_DIR", "/mnt/data/qwen38-demo")
    parser.add_argument("--base", default=f"{demo}/models/Qwen3.8-27B")
    parser.add_argument(
        "--merged", default=f"{demo}/output/qwen3.8-27b-judge-rc0-merged-complete"
    )
    parser.add_argument(
        "--dataset-dir", default=f"{demo}/datasets/judge-sft-rc0-54df8b59"
    )
    parser.add_argument("--rows", type=int, default=6)
    args = parser.parse_args()

    rows = judge_dataset.load_rows(args.dataset_dir)
    # Include a 512x512 row so the mix of render sizes is represented.
    small = [r for r in rows if r["id"].startswith("dac780aa")]
    picked = small + [r for r in rows if not r["id"].startswith("dac780aa")][
        : max(0, args.rows - len(small))
    ]
    print(f"{len(picked)} rows from {args.dataset_dir}")

    processor = judge_dataset.load_processor(args.base)
    collator = JudgeCollator(pad_token_id=processor.tokenizer.pad_token_id)
    batches, spans = [], []
    for row in picked:
        feature = build_feature(processor, row, args.dataset_dir)
        batches.append(collator([feature]))
        spans.append(feature["labels"] != IGNORE_INDEX)

    print(f"\nbase   : {args.base}")
    base_h = final_hidden(args.base, batches)
    print(f"merged : {args.merged}")
    merged_h = final_hidden(args.merged, batches)

    print(f"\n{'row':>10} {'positions':>10} {'cos mean':>10} {'cos min':>10} "
          f"{'relL2 mean':>11} {'relL2 max':>10}")
    all_cos, all_rel, ans_cos = [], [], []
    for row, a, b, span, batch in zip(picked, base_h, merged_h, spans, batches):
        real = batch["attention_mask"][0].bool()
        cos, rel = compare(a[0], b[0], real)
        all_cos.append(cos)
        all_rel.append(rel)
        # The assistant span is where decoding happens, so it is where the
        # drafter is actually used.
        pad = torch.zeros(real.shape[0], dtype=torch.bool)
        pad[: span.shape[0]] = span
        ans_cos.append(compare(a[0], b[0], pad)[0])
        print(f"{row['id'][:8]:>10} {int(real.sum()):>10} {cos.mean():>10.6f} "
              f"{cos.min():>10.6f} {rel.mean():>11.6f} {rel.max():>10.6f}")

    cos = torch.cat(all_cos)
    rel = torch.cat(all_rel)
    ans = torch.cat(ans_cos)
    print(f"\nALL POSITIONS  n={cos.numel()}")
    print(f"  cosine   mean {cos.mean():.6f}  min {cos.min():.6f}  "
          f"p1 {cos.quantile(0.01):.6f}")
    print(f"  rel L2   mean {rel.mean():.6f}  max {rel.max():.6f}  "
          f"p99 {rel.quantile(0.99):.6f}")
    for t in (0.9999, 0.999, 0.99, 0.95):
        print(f"  positions with cosine < {t}: {int((cos < t).sum())} "
              f"({100 * (cos < t).float().mean():.2f}%)")
    print(f"\nASSISTANT SPAN ONLY (where decoding happens)  n={ans.numel()}")
    print(f"  cosine   mean {ans.mean():.6f}  min {ans.min():.6f}")


if __name__ == "__main__":
    main()
