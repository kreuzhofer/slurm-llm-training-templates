"""
merge_judge_lora.py -- merge the judge adapter, KEEPING the vision tower.

Usage:
  python merge_judge_lora.py <adapter_dir> <base_model_dir> <output_dir>

WHY THIS IS NOT merge_lora.py

merge_lora.py loads through `AutoModelForCausalLM`, which resolves to
`Qwen3_5ForCausalLM`, whose `_keys_to_ignore_on_load_unexpected` drops
`^model.visual.*`. The tower is never instantiated, so the merged checkpoint it
writes has `model_type: qwen3_5_text` and NO `vision_config` -- correct for a
text-only SQL adapter, and catastrophic here. A judge that cannot see the eight
renders it is asked to judge is not a judge; it would still load, still serve,
still answer, and be wrong in a way no config diff would announce.

So this loads `Qwen3_5ForConditionalGeneration` and verifies the written
artifact rather than trusting that it worked: the config must carry a
`vision_config` and a multimodal `model_type`, and the shards must actually
contain `visual.*` tensors. Both are checked below, and a failure raises before
anything is uploaded.

The adapter targets only the text stack (q/k/v/o + gate/up/down on the language
model), so the tower passes through untouched -- which is the point: the merged
weights must equal base-tower + adapted-text, not base-tower dropped.
"""

import argparse
import glob
import json
import os
import time


def restore_dropped_tensors(output_path, base_model_path):
    """
    Copy back every tensor the loading class dropped. Returns the names copied.

    `Qwen3_5ForConditionalGeneration` instantiates the vision tower but NOT the
    multi-token-prediction head: `^mtp.*` is in
    `_keys_to_ignore_on_load_unexpected` on the shared base class
    (modeling_qwen3_5.py:807), so it is dropped by BOTH model classes. The merge
    therefore writes 1,184 of the checkpoint's 1,199 tensors and says nothing.

    Found the hard way: rc0's first merged upload was missing all 15, and
    vLLM's recipe runs MTP with 5 draft tokens, so the drafter failed to load
    on the serving cluster (chat3d, 2026-09-24).

    Copying verbatim is correct, and for a precise reason: the module was never
    instantiated during training, so no LoRA adapter could attach to it. Note
    that this is NOT because the target list excludes it -- `q_proj`, `o_proj`,
    `gate_proj`, `up_proj` and `down_proj` all appear inside
    `mtp.layers.0.*`, and a suffix-matched target list WOULD capture them on
    any class that carries the head. If MTP is ever instantiated for training,
    that hazard is live and must be excluded explicitly.

    What this does not do is make the drafter match the tuned model. It is the
    base drafter proposing for a LoRA-merged main model, so draft acceptance
    may fall. Under verification the output is unchanged -- speculative
    decoding is output-equivalent by construction -- so this costs throughput,
    never correctness.
    """
    import json
    import os
    import shutil

    from safetensors import safe_open
    from safetensors.torch import save_file

    base_index = os.path.join(base_model_path, "model.safetensors.index.json")
    out_index = os.path.join(output_path, "model.safetensors.index.json")
    with open(base_index) as handle:
        base_map = json.load(handle)["weight_map"]
    with open(out_index) as handle:
        out_doc = json.load(handle)
    out_map = out_doc["weight_map"]

    missing = sorted(set(base_map) - set(out_map))
    if not missing:
        return []

    print(f"  restoring {len(missing)} tensor(s) the model class dropped")
    by_shard = {}
    for name in missing:
        by_shard.setdefault(base_map[name], []).append(name)

    tensors = {}
    for shard, names in by_shard.items():
        with safe_open(os.path.join(base_model_path, shard), framework="pt") as f:
            for name in names:
                tensors[name] = f.get_tensor(name)

    # Renumber the existing shards so the "of-N" suffix stays honest rather
    # than leaving two files claiming "of-00002" beside a third.
    existing = sorted(set(out_map.values()))
    total = len(existing) + 1
    rename = {
        old: f"model-{i + 1:05d}-of-{total:05d}.safetensors"
        for i, old in enumerate(existing)
    }
    for old, new in rename.items():
        if old != new:
            shutil.move(os.path.join(output_path, old), os.path.join(output_path, new))
    out_map = {name: rename[shard] for name, shard in out_map.items()}

    extra_shard = f"model-{total:05d}-of-{total:05d}.safetensors"
    save_file(tensors, os.path.join(output_path, extra_shard), metadata={"format": "pt"})
    for name in missing:
        out_map[name] = extra_shard

    out_doc["weight_map"] = dict(sorted(out_map.items()))
    out_doc.setdefault("metadata", {})["total_size"] = sum(
        os.path.getsize(os.path.join(output_path, f))
        for f in set(out_map.values())
    )
    with open(out_index, "w") as handle:
        json.dump(out_doc, handle, indent=1)
    return missing


def assert_tensor_parity(output_path, base_model_path):
    """
    The merged checkpoint must hold exactly the base's tensor NAMES.

    By name set, not by count and not by category. The first rc0 merge was
    verified on the vision tower alone -- 333 of 333, correct -- and shipped
    anyway missing 15 MTP tensors, because a check that asks "is the part I
    was worried about present?" cannot find the part nobody was worried about.
    A set difference has no such blind spot.
    """
    import json
    import os

    with open(os.path.join(base_model_path, "model.safetensors.index.json")) as handle:
        base = set(json.load(handle)["weight_map"])
    with open(os.path.join(output_path, "model.safetensors.index.json")) as handle:
        out = set(json.load(handle)["weight_map"])

    missing, extra = sorted(base - out), sorted(out - base)
    assert not missing and not extra, (
        f"tensor parity failed against {base_model_path}:\n"
        f"  {len(missing)} missing: {missing[:8]}\n"
        f"  {len(extra)} unexpected: {extra[:8]}"
    )
    return len(base)


def verify_multimodal(output_path):
    """
    Prove the written checkpoint can still see. Raises if it cannot.

    Reads the files on disk rather than the in-memory model: what gets uploaded
    is the directory, and the failure this guards against is a save-time one.
    """
    with open(os.path.join(output_path, "config.json")) as handle:
        cfg = json.load(handle)

    problems = []
    if "vision_config" not in cfg:
        problems.append(
            f"config.json has no vision_config (model_type={cfg.get('model_type')!r}); "
            "this checkpoint is text-only and cannot read the renders"
        )
    if cfg.get("model_type", "").endswith("_text"):
        problems.append(f"model_type is {cfg['model_type']!r}, a text-only variant")

    # The config could be right while the weights are not.
    index = os.path.join(output_path, "model.safetensors.index.json")
    if os.path.exists(index):
        with open(index) as handle:
            names = json.load(handle)["weight_map"].keys()
    else:
        from safetensors import safe_open

        names = []
        for shard in glob.glob(os.path.join(output_path, "*.safetensors")):
            with safe_open(shard, framework="pt") as f:
                names.extend(f.keys())
    visual = [n for n in names if ".visual." in n or n.startswith("visual.")]
    if not visual:
        problems.append("no visual.* tensors in the saved shards")

    if problems:
        raise AssertionError(
            "merged checkpoint is NOT multimodal:\n  " + "\n  ".join(problems)
        )
    return {
        "model_type": cfg["model_type"],
        "vision_config_keys": sorted(cfg["vision_config"])[:6],
        "visual_tensors": len(visual),
        "total_tensors": len(list(names)),
    }


def main():
    import torch
    from peft import PeftModel
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("adapter_path")
    parser.add_argument("base_model_path")
    parser.add_argument("output_path")
    args = parser.parse_args()

    t0 = time.perf_counter()
    print(f"Loading FULL multimodal base from {args.base_model_path} ...")
    base = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.base_model_path, dtype=torch.bfloat16, device_map="auto"
    )
    n_visual = sum(1 for n, _ in base.named_parameters() if ".visual." in n)
    print(f"  vision tower present: {n_visual} parameter tensors")
    assert n_visual, "base loaded without a vision tower; wrong model class"

    print(f"Loading adapter from {args.adapter_path} ...")
    model = PeftModel.from_pretrained(base, args.adapter_path)

    print("Merging ...")
    model = model.merge_and_unload()
    t_merge = time.perf_counter() - t0

    print(f"Saving to {args.output_path} ...")
    model.save_pretrained(args.output_path, safe_serialization=True)

    # The PROCESSOR, not just the tokenizer: the merged checkpoint has to carry
    # its image processor or nothing downstream can turn a render into patches.
    # train_judge_lora.py saved it beside the adapter.
    AutoProcessor.from_pretrained(args.adapter_path).save_pretrained(args.output_path)

    restored = restore_dropped_tensors(args.output_path, args.base_model_path)
    n_tensors = assert_tensor_parity(args.output_path, args.base_model_path)
    print(f"  tensor parity: {n_tensors} names, exactly matching the base")

    t_total = time.perf_counter() - t0
    facts = verify_multimodal(args.output_path)
    facts["restored_tensors"] = restored
    facts["tensor_parity"] = n_tensors
    size = sum(
        os.path.getsize(os.path.join(args.output_path, f))
        for f in os.listdir(args.output_path)
        if os.path.isfile(os.path.join(args.output_path, f))
    )

    print("\nVerified multimodal:")
    for key, value in facts.items():
        print(f"  {key:20} {value}")
    print(f"\nmerge {t_merge:.1f}s, total {t_total:.1f}s, "
          f"{size / 2**30:.2f} GiB written to {args.output_path}")

    with open(os.path.join(args.output_path, "merge_record.json"), "w") as handle:
        json.dump(
            {"merge_seconds": round(t_merge, 1), "total_seconds": round(t_total, 1),
             "bytes": size, "adapter": args.adapter_path,
             "base": args.base_model_path, **facts},
            handle, indent=1,
        )


if __name__ == "__main__":
    main()
