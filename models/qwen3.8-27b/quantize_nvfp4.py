"""
quantize_nvfp4.py -- NVFP4-mixed quantization of a Qwen3.8-27B checkpoint, with
the MTP head carried through untouched.

    "$TEMPLATES_DIR/venv-quant/bin/python" quantize_nvfp4.py <source_dir> <output_dir>

Runs in venv-quant, NOT the training venv. llm-compressor pins compressed-tensors
0.19.0 while vllm 0.28.0 in the training venv was validated against 0.17.0, and it
also bumps accelerate and datasets. Those are the stack rc1 trains on; this venv
carries the same torch and transformers pins and differs only in the quantizer.

THE RECIPE IS THE REFERENCE'S, LOADED, NOT RESTATED

`nvfp4_reference_recipe.json` is `unsloth/Qwen3.8-27B-NVFP4`'s quantization_config
verbatim, read from its config.json on 2026-10-01. It is not uniformly FP4 -- it
is `format: mixed-precision`, and the mix is specific:

    FP4  (NVFP4, group 16)   mlp.{gate,up,down}_proj on layers 0-55
    FP8  (channel / token)   self_attn.{q,k,v,o}_proj, Gated DeltaNet
                             {in_proj_qkv,in_proj_z,out_proj}, lm_head,
                             and mlp on layers 56-63
    FP8  static              KV cache
    untouched (303 ignores)  192 Gated DeltaNet non-projection tensors,
                             all 110 vision-tower linears, and  re:^mtp.*

Two carve-outs in that list would be easy to get wrong from memory: the LAST
EIGHT MLP layers are deliberately FP8 rather than FP4, and the vision tower is
not quantized at all. The whole thing is loaded from the file so this script
cannot drift from it one entry at a time.

THE MTP HEAD NEVER ENTERS THE QUANTIZER

`re:^mtp.*` in the ignore list is load-bearing: the group targets
`self_attn.(q|k|v|o)_proj` and `mlp.(gate|up|down)_proj` WOULD match
`mtp.layers.0.*`, and ignore is what stops them. But this script does not rely
on that alone. `Qwen3_5ForConditionalGeneration` drops `^mtp.*` on load, so the
quantizer never sees the head at all; the 15 tensors are then copied from the
source checkpoint, verbatim, into a separate `model_mtp.safetensors` -- which is
the reference's own layout, and the one vLLM's Qwen3_5MTP loader expects. In the
reference that file is 849,400,392 bytes: the head is BF16 throughout, not a mix.
Bit-identity against the source is asserted, not assumed.

CALIBRATION

On judge rows, with their images, through the same build_feature/JudgeCollator
the training path uses -- activation scales should come from what the model will
actually see. Only the NVFP4 activation global scales and the static KV-cache
scales depend on calibration; the FP8 group uses dynamic per-token activations.

WHAT IS VERIFIED ON THE WRITTEN OUTPUT

By comparison against the source, never by trusting the run: every mtp.* tensor
bit-identical; every vision-tower tensor bit-identical; every FP4 target has
weight_packed + weight_scale + weight_global_scale; every FP8 target has
weight_scale; nothing outside the recipe's targets changed. A quantizer that
drops a component is the same failure the merge had, one artifact later.
"""

import json
import os
import sys
import time

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

HERE = os.path.dirname(os.path.abspath(__file__))
RECIPE_PATH = os.path.join(HERE, "nvfp4_reference_recipe.json")


def load_recipe():
    with open(RECIPE_PATH) as handle:
        return json.load(handle)


# --- calibration -------------------------------------------------------------

def calibration_features(processor, dataset_dir, n_rows):
    """
    Judge rows as model-ready multimodal features, via the training path's code.

    Uses the training carve (the drift guard excluded) so calibration sees the
    same distribution the adapter was trained on, and reuses build_feature so a
    change to how rows become tensors cannot leave calibration behind.
    """
    from tasks.judge import dataset as judge_dataset
    from tasks.judge.collator import build_feature

    rows = judge_dataset.load_rows(dataset_dir)
    train_rows, _ = judge_dataset.train_eval_split(rows, dataset_dir)
    train_rows = train_rows[:n_rows]
    feats = []
    for i, row in enumerate(train_rows, 1):
        feats.append(build_feature(processor, row, dataset_dir))
        if i % 32 == 0 or i == len(train_rows):
            print(f"    calibration features {i}/{len(train_rows)}")
    return feats


# --- the MTP head, by copy ----------------------------------------------------

def write_mtp_sidecar(source_dir, output_dir):
    """
    Copy the source's mtp.* tensors into model_mtp.safetensors and index them.

    The quantizer never saw these: the model class drops ^mtp.* on load. So the
    only way they reach the output is this copy, and the only way to know the
    copy is right is to read it back and compare -- done in verify().
    """
    from safetensors import safe_open
    from safetensors.torch import save_file

    with open(os.path.join(source_dir, "model.safetensors.index.json")) as handle:
        src_map = json.load(handle)["weight_map"]
    mtp_names = sorted(n for n in src_map if n.startswith("mtp."))
    if not mtp_names:
        raise ValueError(
            f"{source_dir} carries no mtp.* tensors -- it is probably a merge "
            "written by a class that drops the head. Quantizing it would ship a "
            "checkpoint with no drafter, which is the rc0 mistake again."
        )
    tensors = {}
    by_shard = {}
    for n in mtp_names:
        by_shard.setdefault(src_map[n], []).append(n)
    for shard, names in by_shard.items():
        with safe_open(os.path.join(source_dir, shard), framework="pt") as f:
            for n in names:
                tensors[n] = f.get_tensor(n)

    # llm-compressor 0.14.0 turns out to write this same file itself on save
    # (its log says "save_mtp_tensors_to_checkpoint | Copied MTP weights"), so
    # this overwrites it. Kept on purpose, with identical bytes: the copy here
    # is explicit, verified by comparison below, and does not depend on a
    # library version continuing to do something undocumented. Verified on the
    # first run that the two agree -- zero mtp.* tensors in the main shards,
    # zero tensor names present in more than one file.
    sidecar = "model_mtp.safetensors"
    save_file(tensors, os.path.join(output_dir, sidecar), metadata={"format": "pt"})

    index_path = os.path.join(output_dir, "model.safetensors.index.json")
    with open(index_path) as handle:
        index = json.load(handle)
    for n in mtp_names:
        index["weight_map"][n] = sidecar
    index["weight_map"] = dict(sorted(index["weight_map"].items()))
    index.setdefault("metadata", {})["total_size"] = sum(
        os.path.getsize(os.path.join(output_dir, f))
        for f in set(index["weight_map"].values())
    )
    with open(index_path, "w") as handle:
        json.dump(index, handle, indent=1)
    return mtp_names, os.path.getsize(os.path.join(output_dir, sidecar))


# --- verification -------------------------------------------------------------

def _load(root, names):
    from safetensors import safe_open

    with open(os.path.join(root, "model.safetensors.index.json")) as handle:
        wmap = json.load(handle)["weight_map"]
    out, by_shard = {}, {}
    for n in names:
        by_shard.setdefault(wmap[n], []).append(n)
    for shard, ns in by_shard.items():
        with safe_open(os.path.join(root, shard), framework="pt") as f:
            for n in ns:
                out[n] = f.get_tensor(n)
    return out


def verify(source_dir, output_dir, recipe):
    """Compare the written checkpoint against the source and the recipe. Raises."""
    import re

    import torch

    with open(os.path.join(output_dir, "model.safetensors.index.json")) as handle:
        out_map = json.load(handle)["weight_map"]
    with open(os.path.join(source_dir, "model.safetensors.index.json")) as handle:
        src_map = json.load(handle)["weight_map"]
    with open(os.path.join(output_dir, "config.json")) as handle:
        cfg = json.load(handle)

    problems = []
    report = {}

    # 1. The config says what the recipe says.
    qc = cfg.get("quantization_config") or {}
    if qc.get("format") != recipe["format"]:
        problems.append(f"config format {qc.get('format')!r} != recipe {recipe['format']!r}")
    if "re:^mtp.*" not in (qc.get("ignore") or []):
        problems.append("config.json quantization_config.ignore lacks re:^mtp.* -- "
                        "vLLM would try to dequantize a BF16 head")
    if "vision_config" not in cfg:
        problems.append("vision_config missing from output config.json")

    # 2. mtp.* bit-identical to source, and in the sidecar.
    mtp = sorted(n for n in src_map if n.startswith("mtp."))
    src_mtp, out_mtp = _load(source_dir, mtp), _load(output_dir, mtp)
    diff = [n for n in mtp if not torch.equal(src_mtp[n], out_mtp[n])]
    if diff:
        problems.append(f"{len(diff)} mtp.* tensors differ from source: {diff[:3]}")
    not_sidecar = [n for n in mtp if out_map.get(n) != "model_mtp.safetensors"]
    if not_sidecar:
        problems.append(f"{len(not_sidecar)} mtp.* tensors not in model_mtp.safetensors")
    report["mtp_identical"] = f"{len(mtp) - len(diff)}/{len(mtp)}"

    # 3. Vision tower bit-identical (it is wholly ignored by the recipe).
    vis = sorted(n for n in src_map if ".visual." in n)
    src_v, out_v = _load(source_dir, vis), _load(output_dir, vis)
    vdiff = [n for n in vis if n not in out_v or not torch.equal(src_v[n], out_v[n])]
    if vdiff:
        problems.append(f"{len(vdiff)} vision tensors changed or missing: {vdiff[:3]}")
    report["vision_identical"] = f"{len(vis) - len(vdiff)}/{len(vis)}"

    # 4. Targets carry the artifacts their scheme produces.
    fp4_re = [re.compile(t[3:]) for g in recipe["config_groups"].values()
              if g["weights"]["num_bits"] == 4 for t in g["targets"]]
    fp8_re = [re.compile(t[3:]) for g in recipe["config_groups"].values()
              if g["weights"]["num_bits"] == 8 for t in g["targets"]]
    ignore_names = {n for n in recipe["ignore"] if not n.startswith("re:")}
    ignore_re = [re.compile(n[3:]) for n in recipe["ignore"] if n.startswith("re:")]

    def ignored(mod):
        return mod in ignore_names or any(r.match(mod) for r in ignore_re)

    # FP8 is tested BEFORE FP4 on purpose. mlp.{gate,up,down}_proj on layers
    # 56-63 match both groups' regexes, and the reference resolves them to FP8:
    # its group_0 names those eight layers explicitly, which is strictly more
    # specific than group_1's generic mlp pattern. The written artifacts bear
    # that out -- layer 60's gate_proj carries weight + weight_scale, layer 0's
    # carries weight_packed + weight_scale + weight_global_scale -- and the
    # first version of this check, testing FP4 first, flagged all 24 of them as
    # defects in a correct checkpoint.
    modules = {n[: -len(".weight")] for n in src_map if n.endswith(".weight")}
    n_fp4 = n_fp8 = 0
    for mod in sorted(modules):
        if ignored(mod):
            continue
        if any(r.search(mod) for r in fp8_re):
            n_fp8 += 1
            if f"{mod}.weight_scale" not in out_map:
                problems.append(f"FP8 target {mod} lacks weight_scale")
            if f"{mod}.weight_packed" in out_map:
                problems.append(f"FP8 target {mod} was packed as FP4")
        elif any(r.search(mod) for r in fp4_re):
            n_fp4 += 1
            for suffix in ("weight_packed", "weight_scale", "weight_global_scale"):
                if f"{mod}.{suffix}" not in out_map:
                    problems.append(f"FP4 target {mod} lacks {suffix}")
    report["fp4_modules"] = n_fp4
    report["fp8_modules"] = n_fp8

    # 5. Everything the recipe does not touch is unchanged.
    passthrough = sorted(
        n for n in src_map
        if not n.startswith("mtp.") and ".visual." not in n
        and not any(r.search(n[: -len(".weight")] if n.endswith(".weight") else n)
                    for r in fp4_re + fp8_re)
    )
    src_p, out_p = _load(source_dir, passthrough), _load(output_dir, passthrough)
    pdiff = [n for n in passthrough if n not in out_p or not torch.equal(src_p[n], out_p[n])]
    if pdiff:
        problems.append(f"{len(pdiff)} untargeted tensors changed: {pdiff[:3]}")
    report["untargeted_identical"] = f"{len(passthrough) - len(pdiff)}/{len(passthrough)}"

    if problems:
        raise AssertionError("quantized checkpoint failed verification:\n  "
                             + "\n  ".join(problems))
    return report


# --- the quantizer -----------------------------------------------------------

def quantize(source_dir, output_dir, dataset_dir, n_calib, max_seq_len):
    import torch
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier
    from tasks.judge.collator import JudgeCollator

    recipe = load_recipe()

    print(f"Loading {source_dir} ...")
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        source_dir, dtype=torch.bfloat16, device_map="cuda"
    )
    processor = AutoProcessor.from_pretrained(source_dir)
    n_mtp_in_model = sum(1 for n, _ in model.named_parameters() if n.startswith("mtp."))
    assert n_mtp_in_model == 0, "model class instantiated an MTP head; the copy path assumes it did not"

    print(f"Building {n_calib} calibration rows from {dataset_dir} ...")
    feats = calibration_features(processor, dataset_dir, n_calib)
    collator = JudgeCollator(pad_token_id=processor.tokenizer.pad_token_id,
                             max_seq_len=max_seq_len)

    # The reference's config_groups and kv_cache_scheme are serialized
    # QuantizationScheme / QuantizationArgs; pydantic coerces them back, with
    # each group's format intact (nvfp4-pack-quantized, float-quantized). That is
    # what makes the output mixed-precision like the reference rather than one
    # scheme stamped over everything.
    modifier = QuantizationModifier(
        config_groups=recipe["config_groups"],
        ignore=recipe["ignore"],
        kv_cache_scheme=recipe["kv_cache_scheme"],
    )

    # A DataLoader is accepted as-is (datasets/utils.py: get_calibration_dataloader
    # returns it directly), which is the clean way to feed multimodal rows:
    # pixel_values are ~113 MB of float32 per row and do not belong in an arrow
    # table. The collator is the training path's, so calibration batches are
    # built exactly as training batches were.
    from torch.utils.data import DataLoader

    loader = DataLoader(feats, batch_size=1, shuffle=False, collate_fn=collator)

    t0 = time.perf_counter()
    oneshot(
        model=model,
        processor=processor,
        recipe=modifier,
        dataset=loader,
        # "basic" runs the whole model per batch instead of tracing it layer by
        # layer. Tracing a multimodal model whose vision tower is wholly ignored
        # is exactly where the sequential pipeline surprises people, memory is
        # not the constraint on a 268 GiB card with a 51 GiB model, and basic is
        # the documented fallback the sequential pipeline would take anyway.
        pipeline="basic",
    )
    t_quant = time.perf_counter() - t0

    n_restored = restore_untargeted_in_memory(model, source_dir, recipe)
    print(f"  restored {n_restored} untargeted tensors from the source, verbatim")

    # Saved by us rather than by oneshot(output_dir=...), so the restore above
    # lands in the written file. llm-compressor has already patched
    # save_pretrained to accept save_compressed.
    print(f"Saving to {output_dir} ...")
    model.save_pretrained(output_dir, save_compressed=True)
    processor.save_pretrained(output_dir)
    return recipe, t_quant


def restore_untargeted_in_memory(model, source_dir, recipe):
    """
    Put every tensor the recipe does not quantize back to its source bytes.

    Needed because of a precision artifact, measured on the first run.
    Qwen3_5RMSNorm is an OFFSET norm -- forward computes x * (1 + w) -- and
    llm-compressor converts those to standard (1 + w) form for calibration and
    restores w' - 1 afterwards, casting to bf16 at both ends. Near 1.0 bf16's
    spacing is 2^-7, so the round trip re-rounds every norm weight: all 161
    norms changed (64 input_layernorm, 64 post_attention_layernorm, 16 q_norm,
    16 k_norm, the final norm), max |delta| exactly 0.0078125 = one ulp, on
    weights of magnitude ~0.03. The originals are the trained values; the
    rounding buys nothing. The calibrated scales were computed against norms
    within one ulp of these, so restoring does not invalidate them.

    Done in memory before save, so no shard is rewritten and the verifier can
    demand bit-identity for everything untargeted -- which is the guarantee
    that actually matters: anything the recipe does not quantize is the source.
    """
    import re

    import torch

    target_re = [re.compile(t[3:]) for g in recipe["config_groups"].values()
                 for t in g["targets"]]
    with open(os.path.join(source_dir, "model.safetensors.index.json")) as handle:
        src_map = json.load(handle)["weight_map"]
    params = dict(model.named_parameters())
    wanted = [
        n for n in src_map
        if n in params and not n.startswith("mtp.")
        and not any(r.search(n[: -len(".weight")] if n.endswith(".weight") else n)
                    for r in target_re)
    ]
    src = _load(source_dir, wanted)
    with torch.no_grad():
        for n in wanted:
            p = params[n]
            assert p.shape == src[n].shape, (n, p.shape, src[n].shape)
            p.copy_(src[n].to(device=p.device, dtype=p.dtype))
    return len(wanted)


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("source_dir", help="a COMPLETE merged checkpoint (carries mtp.*)")
    parser.add_argument("output_dir")
    templates = (os.environ.get("TEMPLATES_DIR") or os.environ.get("DEMO_DIR")
                 or "/mnt/data/slurm-llm-templates")
    parser.add_argument("--dataset-dir",
                        default=f"{templates}/datasets/judge-sft-rc0-54df8b59")
    parser.add_argument("--calibration-rows", type=int, default=128)
    parser.add_argument("--max-seq-len", type=int, default=10240)
    parser.add_argument("--verify-only", action="store_true",
                        help="skip quantization; verify an existing output_dir "
                        "against source_dir and the recipe")
    args = parser.parse_args()

    if args.verify_only:
        report = verify(args.source_dir, args.output_dir, load_recipe())
        for k, v in report.items():
            print(f"  {k:22} {v}")
        print("\nVERIFIED -- the checkpoint matches the source and the recipe.")
        return 0

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        raise SystemExit(f"{args.output_dir} exists and is not empty; refusing to overwrite")
    os.makedirs(args.output_dir, exist_ok=True)

    t_all = time.perf_counter()
    recipe, t_quant = quantize(args.source_dir, args.output_dir, args.dataset_dir,
                               args.calibration_rows, args.max_seq_len)

    print("Writing the MTP head alongside, untouched ...")
    mtp_names, sidecar_bytes = write_mtp_sidecar(args.source_dir, args.output_dir)
    print(f"  {len(mtp_names)} tensors -> model_mtp.safetensors ({sidecar_bytes:,} bytes)")

    print("Verifying against the source ...")
    report = verify(args.source_dir, args.output_dir, recipe)
    for k, v in report.items():
        print(f"  {k:22} {v}")

    size = sum(os.path.getsize(os.path.join(args.output_dir, f))
               for f in os.listdir(args.output_dir)
               if os.path.isfile(os.path.join(args.output_dir, f)))
    record = {
        "source": args.source_dir, "recipe": RECIPE_PATH,
        "calibration": {"dataset_dir": args.dataset_dir, "rows": args.calibration_rows,
                        "max_seq_len": args.max_seq_len},
        "quantize_seconds": round(t_quant, 1),
        "total_seconds": round(time.perf_counter() - t_all, 1),
        "output_bytes": size, "mtp_sidecar_bytes": sidecar_bytes,
        "verification": report,
    }
    with open(os.path.join(args.output_dir, "quantize_record.json"), "w") as handle:
        json.dump(record, handle, indent=1)
    print(f"\nDone: {size / 2**30:.2f} GiB in {record['total_seconds']:.0f}s "
          f"(quantize {t_quant:.0f}s). Record -> quantize_record.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
