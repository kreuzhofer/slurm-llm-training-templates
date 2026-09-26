"""
model.py -- everything about this pipeline that is specific to Qwen3.8-27B.

Split out from the task-level code in `tasks/` so that a second model can sit
beside this one without either inheriting the other's assumptions. What lives
here is architecture-shaped: how the model is loaded, how FSDP wraps it, and
where its weights live. What lives in `tasks/` is task-shaped -- the dataset
split, the prompt, the metric -- and MUST stay shared, because that is what
makes numbers from two models comparable at all.

Three things here are specific to Qwen3.8-27B:

1. ARCHITECTURE. It reports `model_type: "qwen3_5"` /
   `Qwen3_5ForConditionalGeneration`, is natively multimodal, and ships a vision
   tower plus a multi-token-prediction head. `AutoModelForCausalLM` resolves to
   `Qwen3_5ForCausalLM`, whose `_keys_to_ignore_on_load_unexpected` drops both,
   leaving the 26.896B-parameter text stack (measured).

   There are therefore TWO load paths here, not one with a flag:
   `load_model()` is the text-only stack the SQL work measured and must not
   change shape, and `load_vision_model()` builds the full multimodal stack for
   visual training. The MTP head is dropped by both.

2. HYBRID ATTENTION. 16 of 64 layers use classic attention; the other 48 are
   Gated DeltaNet. This drives the FSDP wrap class below, and it drove the LoRA
   target-list question that train_lora.py answers by measurement.

3. FSDP2. transformers v5 defaults `fsdp_config["version"]` to 2, so the
   FSDP1-only knobs are silently ignored and are absent here rather than
   carried over as dead config.
"""

import os

def load_model(model_path):
    """Load the text-only stack, dropping the vision tower and MTP head."""
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,        # `torch_dtype` is deprecated in transformers v5
        attn_implementation="sdpa",  # only affects the 16 full-attention layers
        use_kernels=True,            # fused Gated DeltaNet kernels from the Hub;
                                     # without this, 48/64 layers run the slow
                                     # pure-PyTorch fallback
    )
    # Avoids allocating the ~152MB/sequence GDN recurrent state every step.
    #
    # This assignment is effective HERE and only here: Qwen3_5ForCausalLM is
    # configured by the flat Qwen3_5TextConfig, so `use_cache` is the attribute
    # the decoder reads. The multimodal class has a composite config where the
    # same line does nothing -- see disable_kv_cache(). Do not merge these two
    # paths into one helper on the strength of the lines looking identical.
    model.config.use_cache = False
    return model


def load_vision_model(model_path, train_tower=False, train_merger=True):
    """
    Load the FULL multimodal stack -- text + vision tower -- for visual training.

    Deliberately a SECOND function rather than a flag on load_model(). The SQL
    numbers in docs/RESULTS.md came out of load_model() and have to stay
    reproducible, so the text-only path keeps exactly the shape it had; a
    visual run opts in by calling this instead.

    Measured (#17), same checkpoint, same GPU, one process:

        AutoModelForCausalLM            26.8960 B params, 50.10 GiB
        Qwen3_5ForConditionalGeneration 27.3567 B params, 50.96 GiB

    so the tower costs +0.4607 B / +0.86 GiB, and there is no memory reason to
    avoid it: text-only training peaked at 29 GiB/GPU (LoRA) and 35 GiB (full
    FT) on 268.6 GiB cards.

    The multi-token-prediction head is dropped either way: `^mtp.*` is in
    _keys_to_ignore_on_load_unexpected on the shared base class
    (modeling_qwen3_5.py:807), not just on Qwen3_5ForCausalLM. Only the vision
    tower is class-dependent. Verified: params outside text+visual are 0 in
    both classes.

    train_tower / train_merger set requires_grad on the 27 vision blocks and on
    the patch merger. Defaults freeze the tower and train the merger, which is
    the usual VLM SFT arrangement and is the defensible default here for two
    reasons: the tower already reads generated charts correctly out of the box
    (#17), so a deficit is more likely in how the LM *uses* visual features
    than in the encoder; and the merger is the projection into LM hidden space,
    which is exactly what a new task shifts.

    NOTE this only bites for full-parameter training. Under LoRA the base
    weights are frozen wholesale by get_peft_model() afterwards, so what
    matters there is which modules the adapter TARGETS, not these flags.
    """
    import torch
    from transformers import Qwen3_5ForConditionalGeneration

    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        # Parity with the text path. Note this also sets the VISION tower's
        # attention: under sdpa Qwen3_5VisionAttention splits the packed batch
        # and runs one attention call per image
        # (modeling_qwen3_5.py:952), whereas the flash path does all images in
        # a single varlen kernel. If image throughput ever matters, that is the
        # first thing to change -- but sdpa keeps this comparable to the SQL
        # runs, which is what this measurement is for.
        attn_implementation="sdpa",
        use_kernels=True,
    )
    disable_kv_cache(model)
    set_vision_trainability(model, train_tower=train_tower, train_merger=train_merger)
    return model


def disable_kv_cache(model):
    """
    Turn the KV cache off for training, on a COMPOSITE config.

    `model.config.use_cache = False` -- which is what load_model() does and what
    every Qwen3-era script does -- is a SILENT NO-OP on this class, and it cost
    a 16-GPU job (#20, job 851) to find out.

    `Qwen3_5ForCausalLM` is configured by the flat `Qwen3_5TextConfig`, where
    `use_cache` is the attribute the decoder actually reads, so the assignment
    works and the text-only path has always been fine.
    `Qwen3_5ForConditionalGeneration` is configured by `Qwen3_5Config`, which
    has **no top-level `use_cache` at all** (verified: `getattr` returns
    absent) and holds the real one at `config.text_config.use_cache`, default
    True. Assigning to the top level therefore invents a brand-new attribute
    that nothing reads, and the cache stays ON.

    The failure is not a clean error either. Training survives the forward pass
    and dies in backward, because activation checkpointing recomputes the
    layer against a cache that the first pass already filled:

        RuntimeError: The expanded size of the tensor (976) must match the
        existing size (488) at non-singleton dimension 3.
        Target sizes: [8, 24, 488, 976]. Tensor sizes: [8, 1, 488, 488]

    976 is exactly 2 x 488: the mask covers the real sequence, the keys cover
    it twice over. Anything that reads "the key length is double the query
    length" on this architecture should suspect this first.

    Sets every level that exists and then ASSERTS the effective one, so a
    future config reshuffle fails loudly here instead of 20 minutes into a
    multi-node run.
    """
    for cfg in (model.config, getattr(model.config, "text_config", None)):
        if cfg is not None and hasattr(cfg, "use_cache"):
            cfg.use_cache = False
    # Belt and braces: create it at the top level too, harmless if unread.
    model.config.use_cache = False
    effective = getattr(model.config, "text_config", model.config)
    assert getattr(effective, "use_cache") is False, (
        "KV cache still enabled after disable_kv_cache(); the config layout "
        "changed and training will fail in backward, not forward"
    )
    return model


def set_vision_trainability(model, train_tower=False, train_merger=True):
    """
    Freeze or unfreeze the vision tower and the patch merger.

    Split out from load_vision_model() so a caller that builds the model some
    other way (or one inspecting the split before training) can reuse it.
    Returns (tower_params, merger_params) actually touched, so a caller can
    assert it matched something rather than silently matching nothing -- the
    module paths below are the failure mode if transformers renames anything.
    """
    tower = merger = 0
    for name, p in model.named_parameters():
        if ".visual." not in name and not name.startswith("visual."):
            continue
        # The merger is the projector into LM hidden space; the blocks are the
        # encoder proper.
        if ".merger." in name:
            p.requires_grad = train_merger
            merger += p.numel()
        else:
            p.requires_grad = train_tower
            tower += p.numel()
    return tower, merger


def fsdp_config(state_dict_type="FULL_STATE_DICT", wrap_vision=False):
    """
    FSDP2 settings.

    transformers v5 defaults fsdp_config["version"] to 2, and the FSDP1-only
    knobs from the Qwen3-era scripts (backward_prefetch, forward_prefetch,
    use_orig_params) are silently ignored under it -- so they are gone here
    rather than carried over as dead config.

    wrap_vision adds Qwen3_5VisionBlock to the wrap set, for a model built by
    load_vision_model(). It is OFF by default and must stay off for the
    text-only path, where no such module exists.

    OFF is also the measured default, not just the safe one. #20 ran both, 16
    GPUs, identical batches, LoRA, tower frozen:

        wrap_vision=False   1.13 s/step   30.10 GiB peak allocated
        wrap_vision=True    1.24 s/step   29.31 GiB peak allocated

    so wrapping buys 0.79 GiB and costs 9.7% throughput -- a bad trade on a
    268.6 GiB card, and for the expected reason: the tower's 27 blocks are
    small (hidden_size 1152 against the text stack's 5120), so there is little
    to shard, while you still pay 27 more all-gather/reduce-scatter pairs every
    step. Losses matched to three decimals, confirming the wrap changes
    mechanics and not maths.

    Keep the flag: if the tower is ever unfrozen (train_tower=True), its
    gradients and optimizer state become worth sharding and this should be
    re-measured rather than assumed to still lose.
    """
    wrap = ["Qwen3_5DecoderLayer"]
    if wrap_vision:
        wrap.append("Qwen3_5VisionBlock")
    return {
        "version": 2,
        "auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
        # Set explicitly. Auto-detection reads _no_split_modules off the model
        # class, which lists Qwen3_5VisionBlock on BOTH classes (verified in
        # #17) -- including Qwen3_5ForCausalLM, which has no such module.
        "transformer_layer_cls_to_wrap": wrap,
        "reshard_after_forward": True,   # equivalent to full_shard
        # Prefer this over TrainingArguments(gradient_checkpointing=True): the
        # latter adds a redundant AllGather in the backward pass under FSDP
        # (transformers#30404).
        "activation_checkpointing": True,
        "cpu_ram_efficient_loading": True,
        "state_dict_type": state_dict_type,
    }


def env_config():
    """Read the knobs the .sbatch files set, with defaults."""
    demo_dir = os.environ.get("DEMO_DIR", "/mnt/data/qwen38-demo")
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    return {
        "demo_dir": demo_dir,
        "model_path": os.environ.get("MODEL_PATH", f"{demo_dir}/models/Qwen3.8-27B"),
        "dataset_path": os.environ.get(
            "DATASET_PATH", f"{demo_dir}/datasets/sql-create-context"
        ),
        "per_device_bs": int(os.environ.get("PER_DEVICE_BATCH_SIZE", "8")),
        "grad_accum": int(os.environ.get("GRADIENT_ACCUMULATION_STEPS", "1")),
        "num_epochs": int(os.environ.get("NUM_EPOCHS", "1")),
        "max_seq_len": int(os.environ.get("MAX_SEQ_LEN", "1024")),
        # Short-run knobs. MAX_STEPS=-1 means "run NUM_EPOCHS epochs", which is
        # how transformers itself spells "no step cap" -- so the default here is
        # exactly the previous behaviour: one full epoch, 584 optimizer steps at
        # effective batch 128 over the 74,648-example train split.
        #
        # A real smoke run is MAX_STEPS=25 SAVE_STEPS=10, which exercises the
        # checkpoint-save path twice in a couple of minutes instead of once at
        # the very end of a full run.
        "max_steps": int(os.environ.get("MAX_STEPS", "-1")),
        "save_steps": int(os.environ.get("SAVE_STEPS", "500")),
        # Defaults to SAVE_STEPS so a short run evaluates as often as it saves;
        # override independently when that is too expensive.
        "eval_steps": int(
            os.environ.get("EVAL_STEPS", os.environ.get("SAVE_STEPS", "500"))
        ),
        # -1 = the whole 3,929-example eval split. A 25-step smoke run that
        # evaluates over all of it spends far longer evaluating than training,
        # so cap it there.
        "max_eval_examples": int(os.environ.get("MAX_EVAL_EXAMPLES", "-1")),
        "rank": rank,
        "is_main": rank == 0,
    }
