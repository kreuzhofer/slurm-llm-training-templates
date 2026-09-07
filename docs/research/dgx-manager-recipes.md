# What Daniel's other Qwen3.8 / `qwen3_5` repos know that this one doesn't

Research for issue #3 (map: issue #1). Sources read 2026-09-07.

**How to read the citations.** Sibling repos were fetched as `HEAD`-of-default-branch
tarballs via `gh api repos/kreuzhofer/<repo>/tarball` on 2026-09-07; line numbers are
against that snapshot. Upstream library citations are pinned to the exact versions in
this repo's `requirements.txt` (`transformers==5.16.1`, `vllm==0.28.0`, `peft==0.20.0`),
fetched via `gh api repos/<org>/<repo>/contents/<path>?ref=<tag>`. Model-card facts come
from `huggingface.co/Qwen/Qwen3.8-27B/raw/main/{config.json,chat_template.jinja}`.

Throughout: **CONFIRMED** = a primary source says so. **MEASURED** = someone ran it on
hardware and recorded the number. **ASSERTED** = a source states it with no measurement
behind it. **UNVERIFIED** = nobody has checked.

The single most important structural fact, and the reason this whole exercise transfers:
`dgx-manager/docs/qwen3.8-model-survey.md:83-87` diffed the Qwen3.8-27B and Qwen3.6-27B
`config.json` field-by-field and found them **"structurally identical"** — same 64 layers,
hidden 5120, head_dim 256, `full_attention_interval: 4`, vocab 248320,
`mtp_num_hidden_layers: 1`, `vision_config` present. Its TL;DR calls Qwen3.8 *"a weights
refresh on the Qwen3.5 architecture"* (line 19-20). **So every Qwen 3.6-27B finding in
`dgx-manager` and `dgx-manager-fine-tune-recipes` applies to this repo's model.** That is
what makes the contradictions below load-bearing rather than analogies.

---

# ⚠️ CONTRADICTIONS — READ THESE FIRST

## C1 (LOUD) — every prior run on this architecture deliberately kept LoRA OFF the Gated DeltaNet projections. This repo puts it on.

This repo, `scripts/train_lora.py:36-40`:

```python
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",          # 16 full-attention layers
    "in_proj_qkv", "in_proj_z", "out_proj",          # 48 Gated DeltaNet layers
    "gate_proj", "up_proj", "down_proj",             # all 64 MLPs
]
```

Every sibling recipe for this architecture ships **`["q_proj","k_proj","v_proj","o_proj"]`
only**, and two of them say in so many words not to do what this repo does:

- `dgx-manager/docs/qwen3.6-fine-tuning-on-dgx-spark.md:50` — *"**Mamba layers are not
  LoRA-friendly.** Stateful SSM updates don't compose with LoRA's low-rank additive trick.
  Target attention projections on the full-attn layers only — **do not add LoRA to
  `in_proj` / `out_proj` on the linear-attn layers**. In the recipe:
  `target_modules=["q_proj","k_proj","v_proj","o_proj"]`."*
- `dgx-manager/docs/qwen3.6-27b-fine-tuning-on-dgx-spark.md:52-53` — for the
  structurally identical 27B: *"`target_modules=["q_proj","k_proj","v_proj","o_proj"]` is
  sufficient… **GatedDeltaNet layers are skipped automatically by suffix matching.**
  Static check confirmed `q_proj` only appears on the 16 full-attention layers."*
- Shipped code: `dgx-manager-fine-tune-recipes/recipes/qwen3.6-27b-base-lora/train.py:229`
  and `recipe.yaml:41`; same in `.../qwen3.6-27b-base-lora-longctx/train.py:231`.
- `dgx-manager-fine-tune-recipes/recipes/qwen3.6-35b-a3b-base-lora/recipe.yaml:36` —
  the MoE sibling, `q_proj,k_proj,v_proj,o_proj`.
- `dgx-manager/docs/superpowers/plans/2026-05-04-qwen36-27b-training.md:45` shows this was
  a *considered* decision, not an oversight: *"If GatedDeltaNet layers also have
  `q_proj`-named tensors, LoRA would attach to stateful linear-attn layers — same
  regression mode as Mamba layers. We must verify the suffix discriminates before
  training."*

**How strong is the evidence against this repo?** Weaker than it first looks, and I want to
be precise about that:

- The **narrow list is MEASURED to work** on this architecture and this exact dataset:
  base **39% → 76%** exact match on `b-mc2/sql-create-context`, 100 held-out, 500 steps,
  `r=16` (`dgx-manager/docs/qwen3.6-27b-fine-tuning-on-dgx-spark.md:15,108-125`). So there
  is a known-good baseline that does *not* touch the GDN blocks.
- The **prohibition itself is ASSERTED, never measured.** It originates in the 35B-A3B doc
  as a statement about *Mamba SSM* layers and is carried forward by inheritance. No source
  in this corpus runs an A/B of GDN-LoRA vs no-GDN-LoRA. `dgx-manager`'s own framing is
  that suffix matching *conveniently* skips those layers — they never had to decide.
- The 48 GDN layers are **75% of the token-mixing blocks**. This repo's reasoning (that the
  Qwen3-era list adapts only a quarter of them) is sound on its face.

**Verdict:** this is a deliberate, unvalidated departure from the only configuration anyone
has ever made work on this architecture — not a proven defect. But it is the *one* axis
every prior run avoided, and if the LoRA run underperforms, this is the first suspect.

**Recommendation:** make the GDN block a knob (e.g. `LORA_TARGET_GDN=0|1`, default matching
the proven list) and smoke both at the 25-step scale the map already settled. That converts
an untested belief into a measurement for the price of one extra smoke run.

## C2 (LOUD, and this repo is the one that's RIGHT) — `gate_proj/up_proj/down_proj` exist on all 64 layers. The sibling repo's comments say otherwise and are wrong.

`dgx-manager-fine-tune-recipes` states, in two places, that the MLP names live only on the
full-attention block:

- `recipes/qwen3.6-27b-base-lora-attn-mlp/train.py:36-40` — *"The
  `q/k/v/o/gate/up/down_proj` names appear **ONLY** on the full-attn block — GatedDeltaNet
  uses `in_proj`/`out_proj`. So PEFT suffix matching automatically skips the linear-attn
  layers (verified via static check on the safetensors index)."*
- `recipes/qwen3.6-27b-base-lora-attn-mlp/recipe.yaml:49-52` — *"GatedDeltaNet
  (linear-attn) layers **don't expose** `gate_proj`/`up_proj`/`down_proj` — they use
  `in_proj`/`out_proj` — so suffix matching still automatically skips them."*

**That is false.** Primary source, `transformers` v5.16.1
`src/transformers/models/qwen3_5/modeling_qwen3_5.py:743-752` —
`Qwen3_5DecoderLayer.__init__` picks the token mixer conditionally and then builds the MLP
**unconditionally**:

```
743  class Qwen3_5DecoderLayer(GradientCheckpointingLayer):
749          self.linear_attn = Qwen3_5GatedDeltaNet(config, layer_idx)
751          self.self_attn = Qwen3_5Attention(config, layer_idx)
752      self.mlp = Qwen3_5MLP(config, config.intermediate_size)
```

and `Qwen3_5MLP` (lines 707-719) is the SwiGLU triple `gate_proj` / `up_proj` / `down_proj`.
Corroborated independently by `spark-vllm-docker/recipes/qwen3.6-27b-bf16.yaml:5-6`, which
describes the block pattern as *"hybrid 16 × (3× Gated-DeltaNet → **FFN**, 1×
Gated-Attention → **FFN**)"* — i.e. an FFN after **every** layer, 64 in total. The HF
`config.json` carries a single top-level `intermediate_size: 17408`, not a per-layer-type
pair.

**Consequences, both directions:**

1. **This repo's `# all 64 MLPs` comment is CONFIRMED correct.** No change needed.
2. **The sibling `attn-mlp` recipe has been LoRA-ing all 64 MLPs while believing it touched
   16.** Its stated capacity figure — *"10.5M → ~30M trainable params"*
   (`recipe.yaml:44-47`) — is inconsistent with that: at `r=16`, hidden 5120, intermediate
   17408, MLP LoRA over 64 layers is `64 × 3 × 16 × (5120+17408) ≈ 69M`, not ~19M. So
   either that figure is an estimate rather than a reading, or something else is going on.
   Worth reporting upstream to `dgx-manager-fine-tune-recipes`.
3. **Silver lining for this repo:** that same recipe is *"Validated end-to-end at seq=8192
   on single-node 50-step (eval_loss=0.105) and 4-node 5-step (eval_loss=0.316)"*
   (`recipe.yaml:5`). If it was in fact adapting all 64 MLPs, that is an unwitting
   end-to-end validation of exactly the MLP half of this repo's target list.

## C3 — `--language-model-only` is REAL, but this repo describes the wrong behaviour, and its stated benefit is not what the flag does.

This repo's `scripts/serve.sbatch:72-75`:

> `--language-model-only` — *"Skips building the vision tower, freeing memory for KV cache.
> This flag is reported to exist for this model family but I have not verified it against
> vllm 0.28.0."*

**It exists at exactly this repo's pin.** vLLM `v0.28.0`, `vllm/config/multimodal.py`:

```
language_model_only: bool = False
"""If True, disables all multimodal inputs by setting all modality limits to 0.
Equivalent to setting `--limit-mm-per-prompt` to 0 for every modality."""
```

with `if self.language_model_only: return 0` inside `get_limit_per_prompt`. Since it is a
field on the multimodal config dataclass, vLLM derives the CLI flag from the field name.

**And it appears somewhere real, measured:**
`community-recipe-registry/recipes/qwen3.6-35b-a3b/cipherfoxie/qwen3.6-35b-a3b-autoround-int4-dflash-vllm-cipherfoxie.yaml`
sets `extra_flags: --language-model-only` as its production default alongside
`default_chat_template_kwargs: '{"enable_thinking":false}'`, and its `README.md:23-25` says
*"Defaults to text-only (`--language-model-only`), which is the measured config… Verified
on GB10."* That is a `qwen3_5`-family multimodal checkpoint serving as a daily driver with
this exact flag pair. **This resolves the repo's "Known risks" item 1: the flag will not be
rejected.**

**But the description in `serve.sbatch` is wrong on two counts:**

1. It does **not** "skip building the vision tower" — it sets every modality limit to 0,
   i.e. it refuses image *inputs*. The docstring says exactly that.
2. The memory-freeing job belongs to a *different* flag. Same file, `skip_mm_profiling`:
   *"skips multimodal memory profiling and only profiles with language backbone model
   during engine initialization. This reduces engine startup time but shifts the
   responsibility to users for estimating the peak memory usage of the activation of
   multimodal encoder and embedding cache."*
   `dgx-manager/recipes/dgxrun/qwen3.8-27b-bf16.yaml:206-211` corroborates by treating the
   three flags as a set and noting the ViT *"must stay live and mm profiling must run so
   its memory is actually reserved"*.

**Also worth noting:** for the *merged* checkpoint this repo serves by default, the flag is
a **no-op** — the merged model is produced by `Qwen3_5ForCausalLM` and has no vision tower
at all (see C5). It only means anything for the base-model A/B server on port 8001.

**Fix:** correct the comment; if memory headroom is actually the goal, the flag you want is
`--skip-mm-profiling` (present at v0.28.0, same file), and only for the base-model server.

Residual **UNVERIFIED**: I confirmed the config *field*, not the argparse *spelling*, at
0.28.0. `dgx-manager/docs/qwen3.8-model-survey.md:270` flags the same gap
(*"the argparse spelling at our pinned image tag — check `vllm serve --help` in the
container first"*). One `vllm serve --help 2>&1 | grep language` settles it.

## C4 — "TP in {1,2,4,8} should be safe" is an unearned claim, and the stated reason TP=16 fails is wrong.

This repo, `scripts/serve.sbatch:18-22`: *"the head counts (24 query / 4 KV / 16 linear-key
/ 48 linear-value) constrain the valid sizes. TP in {1,2,4,8} should be safe; TP=16 is
expected to fail on the linear-key head count."*

The head counts are **CONFIRMED** from `config.json`: `num_attention_heads: 24`,
`num_key_value_heads: 4`, linear key heads 16, linear value heads 48, linear key/value head
dim 128, `linear_conv_kernel_dim: 4`.

**But the arithmetic points at a different head.** At TP=16 the 16 linear-**key** heads
divide *exactly* (one per rank). The indivisible count is the **24 query heads**
(24/16 = 1.5). The conclusion (`TP=16` invalid) survives; the stated reason does not.

**And "TP∈{1,2,4,8} should be safe" is contradicted by a real reproduction.**
`dgx-manager/docs/vllm-issue-draft-tp4-hang.md` is a full upstream-issue draft written from
an actual reproduction: *"Deploying **any Qwen3.5 or Qwen3.6 family model** with
`--tensor-parallel-size 4` … consistently hangs **silently** after weight load completes,
with no error, no exception, no NCCL timeout"* (lines 39-42), reproduced on
`Qwen/Qwen3.6-27B-FP8`, `Qwen3.5-397B-A17B-FP8`, and an int4 build, with *"TP=1 and TP=2
work perfectly on the same hardware/software, so the problem is in the cluster coordination
path that engages at TP≥3"* (line 51-53). They ruled out node parity, `fastsafetensors`,
CUDA-graph capture, and Marlin (lines 116-169), and found *"no public report exists of vLLM
successfully serving a Qwen3.5/3.6 family model at TP=4 on DGX Spark"* (line 211-213).

**Honest scoping:** that reproduction is 4× DGX Spark, sm_121, multi-node Ray executor,
vLLM `0.20.1rc1.dev152`. This repo is single-node 8× B300 (sm_103) on vLLM 0.28.0. It is
**not** evidence that TP=4 fails here. It **is** evidence that "should be safe" is an
inference nobody has ever validated on this family, on any hardware, above TP=2.

Adjacent, quantization-only: `spark-vllm-docker/recipes/qwen3.6-27b-bf16.yaml:9-14` — with
INT4-AutoRound at any TP>1, `num_v_heads = 48` violates Marlin's `MIN_THREAD_N=64` floor
and engine init fails. *"BF16 doesn't go through Marlin, so this recipe (TP=1, BF16) avoids
the issue entirely."* Not applicable here (BF16), but it is a second head-count landmine on
this exact layout.

The map already lists "whether TP > 1 is worth testing at all" as unspecified. This is the
evidence for leaving TP=1 alone.

## C5 — `merge_lora.py` produces precisely the artifact a sibling repo says vLLM can't load. It probably *does* load on this repo's pins — but nobody has run it.

`dgx-manager-fine-tune-recipes/recipes/qwen3.6-27b-base-lora-attn-mlp/recipe.yaml`
(`scripts.merge` comment block, lines 18-24), for the **dense 27B** — the same class family
as this repo's model:

> *"We don't use `scripts/merge.py` because it relies on PEFT's `merge_and_unload()`, which
> strips Qwen 3.6's multimodal wrapper and produces a **`model_type=qwen3_5_text` leaf
> config that vLLM can't load**."*

The MoE analogue is documented in full at
`dgx-manager/docs/qwen3.6-fine-tuning-on-dgx-spark.md:26,120-138`: *"`config.json` has
`"model_type": "qwen3_5_moe_text"` — a leaf config that transformers/vLLM doesn't register.
You get `KeyError: 'qwen3_5_moe_text'` on load."* Root cause (line 136):
*"`merge_and_unload()` operates on the inner `Qwen3_5MoeForCausalLM`. `save_pretrained()`
writes that inner class's config. The outer `…ForConditionalGeneration` wrapper — which
vLLM needs to dispatch the arch — is gone."* Their fix was to abandon PEFT merging entirely
and add LoRA deltas into the base safetensors in place
(`dgx-manager-fine-tune-recipes/scripts/merge_qwen3moe.py`, docstring lines 1-27), whose 2D
`delta = (B @ A) * (alpha/r)` case is *"what the 27B uses"*
(`qwen3.6-27b-fine-tuning-on-dgx-spark.md:88`).

**This repo does exactly the thing they abandoned.** `scripts/merge_lora.py:40-53`:
`AutoModelForCausalLM.from_pretrained` → `PeftModel.from_pretrained` →
`merge_and_unload()` → `save_pretrained()`. And the resulting config is predictable from
primary source: `transformers` v5.16.1 `configuration_qwen3_5.py:56` gives
`Qwen3_5TextConfig.model_type = "qwen3_5_text"`, and `modeling_qwen3_5.py:1578-1584` shows
`Qwen3_5ForCausalLM` declares `config: Qwen3_5TextConfig`. So the merged `config.json`
**will** read `model_type: "qwen3_5_text"`, `architectures: ["Qwen3_5ForCausalLM"]`, with no
`vision_config` and no `mtp.*` / `model.visual.*` tensors on disk.

**Counter-evidence at this repo's exact pins — the failure may not transfer:**

- `transformers` v5.16.1 `src/transformers/models/auto/modeling_auto.py:838` maps
  `("qwen3_5_text", "Qwen3_5ForCausalLM")` — and line 837 maps `qwen3_5_moe_text` too, the
  very key `dgx-manager` got a `KeyError` on. So their break was a version artifact.
- vLLM `v0.28.0` `vllm/model_executor/models/registry.py:203` registers
  `"Qwen3_5ForCausalLM": ("qwen3_5", "Qwen3_5ForCausalLM")` in the text-generation table
  (the multimodal `Qwen3_5ForConditionalGeneration` is separately at line 592).

**Verdict:** the class this repo's merge emits **is** registered in both libraries at the
pinned versions, so C5 is a *risk*, not a confirmed break — and the sibling repo's blanket
"vLLM can't load it" is stale. But nobody in this corpus has served a `qwen3_5_text` merge
from vLLM 0.28.0. **Smoke `vllm serve` on the merged directory before anything expensive
runs on it.** If it fails, the fix already exists and is 2D-only for a dense model:
`merge_qwen3moe.py`.

---

# CONFIRMATIONS

## Architecture — every claim in `sft_common.py`'s docstring holds

From `huggingface.co/Qwen/Qwen3.8-27B/raw/main/config.json` (primary): `model_type:
qwen3_5`, `architectures: ["Qwen3_5ForConditionalGeneration"]`, 64 layers, hidden 5120,
`intermediate_size` 17408, `head_dim` 256, 24 attention heads, 4 KV heads,
`full_attention_interval: 4`, linear key heads 16 / value heads 48 / head dim 128,
`linear_conv_kernel_dim: 4`, vision depth 27 / hidden 1152,
`max_position_embeddings: 262144`, stamped `transformers_version 5.8.0.dev0`.

| Repo claim | Verdict | Source |
|---|---|---|
| `model_type: "qwen3_5"` / `Qwen3_5ForConditionalGeneration` | CONFIRMED | HF `config.json` |
| ships a vision tower + MTP head in the checkpoint | CONFIRMED | HF `config.json` (`vision_config`, `mtp_num_hidden_layers: 1`); `qwen3.8-model-survey.md:58` counts **333 `model.visual.*` tensors** and **15 `mtp.*` keys** in the BF16 index |
| `AutoModelForCausalLM` → `Qwen3_5ForCausalLM` with `_keys_to_ignore_on_load_unexpected = [r"^mtp.*", r"^model.visual.*"]` | CONFIRMED **verbatim** | `modeling_qwen3_5.py:1578-1584` |
| 16 of 64 layers classic `self_attn` with q/k/v/o_proj, 48 Gated DeltaNet | CONFIRMED | `config.json` `full_attention_interval: 4`; `modeling_qwen3_5.py:743-751`. Full-attn layer indices are `[3,7,11,…,63]` per `qwen3.6-27b-fine-tuning-on-dgx-spark.md:42` |
| GDN projections named `linear_attn.in_proj_{qkv,z,a,b}` and `linear_attn.out_proj` | CONFIRMED **exactly** | `modeling_qwen3_5.py:423,427-430` (`out_proj`, `in_proj_qkv`, `in_proj_z`, `in_proj_b`, `in_proj_a`) |
| `in_proj_a` / `in_proj_b` are `[48, 5120]`, so rank > 48 is degenerate | CONFIRMED | `modeling_qwen3_5.py:429-430` — both are `nn.Linear(hidden_size, num_v_heads)`; `num_v_heads` = 48, hidden = 5120 |
| `conv1d` is `nn.Conv1d` (depthwise), not a Linear | CONFIRMED | `modeling_qwen3_5.py:405`; the depthwise `groups=hidden_size` call at line 212 |
| `_no_split_modules` still lists `Qwen3_5VisionBlock`, so `transformer_layer_cls_to_wrap` must be set explicitly | CONFIRMED | `modeling_qwen3_5.py:803` and `:1229` — `["Qwen3_5DecoderLayer", "Qwen3_5VisionBlock"]` |
| `peft` ships no default target-module mapping for `qwen3_5`, so an explicit list is mandatory | CONFIRMED | `peft` v0.20.0 `src/peft/utils/constants.py` — `"qwen2"` and `"qwen3"` entries exist (lines 101-102, 341-342 etc.); **no `qwen3_5` / `qwen3_5_text` key anywhere in the file** |
| `use_kernels=True` and `attn_implementation=` are real `from_pretrained` kwargs in v5 | CONFIRMED | `transformers` v5.16.1 `modeling_utils.py:4150,4396` (`use_kernels`), `:1506-1507,4264` (`attn_implementation`) |
| without Hub kernels, 48/64 layers fall back to the slow pure-PyTorch path | CONFIRMED | `modeling_qwen3_5.py:385` names the fallbacks: `torch_recurrent_gated_delta_rule`, `torch_chunk_gated_delta_rule`, `causal_conv1d_fn`, `causal_conv1d_update` — the exact function this repo's `requirements.txt:23` calls out |
| ~26.9B-param text stack | CONSISTENT | `qwen3.6-27b-fine-tuning-on-dgx-spark.md:56` reports total **26,906,484,224** params for the loaded model |
| ~50 GB of BF16 weights | CONFIRMED, refine to **51.7 GiB** | `qwen3.8-model-survey.md:42,47` — *"Sizes are exact, summed from HF blob metadata, not estimated"* |
| KV cache unusually cheap because only 16 layers cache | CONFIRMED and MEASURED | `qwen3.8-27b-bf16.yaml:54-59` computes 64 KiB/token (`4 KV heads × 256 head_dim × 2 × 2 bytes`) and **measured a 716,119-token KV pool at `max_model_len 262144`** on a 121.6 GB GB10 |
| native context 262144 | CONFIRMED | `config.json`; `qwen3.8-model-survey.md:65-68` adds that the 1M headline needs explicit YaRN and that Qwen warns static YaRN *"potentially impacts performance on shorter texts"* |
| no `--trust-remote-code` needed | CONFIRMED | `qwen3.8-model-survey.md:100-102` — the multimodal processor is native in-tree for 3.8; contrast `spark-vllm-docker/recipes/qwen3.6-27b-bf16.yaml:64`, which *does* pass it for 3.6 |

## Thinking mode and the chat template — all four claims confirmed against the template itself

Primary source: `huggingface.co/Qwen/Qwen3.8-27B/raw/main/chat_template.jinja`.

| `sft_common.py` claim | Verdict | Template evidence |
|---|---|---|
| thinking is ON by default | CONFIRMED | guard is `enable_thinking is undefined or enable_thinking is true` |
| default `reasoning_effort` is `'xhigh'` | CONFIRMED | `reasoning_effort|default('xhigh')` |
| with thinking on, the template injects *"Reasoning effort is set to xhigh. Please think carefully through…"* into the system message | CONFIRMED, verbatim | *"Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer."* (a `low` variant exists; `medium` injects nothing) |
| `add_generation_prompt=True` + `enable_thinking=False` emits a **pre-filled empty** `<think>\n\n</think>\n\n` | CONFIRMED | thinking off → `'<think>\n\n</think>\n\n'`; thinking on → `'<think>\n'` (opened, unclosed) |

The "thinking on emits an *unclosed* `<think>\n`" detail is what makes this repo's
*"`--reasoning-parser qwen3` is effectively mandatory"* comment right for the
thinking-enabled case.

**The masking rationale is backed by a real observed failure elsewhere.** `dgx-manager` hit
the un-masked version of this on the architecturally identical 27B
(`qwen3.6-27b-fine-tuning-on-dgx-spark.md:27,140-144`): *"The base model emits clean SQL
directly; **LoRA training induces a verbose `<think>...</think>` preamble**"* — base 0/100
outputs contain `</think>`, but tuned models do; at 50 steps only **17/100** outputs reached
`</think>` inside a 512-token budget, capping accuracy at 21%. Their suspected cause, never
chased: *"the chat template under the LoRA-saved tokenizer is rendering with
`enable_thinking=True` while the base model's tokenizer renders without."* This repo's
`enable_thinking=False` + masking the pre-filled block is precisely the mitigation for that
failure. Nobody has verified the mitigation works, but the failure it targets is real and
measured.

**Independent hardware confirmation that `enable_thinking=False` is the right call for a
terse task** — on Qwen3.8-27B itself, not an analogue
(`qwen3.8-model-survey.md:346-361`, repeated at `qwen3.8-27b-bf16.yaml:562-577`):

| Setting | Tokens | Result |
|---|---|---|
| default (`xhigh`) | >4000, `finish=length` | **completely empty** — both `content` *and* `reasoning_content` empty |
| `reasoning_effort: low` | 564, `finish=stop` | full answer |
| `enable_thinking: false` | 356, `finish=stop` | full answer |

*"Not truncated — empty."* At `xhigh` this silently scores as wrong with no error recorded
anywhere; their GPQA null rates were 33% at a 4096 cap and still 13.5% at 32768
(`qwen3.8-27b-bf16.yaml:331-337`).

## `--reasoning-parser qwen3` + `enable_thinking: false` is SAFE — traced end to end in vLLM 0.28.0

I went looking for a specific silent failure — that with `enable_thinking=false` the model
emits no `</think>` at all, so a `<think>`-seeking reasoning parser would sweep the whole
SQL into `reasoning_content` and leave `message.content` empty. **It does not happen at
v0.28.0.** The chain:

1. `vllm/entrypoints/openai/chat_completion/serving.py:136,151` — `--default-chat-template-kwargs`
   arrives as `default_chat_template_kwargs` and is stored.
2. `:186-195` — `_effective_chat_template_kwargs()` merges it via `.with_defaults(self.default_chat_template_kwargs)`.
3. `:249` — `chat_template_kwargs = self._effective_chat_template_kwargs(request)`.
4. `:361-363` — passed straight into the parser as
   `reasoning_parser_kwargs={"chat_template_kwargs": chat_template_kwargs}`.
5. `vllm/parser/qwen3.py:225-226` — `chat_kwargs = kwargs.get("chat_template_kwargs", {}) or {}`;
   `self.thinking_enabled = chat_kwargs.get("enable_thinking", True)`.
6. `vllm/parser/qwen3.py:100` — `initial_state=ParserState.REASONING if thinking else ParserState.CONTENT`.
7. `vllm/parser/qwen3.py:247-254` —
   ```python
   def extract_reasoning(self, model_output, request):
       if not self.thinking_enabled:
           return None, model_output
   ```

So with the repo's flags the parser starts in `CONTENT` and returns
`(reasoning_content=None, content=<full output>)`. **The SQL lands in `message.content`.**
`serve.sbatch`'s flag pair is correct as written.

The `--default-chat-template-kwargs` flag itself is CONFIRMED present at v0.28.0 both from
that source and from `dgx-manager/recipes/dgxrun/qwen3.8-27b-nvfp4-rtx.yaml:244` —
*"To change it for EVERY request, vLLM v0.28.0 has `--default-chat-template-kwargs`"* — a
recipe running the stock `vllm/vllm-openai:v0.28.0` image.

One thing that chain also reveals, and it cuts against this repo: `vllm/parser/qwen3.py:130-135`
*"Absorb duplicate `</think>` — model may emit it after already transitioning to CONTENT;
drop it silently."* If the label masking were wrong and the model learned to emit a second
empty think block, **the server would hide it**. That bug would surface only in
`evaluate.py`'s exact-match numbers (where `scripts/evaluate.py:53-54` splits on
`</think>`), never in a `query.sh` eyeball test.

## Hyperparameters that are known to work on this family

| Source | Model | r | α | dropout | LR | seq | bs/dev | accum | eff. batch | Outcome |
|---|---|---|---|---|---|---|---|---|---|---|
| `dgx-manager-fine-tune-recipes/recipes/qwen3.6-27b-base-lora/recipe.yaml:30-41`; doc `:66` | Qwen3.6-27B (**same arch**) | 16 | 16 | 0.0 | 2e-4 | 256 | 1 | 4 | 4 / 8 | **MEASURED: 39% → 73% (1 node) / 76% (2 nodes)** exact match, `b-mc2/sql-create-context`, 100 held out, 500 steps, warmup 5 |
| `.../qwen3.6-27b-base-lora-attn-mlp/recipe.yaml:30-41` | Qwen3.6-27B | 16 | 16 | 0.0 | 2e-4 | 256 (run at 8192) | 1 | 4 | 8 | validated 50-step single-node `eval_loss=0.105`; 4-node 5-step `0.316` |
| `dgx-spark-unsloth-qwen3.5-training/train.py:156-196,237-249` | Qwen3.5-35B-A3B | 16 | =r | 0 | 2e-4 | 2048 | 2 | 4 | 8 | bf16 LoRA, one DGX Spark; `optim="adamw_8bit"`, warmup 5. Explicitly **not** QLoRA: *"Unsloth explicitly recommends against 4-bit quantization for Qwen3.5 MoE models due to quality degradation"* (`:175-176`) |
| `nebius-slurm-ml-training-and-inference-demo/demo/scripts/train_32b_lora.sbatch:32-39` | Qwen3-32B (ancestor) | 16 | 32 | 0.05 | 2e-4 | 1024 | 2 | 4 | **128** | 84% per this repo's README |
| **THIS REPO** `train_lora.sbatch:39-47` | Qwen3.8-27B | **32** | **64** | 0.05 | 2e-4 | 1024 | 8 | 1 | **128** | unrun |

What that table says:

- **`LEARNING_RATE=2e-4` is CONFIRMED** — all four sibling recipes use it, on three
  different model families.
- **`MAX_SEQ_LEN=1024` is safely bracketed** — 256 and 8192 both ran on this architecture.
- **Effective batch 128 is inherited from the ancestor, not from anything measured on this
  architecture.** Every measured `qwen3_5`-family run used effective batch 4-12. This isn't
  wrong (B300 has the memory; `train_lora.sbatch:36-38` argues it correctly), but the
  76%-at-500-steps reference point was produced at batch 8, and 500 steps at batch 128 is
  ~16× the tokens. Step-count intuitions from `dgx-manager` do **not** transfer.
- **`r=32, alpha=64` is untested on this family.** Every measured run is `r=16`. The
  α/r = 2.0 ratio matches the ancestor; `dgx-manager` used 1.0. `dropout=0.05` matches the
  ancestor; every `qwen3_5` recipe used 0.0.
- **Trainable-parameter reference point:** `q/k/v/o` only at `r=16` measured
  **10,485,760 / 26,906,484,224 = 0.0390%** (`qwen3.6-27b-fine-tuning-on-dgx-spark.md:56`).
  This repo's README claims ~0.4% for its wider list at `r=32` — ~10× that, which is
  plausible for 10 module types at double the rank, but unverified.
  `dgx-manager`'s recipes hard-fail if trainable% < 0.001% as a mis-targeting tripwire
  (`qwen3.6-27b-base-lora-attn-mlp/train.py:276-281`) — a cheap guard this repo lacks
  (it only `print`s, `train_lora.py:79-80`).

## Serving flags actually used for this family

| Flag | Used by | Note |
|---|---|---|
| `--reasoning-parser qwen3` | every single one: `qwen3.8-27b-bf16.yaml:134`, `qwen3.8-27b-nvfp4-rtx.yaml:159`, `spark-vllm-docker/recipes/qwen3.6-27b-bf16.yaml:59`, cipherfoxie | CONFIRMED. Parser name verified in-source (`vllm/parser/qwen3.py:213 CONFIG_NAME = "qwen3"`) |
| `--default-chat-template-kwargs` | `qwen3.8-27b-bf16.yaml:136` (`reasoning_effort: medium`), cipherfoxie (`enable_thinking: false`) | CONFIRMED at v0.28.0 |
| `--tool-call-parser qwen3_coder` + `--enable-auto-tool-choice` | all agentic recipes | not needed for SQL |
| `--language-model-only` | cipherfoxie community recipe, **measured/verified on GB10** | see C3 |
| `--tensor-parallel-size 1` | every recipe in the corpus | TP=1 is universal here |
| `--speculative-config '{"method":"mtp",…}'` | `qwen3.8-27b-bf16.yaml:137` | MTP ships **inside** the checkpoint, no separate drafter (`qwen3.8-model-survey.md:321`). This repo passes none — fine, and it sidesteps the GDN prefix-cache interaction below |
| `--kv-cache-dtype fp8` | 3.6 recipes; deliberately **omitted** from the 3.8 BF16 recipe as a precision confound (`:142-145`) | this repo correctly omits it |
| `--attention-backend flashinfer` | 3.6 recipes; deliberately **left unset** for 3.8 because `vllm#51987` ("Revert FlashInfer XQA decode on SM12x") is open (`:201-204`) | this repo leaves it unset — matches |
| `--trust-remote-code` | 3.6 only, **not** 3.8 (`:267-268`) | this repo correctly omits |

---

# GOTCHAS THIS REPO HAS NOT ACCOUNTED FOR

Ordered by how likely they are to cost a run.

**G1 — the LoRA path raises no collective timeout.** Every `qwen3_5`-family recipe in
`dgx-manager-fine-tune-recipes` sets, before process-group init:

```python
torch.distributed.constants.default_pg_timeout      = datetime.timedelta(hours=4)
torch.distributed.constants.default_pg_nccl_timeout = datetime.timedelta(hours=4)
```

(`recipes/qwen3.6-27b-base-lora/train.py:175-176`,
`qwen3.6-27b-base-lora-attn-mlp/train.py:191-192`,
`qwen3.6-35b-a3b-base-lora/train.py:35-36`, and the two Gemma recipes.) Reason, stated at
`qwen3.6-27b-fine-tuning-on-dgx-spark.md:74`: *"bf16 ZeRO-3 of 27B does hundreds of
broadcasts during `from_pretrained`."* This repo raises `ddp_timeout` only in
`train_full.py`; `train_lora.py` runs on the default (~30 min) while doing
`cpu_ram_efficient_loading` sharded load of 51.7 GiB across 16 ranks. Same shape of risk,
no mitigation. Cheap to add.

**G2 — `datasets.map()` writing Arrow cache to shared NFS from 16 ranks.**
`dgx-manager-fine-tune-recipes/lib/dataset.py:231-247` passes `keep_in_memory=True` and
`load_from_cache_file=False`, with this comment:

> *"keep_in_memory=True: skip writing to the NFS-backed cache. Without this, multi-rank
> training has every rank simultaneously writing the same `.arrow` files; the resulting
> size races mean a rank later mmaps a file shorter than its on-disk header advertises and
> **dies with SIGBUS deep inside pyarrow**. The dataset is small enough to fit in RAM
> (~78K examples × 256 tokens × ~10B per token = ~200 MB)."*

This repo's `sft_common.py:113-125` calls `ds.map(...)` and `ds.filter(...)` with cache
defaults, on `/mnt/data` (shared), from 16 ranks, on the same ~75K-example dataset. Their
comment also notes `datasets`' fingerprint hashing *"misses changes behind a `lambda`"* —
and `prepare_datasets` passes a `lambda`, so edits to `build_example` may not invalidate a
stale cache. Both fixes are one kwarg each.

**G3 — no `TRITON_CACHE_DIR` isolation on the training jobs.** `serve.sbatch:47-49` already
does this (*"Triton/torch.compile caches on shared NFS race between concurrent jobs"*), but
`train_lora.sbatch` and `train_full.sbatch` do not — and `use_kernels=True` pulls fused
GDN / `causal_conv1d` kernels from the Hub which compile and cache. Mirror the `serve.sbatch`
treatment, plus `HF_HOME` if two jobs might run concurrently.

**G4 — expectation calibration for the README's headline.** This repo's Provenance section
cites the ancestor's *"88% and 84% exact match respectively from base rates of 2–3%"*. On the
**architecturally identical** Qwen3.6-27B, the same dataset and the same exact-match
methodology measured **base at 39%** (`qwen3.6-27b-fine-tuning-on-dgx-spark.md:110,118`), and
the note explains why: *"base output averages 164 chars and never opens a `<think>` block"*.
The tuned result was 76%. So on this architecture the realistic story is roughly
**39% → mid-70s**, i.e. a ~+37 pp delta, not a 2% → 88% transformation. Prompts and
normalization differ between the two harnesses so this is a **forecast, not a measurement** —
but the map's instruction to treat README claims as claims applies squarely here.

**G5 — eval-time logits are the memory event, not training.** `dgx-manager` calls
`per_device_eval_batch_size=1` *"load-bearing"*
(`qwen3.6-27b-base-lora-attn-mlp/train.py:23-31,293-303`): HF defaults eval batch to 8, and
with vocab 248,320 the eval forward materializes `[bs, seq, 248320]` in bf16 and then
`ForCausalLMLoss` casts `.float()`, doubling it — *"~65 GB … OOMs even when training fits"*
at seq 8192. This repo sets `per_device_eval_batch_size = per_device_bs = 8` at seq 1024, so
~4.1 GB bf16 + ~8.1 GB after the float cast, per rank. **B300's 275 GB absorbs that**, so
this is a note rather than a blocker — but it is the mechanism by which eval OOMs where
training fits, and it scales linearly with `MAX_SEQ_LEN`. Raising `MAX_SEQ_LEN` past ~4096
without pinning eval batch to 1 would be the trap.

**G6 — Liger fused linear cross-entropy is a free ~30% step-time win here, unused.**
`qwen3.6-27b-base-lora-attn-mlp/recipe.yaml:5` — *"Liger fused linear CE (~30% speedup,
bit-identical loss)"*, with the important caveat spelled out at `train.py:12-19`: **only**
the loss-layer fusion is safe; *"RoPE / RMSNorm / SwiGLU patches are left off because Qwen
3.6's hybrid GatedDeltaNet + attention is not validated against Liger's other Qwen3
kernels."* Their `train.py:65-135` also documents that Liger's `apply_liger_kernel_to_*`
functions **silently no-op** when the target arch module isn't loaded, so they verify the
patch landed and force-attach `lce_forward` if not. Worth knowing before reaching for it.

**G7 — `--enable-prefix-caching` is ON BY DEFAULT.** Corrected-by-measurement note at
`qwen3.8-27b-bf16.yaml:147-150`: *"vLLM defaults this to TRUE, so omitting the flag does NOT
disable it… To actually turn it off the flag is `--no-enable-prefix-caching`."* This repo's
`serve.sbatch` passes neither, so caching is on. Harmless for this workload; relevant if
anyone tries to measure cold prefill.

**G8 — hybrid-GDN prefix-cache hash unit is 816 tokens.** `qwen3.8-27b-bf16.yaml:156-175`
derives, from measurement on Qwen3.8-27B, `hits(L) = max(0, floor((L-1)/U) - 1) * U` with
`U = 816`, i.e. **zero reuse below 1632 prompt tokens**. Only manifests with
`--speculative-config` active. This repo passes no speculative config, so it's context —
but it's the reason a short-prompt SQL workload would see no cache reuse if MTP were ever
switched on.

**G9 — shared HF-cache permission collisions.**
`qwen3.6-27b-fine-tuning-on-dgx-spark.md:290-296`: a training run as root left a 0-byte
root-owned `_builder.lock` in the datasets cache; the next eval run as a normal user got
`EACCES` on `os.open(lock, O_RDWR)`. Only bites if jobs ever run as different users on the
same `/mnt/data` cache — worth knowing since the map notes `/mnt/data` is shared with other
users.

**G10 — no MTP/router freeze loop, and here that's correct — but say so.** Every
`dgx-manager` recipe carries a defensive loop freezing any parameter whose name contains
`mtp.`, `router`, or `.gate.weight`
(`qwen3.6-27b-base-lora-attn-mlp/train.py:256-264`), because *"suffix matching also
captures the MTP head's `q/k/v/o_proj`"* and *"the vision tower's attention"*
(`train.py:41-44`). This repo needs none of that — `Qwen3_5ForCausalLM` drops both
subtrees at load (`modeling_qwen3_5.py:1584`). But a reader arriving from `dgx-manager` will
look for the freeze loop and worry at its absence; one line in the docstring saying *"no
freeze loop needed, the MTP head and vision tower aren't instantiated"* would close that
gap. **Corollary worth checking once at runtime:** `model.print_trainable_parameters()`
should report zero modules matching `mtp` or `visual`. If it doesn't, the load path isn't
doing what the docstring says.

---

# STILL UNRESOLVED — no source settles these

1. **Does LoRA on the GDN `in_proj_qkv` / `in_proj_z` / `out_proj` help, hurt, or do
   nothing?** Nobody has measured it. The only statement in the corpus is an unmeasured
   assertion inherited from a Mamba-SSM MoE model (C1). This is the single highest-value
   unknown on this ticket, and one extra 25-step smoke would answer it.
2. **Is sequence packing genuinely unavailable?** `train_lora.sbatch:54-56` asserts the GDN
   recurrent state can't be reset mid-sequence. `dgx-manager-fine-tune-recipes/lib/args.py:24-30`
   exposes `--packing` (default `False`) and *does* wire it into the two Qwen3.6-27B recipes
   (`qwen3.6-27b-base-lora/train.py:282`, `…-attn-mlp/train.py:311`), citing *"+30-40% on the
   build123d set"* — but no recipe sets it, no run is recorded with it on, and nothing in the
   corpus says anything about GDN state. **Neither confirmed nor contradicted.**
3. **Does the merged `qwen3_5_text` checkpoint actually serve from vLLM 0.28.0?** Registration
   says yes; a sibling repo says no on older versions; nobody has run it on this stack (C5).
4. **Is `--language-model-only` the exact argparse spelling at 0.28.0?** The config field is
   confirmed; the derived flag name is not (C3).
5. **TP > 1 on 8× B300 single-node for this family.** No datapoint exists anywhere. The only
   real report is a multi-node sm_121 silent hang at TP≥3 (C4).
6. **The sm_103 `flash-linear-attention` determinism bug** (`requirements.txt:30-35`).
   Nothing in the sibling corpus mentions it — everything there is sm_121. Unverified from
   this direction; the commented-out pin is the right posture.
7. **Whether fine-tuning purely on non-thinking data degrades thinking mode.** This repo
   lists it as a known risk. `dgx-manager` observed the *opposite* direction — LoRA
   *inducing* verbose thinking where the base emitted none — but on an unmasked,
   `enable_thinking=True`-rendered pipeline, so it isn't the same experiment. Nothing
   measures the degradation this repo warns about.
8. **`~152MB/sequence` GDN recurrent state** (`sft_common.py:150`) and the `use_cache=False`
   rationale. Not corroborated anywhere; note that `dgx-manager`'s recipes call
   `fix_gemma4_use_cache(model)` which sets `model.config.use_cache = **True**`
   (`lib/patches.py:186-193`), described as a *"safe no-op on Qwen"* — so they run the
   opposite setting with no stated Qwen-specific reason. Not a contradiction, but nobody has
   measured the GDN state cost.

---

# Recommended follow-ups, in order

1. **Smoke `vllm serve` on a merged checkpoint before the real LoRA run finishes** (C5). If
   `model_type: qwen3_5_text` doesn't dispatch, the LoRA path has no serving story and the
   fix is a different merge script. This is the cheapest test with the largest blast radius.
2. **Make the GDN target block a knob and smoke both lists** (C1). Converts the map's worst
   silent-failure mode into a measurement.
3. **Add the 4-hour collective timeout and `keep_in_memory=True`** (G1, G2) — two lines, both
   guarding against failures that only appear at 16-rank scale.
4. **`vllm serve --help | grep -E 'language-model|skip-mm'`** and fix the `serve.sbatch`
   comment (C3), then decide whether `--skip-mm-profiling` is what was actually wanted.
5. **Correct the TP comment** (C4): the TP=16 blocker is the 24 query heads, and TP∈{2,4,8}
   is untested on this family with one recorded TP≥3 hang against it.
6. **Report C2 upstream** to `dgx-manager-fine-tune-recipes` — its `attn-mlp` recipe's
   comments and capacity estimate are wrong about where the MLPs live.
7. **Recalibrate the README's expected numbers** to the ~39% base / mid-70s tuned range that
   this architecture actually produced on this dataset (G4), pending the real run.
