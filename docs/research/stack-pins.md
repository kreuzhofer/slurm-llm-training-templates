# Stack pins, the vLLM `--language-model-only` flag, and the sm_103 determinism bug

Research for [issue #4](https://github.com/kreuzhofer/nebius-slurm-qwen38-lora-demo/issues/4).
Verified 2026-09-07. No packages were installed and no `pip` was run — this is
metadata, source and documentation research only.

## Method and what counts as a source here

Everything below is checked against the thing that owns the claim:

- **Package existence, dates, declared dependencies** — the PyPI JSON API
  (`https://pypi.org/pypi/<pkg>/<version>/json`), which serves the same
  `Requires-Dist` metadata pip resolves against.
- **vLLM behaviour** — the `vllm-0.28.0.tar.gz` sdist from PyPI, unpacked and
  grepped. File:line references are into that sdist.
- **transformers behaviour** — the `v5.16.1` git tag on
  `huggingface/transformers` (`raw.githubusercontent.com/.../v5.16.1/...`), which
  matches the 5.16.1 sdist.
- **PyTorch wheel contents** — `pytorch/pytorch` at the `v2.13.0` tag: the CI
  build-matrix and arch-list tables that generate the published wheels.
- **The model's architecture** — `Qwen/Qwen3.8-27B`'s own `config.json` from the
  HuggingFace Hub. (The map says the downloaded checkpoint is the primary source
  for architecture questions; the Hub copy of `config.json` is the same file,
  and issue #2 will produce the local one.)
- **PEP 440 / packaging semantics** — the PyPA version-specifier specification.

Where nothing settles a claim, it is marked **UNVERIFIABLE** with a note on what
would settle it.

---

## Contradictions first

Five things in the repo are wrong or misleading. None of them stop the install.
Two of them are *reasons* attached to correct conclusions, which is the kind of
error that survives a successful run and then misleads the next person.

### C1 — `requirements.txt`'s rationale for the cu130 index is wrong for torch 2.13.0

`requirements.txt` says:

> * B300 is compute capability 10.3 (sm_103). The default PyPI torch wheels and
>   anything built for CUDA 12.4 contain no sm_103 kernels and will not run.

Both halves of that are off for this particular version:

1. **The default PyPI `torch==2.13.0` *is* the CUDA 13.0 build.**
   `pytorch/pytorch@v2.13.0:.github/scripts/generate_binary_build_matrix.py:28`
   sets `CUDA_STABLE = "13.0"`, and PyPI's own metadata for `torch 2.13.0` lists
   `cuda-toolkit[...]==13.0.3`, `nvidia-cudnn-cu13==9.20.0.48`,
   `nvidia-nccl-cu13==2.29.7`. There is no CUDA-12.4 torch 2.13.0 to be pulled
   "over the top". CUDA 12.4 is not even in `CUDA_ARCHES`
   (`["12.6", "12.9", "13.0", "13.2"]`).
2. **Neither wheel contains sm_103 kernels — and that is fine.**
   `pytorch/pytorch@v2.13.0:.ci/manywheel/build_env_setup.py:72-89`:

   ```python
   TORCH_CUDA_ARCH_LIST_TABLE: dict[str, dict[str, set[int]]] = {
       ...
       "13.0": {
           "x86_64": {75, 80, 86, 90, 100, 120},
           "aarch64": {80, 90, 100, 110, 120},
       },
   ```

   `103` is absent, and release wheels ship **SASS only** — `_PTX_ARCHES = {120}`
   applies to nightlies, and `_ptx_arches()` returns the empty set on release
   builds (`build_env_setup.py:91-113`), so there is no PTX to JIT from either.

   PyTorch's own runtime compatibility table nevertheless declares sm_100 code
   valid on a 10.3 device — `torch/cuda/__init__.py:315`:

   ```python
   100: _CompatInterval(start=100, exclude={101}),
   ```

   `_CompatInterval.__contains__` accepts any device with the same major and a
   minor `>=` the start, so device cc `103` is covered by code cc `100`. So torch
   2.13.0+cu130 should run on B300 and should *not* print the
   `_warn_unsupported_code` warning.

   NVIDIA's own [Blackwell Compatibility Guide](https://docs.nvidia.com/cuda/blackwell-compatibility-guide/)
   agrees with the rule PyTorch encodes:

   > A cubin generated for a certain compute capability is supported to run on
   > any GPU with the same major revision and same or higher minor revision of
   > compute capability.

   10.0 → 10.3 is same major, higher minor. (That page says nothing about the
   `sm_100f`/`sm_103f` family-specific targets, which is a separate mechanism and
   not what these wheels use.)

**What it means for the code.** `torch==2.13.0+cu130` plus
`--extra-index-url https://download.pytorch.org/whl/cu130` is harmless and does
pin the install to the CUDA-13 index (the local label only exists there). But
the comment's reasoning is wrong, and `setup.sh`'s failure hint —

> `!! GPU verification FAILED. Most likely cause: torch wheel without sm_103 kernels. Check that torch reports '+cu130'.`

— would mis-diagnose: `+cu130` will be reported whether or not sm_103 works,
because no published 2.13.0 wheel has sm_103 SASS. If step 5 fails, the thing to
read is the `Found GPU0 ... which is of compute capability (CC) 10.3` warning
text from `torch/cuda/__init__.py:372-404`, not the version string. Also worth
noting: by the same table, `cu129` and `cu132` x86_64 wheels also contain cc
`100` and would be equally usable — the cu130 pin is a reasonable choice, not a
requirement.

### C2 — the reason given for `--tensor-parallel-size 16` failing is the wrong head count

`serve.sbatch` and README both say TP=16 "is expected to fail on the linear-key
head count". It will fail, but not there. Using the real head counts from
`Qwen/Qwen3.8-27B/config.json` (`num_attention_heads: 24`,
`num_key_value_heads: 4`, `linear_num_key_heads: 16`, `linear_num_value_heads: 48`,
`linear_key_head_dim: 128`, `linear_value_head_dim: 128`):

| constraint | code | TP=16? |
|---|---|---|
| 24 query heads | `qwen3_next.py:242` `assert self.total_num_heads % tp_size == 0` | **24 % 16 ≠ 0 → fails here** |
| 4 KV heads | `qwen3_next.py:248-252` — replicated when `kv_heads < tp_size`, needs `tp_size % 4 == 0` | 16 % 4 = 0, passes |
| 16 linear-key heads | `key_dim = 128*16 = 2048`, split by `divide()` in `linear.py:466` | 2048 % 16 = 0, **passes** |
| 48 linear-value heads | `divide(num_v_heads, tp)` in `mamba_utils.py:264` and `qwen_gdn_linear_attn.py:452` | 48 % 16 = 0, passes |

So the linear-key head count is the one thing at TP=16 that is *not* a problem.
The 24 query heads are.

### C3 — TP ∈ {1,2,4,8} is right, but for a reason the README doesn't give

The conclusion holds. Enumerating all TP ≤ 16 against the actual assertions:

- `divide()` (`vllm/distributed/utils.py:53-64`) is an `assert`, so violations
  are hard failures at model construction.
- `MergedColumnParallelLinear` divides **each** entry of `output_sizes`
  (`linear.py:466`). For Qwen3.5 the GDN in-projection is *not* fused:
  `qwen_gdn_linear_attn.py:549-553` gives `output_sizes = [2048, 2048, 6144, 6144]`.

| TP | 24 % TP | 2048 % TP | 48 % TP | verdict |
|---|---|---|---|---|
| 1 | 0 | 0 | 0 | OK |
| 2 | 0 | 0 | 0 | OK |
| 3 | 0 | **2** | 0 | fails on `key_dim`, **not** on any head count |
| 4 | 0 | 0 | 0 | OK |
| 6 | 0 | **2** | 0 | fails on `key_dim` |
| 8 | 0 | 0 | 0 | OK |
| 12 | 0 | **8** | 0 | fails on `key_dim` |
| 16 | **8** | 0 | 0 | fails on query heads |

`{1,2,4,8}` is exactly right for a single 8-GPU node. TP=3 and TP=6 pass every
head-count check and still fail, on the 2048-wide key projection.

### C4 — `use_kernels=True` does not give B300 the fused *layer* kernel

`requirements.txt` is right that `use_kernels=True` is required and right that
without it 48 of 64 layers take the pure-PyTorch path. But the whole-layer fused
Gated DeltaNet kernel is compute-capability-gated to SM121 only —
`transformers@v5.16.1:src/transformers/integrations/hub_kernels.py:156-168`:

```python
# GB10/SM121 GDN fast path (no fla/causal_conv1d build there); dense and MoE share it.
"Qwen3_5GatedDeltaNet": {
    Device(type="cuda",
           properties=CUDAProperties(min_capability=121, max_capability=121)):
        LayerRepository(repo_id="Atlas-Inference/gdn",
                        layer_name="Qwen3_5GatedDeltaNet", ...),
},
```

On sm_103 that mapping does not apply. What B300 *does* get from
`use_kernels=True` is the function-level kernels, which are not gated:
`chunk_gated_delta_rule` and `recurrent_gated_delta_rule` from
`kernels-community/fla` (`hub_kernels.py:197-222`) and `causal_conv1d_fn` /
`causal_conv1d_update` from `kernels-community/mamba-ssm`
(`hub_kernels.py:169-195`). That is still the fast path and worth having —
subject to those Hub repos actually resolving on the cluster, which I could not
check from here for `kernels-community/fla` (see §6d).

**What it means for the code.** `use_kernels=True` stays. Don't expect the
`1.49x prefill` number from `docs/model_doc/qwen3_5.md` — that is the SM121
layer kernel. And see C5: the function kernels are exactly where the determinism
question lands.

### C5 — the determinism-bug caveat is scoped to the wrong install path

`requirements.txt` says the chunked gated-delta-rule determinism bug is

> Only relevant if you install the classic fast path instead of using Hub kernels.

Every clause of that is off:

- The Hub path **is** FLA. `hub_kernels.py:197-210` maps
  `chunk_gated_delta_rule` to `repo_id="kernels-community/fla"` — vendored
  flash-linear-attention Triton source.
- The Hub copy **already carries the fix**, so it is the safe path, not the
  exposed one.
- The pip `fla` module is picked up **whether or not `use_kernels=True` is set**,
  because `use_kernel_func_from_hub_with_fallback` resolves the "original
  package" tier with a bare `importlib.import_module("fla")` at import time
  (`hub_kernels.py:839-847`).

The real rule is: **never let `fla-core < 0.5.2` be importable.** As written,
`requirements.txt` installs no FLA and nothing in the stack pulls `fla-core`
transitively, so the repo is **not exposed today** — which means this "known
risk" can be closed as not-applicable rather than carried. Full working in §6.

---

## Verdict table

| # | Claim (as the repo states it) | Verdict | Source | What it means for the code |
|---|---|---|---|---|
| 1.1 | All nine pinned versions exist on PyPI, none yanked | **CONFIRMED** | PyPI JSON API, all 200 with `yanked: false` | Nothing to change |
| 1.2 | Python 3.12.3 (the cluster's interpreter) satisfies every `Requires-Python` | **CONFIRMED** | tightest is `matplotlib 3.11.0 >=3.11`; `vllm <3.15,>=3.10` | `setup.sh`'s bare `python3 -m venv` is fine |
| 1.3 | The pins co-install (no unsatisfiable version conflict) | **CONFIRMED** (metadata-level) | shared-dep intersection non-empty; see §1 | Resolution should succeed; the real proof is issue #2 |
| 2.1 | `vllm==0.28.0` hard-pins `torch==2.13.0` | **CONFIRMED** | PyPI `Requires-Dist: torch==2.13.0`; sdist `requirements/cuda.txt:7` | Nothing to change |
| 2.2 | A `2.13.0+cu130` local-version wheel satisfies that pin | **CONFIRMED** | PyPA version-specifiers spec: local labels ignored when the specifier has none | The install order in `requirements.txt` works as designed |
| 2.3 | Installing torch first "stops pip pulling the non-CUDA-13 build over the top" | **CONTRADICTED** | plain PyPI `torch 2.13.0` *is* the cu130 build (`CUDA_STABLE = "13.0"`) — see C1 | Fix the comment; keep the pin |
| 2.4 | Default PyPI torch wheels "contain no sm_103 kernels and will not run" on B300 | **CONTRADICTED** | no 2.13.0 wheel has sm_103 SASS, but torch's `DEVICE_REQUIREMENT[100]` covers cc 10.3 — see C1 | Fix the comment and `setup.sh`'s failure hint |
| 3.1 | transformers 5.x defaults `fsdp_config["version"]` to `2` | **CONFIRMED** | `training_args.py:2777` `int(self.fsdp_config.get("version", 2))`; docstring `:674-675`; FSDP1 deprecation warning `:2817` says removal in v5.20 | `"version": 2` in `sft_common.fsdp_config()` is redundant but correct and self-documenting — keep it |
| 3.2 | `backward_prefetch`, `forward_prefetch`, `use_orig_params` are **silently** ignored under FSDP2 | **CONFIRMED** | all three read only inside the FSDP1 `else:` branch — `training_args.py:2823-2829`; nothing else in `src/transformers` touches them | Dropping them was correct. (Had they been forwarded, accelerate would `ValueError` on `forward_prefetch`) |
| 3.3 | `auto_wrap` is a documented no-op | **CONFIRMED, but not the thing the code uses** | the no-op is `TrainingArguments(fsdp="auto_wrap")` — `training_args.py:2848`, `:2886-2887` (`elif item == FSDPOption.AUTO_WRAP: pass`). There is no `fsdp_config["auto_wrap"]` key at all | Irrelevant to `sft_common`, which sets `auto_wrap_policy` |
| 3.4 | The `fsdp_config` keys `sft_common` does set are honoured under FSDP2 | **CONFIRMED** | `auto_wrap_policy` `:2784-2787`; `transformer_layer_cls_to_wrap` `:2790-2796`; `reshard_after_forward` `:2810-2813`; `activation_checkpointing` `:2805-2808`; `cpu_ram_efficient_loading` `:2798-2801`; `state_dict_type` `:2803`. Applied by `accelerate/utils/fsdp_utils.py:665,744-748,810,830` | `fsdp_config()` is a live config, not a dead pass-through |
| 3.5 | `activation_checkpointing` is preferred over `TrainingArguments(gradient_checkpointing=True)` | **CONFIRMED** | `docs/source/en/fsdp.md:78` "Use this instead of gradient checkpointing in `TrainingArguments`. Setting both raises an error."; hard guard `trainer.py:843-849` | Keep it. Never set both — that is a `ValueError`, not a warning |
| 3.6 | The reason is a redundant AllGather in the backward pass under FSDP | **CONFIRMED** | transformers' own runtime warning `training_args.py:2747-2752` uses that exact wording and cites the same issue; docstring `:681-684` repeats it. The docs recommend the swap but never give this reason | Nothing to change |
| 3.7 | `transformers#30404` says that | **CONFIRMED** | [#30404](https://github.com/huggingface/transformers/issues/30404), "[FSDP] redundant additional allgather during backward when using FSDP FULL_SHARD with gradient checkpointing", by `yundai424`, opened 2024-04-22, closed 2024-07-09 | Citation is accurate. Could additionally cite `training_args.py:2747-2752` |
| 4.1 | `use_kernels=True` is the correct load-time switch in 5.16.1 | **CONFIRMED** | `modeling_utils.py:4150` `kwargs.pop("use_kernels", False)` → `:4396` `model.set_use_kernels(...)`; documented in `docs/source/en/kernels.md:35` | Nothing to change. Note it is absent from the `from_pretrained` docstring |
| 4.2 | It fetches fused Gated DeltaNet + causal-conv1d kernels for this model | **CONFIRMED (function-level), CONTRADICTED (layer-level on sm_103)** | `hub_kernels.py:156-222`; layer kernel gated to cc exactly 121 — see C4 | Keep `use_kernels=True`; correct the expectation |
| 4.3 | `kernels>=0.16.0,<0.17` is the right range | **CONFIRMED** | identical to transformers 5.16.1's own `setup.py:93` pin; resolves to `kernels 0.16.1` (latest; no 0.17 exists as of 2026-09-07) | Nothing to change |
| 4.4 | `Trainer(processing_class=...)` is correct for v5 and `tokenizer=` is gone | **CONFIRMED** | `trainer.py:368-387` — `processing_class` present, no `tokenizer`, and **no `**kwargs`**, so `tokenizer=` raises `TypeError` | Any leftover `tokenizer=` is a hard crash, not a warning |
| 4.5 | `dtype=` is correct and `torch_dtype=` is deprecated in v5 | **CONFIRMED** | `dtype` is the documented param (`modeling_utils.py:4008-4009`); `torch_dtype` popped "kept for BC" `:4130-4131`, merged silently `:4158-4160`; warns via `_from_config` `:1488-1493` and the config property `configuration_utils.py:461-474` | `dtype=torch.bfloat16` is right. The code comment says "deprecated", which is accurate — it is **not** removed |
| 5.1 | `--language-model-only` exists as a vLLM 0.28.0 CLI flag | **CONFIRMED** | registered at `vllm/engine/arg_utils.py:1297`; field `vllm/config/multimodal.py:101` | Keep the flag; drop the README hedge |
| 5.2 | It exists "for this model family" | **CONFIRMED** | `docs/models/supported_models.md:508` names Qwen-3.5 explicitly; `Qwen3_5ForConditionalGeneration` registered at `registry.py:592` | — |
| 5.3 | It "skips building the vision tower, freeing memory for KV cache" | **CONFIRMED** | `language_model_only` → `get_limit_per_prompt() == 0` (`multimodal.py:492`) → `_mark_tower_model` replaces the tower with `StageMissingLayer` under `no_init_weights` (`interfaces.py:296-338`); the tower is built inside that context at `qwen3_5.py:495-501` | Comment is accurate |
| 5.4 | (implied) the flag helps on the *merged fine-tuned* model too | **CONTRADICTED (harmlessly)** | `save_pretrained` rewrites `config.architectures` to `["Qwen3_5ForCausalLM"]` (`modeling_utils.py:3418`), which `registry.py:203` registers as text-only → `ModelConfig.__post_init__` never builds a `multimodal_config` (`config/model.py:752`) → the flag's `InitVar` is discarded with no error | The flag is a silent no-op on the merged model and does real work on the base model served for A/B. Safe either way — no code change needed |
| 6.1 | A determinism bug exists in the chunked gated-delta-rule kernel | **CONFIRMED** | [fla#945](https://github.com/fla-org/flash-linear-attention/issues/945), opened 2026-06-12, closed 2026-06-20, label `bug`; corroborated by [Liger-Kernel#1255](https://github.com/linkedin/Liger-Kernel/issues/1255), [triton#10590](https://github.com/triton-lang/triton/issues/10590), [VeOmni#1101](https://github.com/ByteDance-Seed/VeOmni/pull/1101) | Real, and worse than it sounds: the backward recomputes the forward state, so gradients disagree with the loss |
| 6.2 | It "affects sm_103 specifically" | **CONFIRMED** | the sm_100 sibling ([triton#9871](https://github.com/triton-lang/triton/issues/9871)) was fixed in Triton 3.7.0; sm_103 still failed after that, with a wider config envelope, and did not reproduce on sm_90 | Root cause is a Triton miscompile, not FLA logic; gone on Triton ≥3.8, and `torch 2.13.0` pins `triton==3.7.1` |
| 6.3 | It is fixed in `flash-linear-attention==0.5.2` | **CONFIRMED** | [PR #953](https://github.com/fla-org/flash-linear-attention/pull/953) merged 2026-06-20, `Closes #945`, in `compare/v0.5.1...v0.5.2`; sdist diff on `fla/ops/common/chunk_delta_h.py`; `flash-linear-attention 0.5.2` pins `fla-core==0.5.2` | The commented-out pin is a correct *floor*. Caveats in §6b — notably the sibling backward kernel was never guarded |
| 6.4 | It reaches you "only via the classic FLA fast path, not via Hub `kernels`" | **CONTRADICTED** | the Hub repo `kernels-community/fla` *is* vendored FLA (`hub_kernels.py:197-210`) and its source already carries the fix; meanwhile the pip `fla` module is imported regardless of `use_kernels` (`hub_kernels.py:839-847`) | Rewrite the caveat as "never let `fla-core < 0.5.2` be importable". As pinned, no FLA is installed and nothing pulls it transitively — **the repo is not exposed**, so this risk closes rather than persists |
| 7.1 | Head counts are 24 query / 4 KV / 16 linear-key / 48 linear-value | **CONFIRMED** | `Qwen/Qwen3.8-27B/config.json` `text_config` | — |
| 7.2 | Those constrain valid TP to `{1,2,4,8}` | **CONFIRMED (conclusion)**, **CONTRADICTED (reason)** | see C2, C3 | Conclusion stands for an 8-GPU node; fix the "linear-key head count" explanation |
| A.1 | 64 layers, only 16 classic attention, 48 Gated DeltaNet | **CONFIRMED** | `config.json` `layer_types` — `full_attention` every 4th, `full_attention_interval: 4` | `sft_common`'s module-count reasoning is right |
| A.2 | `Qwen3_5ForCausalLM._keys_to_ignore_on_load_unexpected = [r"^mtp.*", r"^model.visual.*"]` | **CONFIRMED** | `modeling_qwen3_5.py:1584` (v5.16.1) | The text-only load strategy works as documented |
| A.3 | `_no_split_modules` still lists `Qwen3_5VisionBlock`, so auto-detection would over-wrap | **CONFIRMED** | `modeling_qwen3_5.py:803` `_no_split_modules = ["Qwen3_5DecoderLayer", "Qwen3_5VisionBlock"]`, inherited by `Qwen3_5ForCausalLM` | Setting `transformer_layer_cls_to_wrap` explicitly is justified |

---

## 1. Do the versions exist, when were they released, what do they declare?

All nine resolve with HTTP 200 from the PyPI JSON API and none of the
distributions are yanked.

| Package | Version | First upload (UTC) | `Requires-Python` |
|---|---|---|---|
| torch | 2.13.0 | 2026-07-08 | `>=3.10` |
| vllm | 0.28.0 | 2026-08-26 | `<3.15,>=3.10` |
| transformers | 5.16.1 | 2026-08-26 | `>=3.10.0` |
| peft | 0.20.0 | 2026-07-28 | `>=3.10.0` |
| datasets | 4.6.0 | 2026-02-25 | `>=3.10.0` |
| accelerate | 1.13.0 | 2026-03-04 | `>=3.10.0` |
| kernels | 0.16.0 | 2026-06-26 | `>=3.10` |
| matplotlib | 3.11.0 | 2026-06-12 | `>=3.11` |
| flash-linear-attention | 0.5.2 | 2026-07-27 | `>=3.10` |

Corroborating GitHub releases: `pytorch/pytorch@v2.13.0` published
2026-07-08T17:39:58Z; `vllm-project/vllm@v0.28.0` published 2026-08-26T09:46:30Z.
For context, `torch 2.14.0` shipped 2026-09-02 — the 2.13.0 pin is one minor
behind latest, which is exactly what vLLM 0.28.0 requires.

**`kernels>=0.16.0,<0.17`** resolves to **0.16.1**, the current latest; no 0.17
exists yet. That range is byte-identical to transformers 5.16.1's own
`setup.py:93` pin, which is the best possible justification for it.

**Declared core dependencies** (extras omitted) of each pin, and the shared
constraints they create:

- `vllm 0.28.0` → `torch==2.13.0`, `torchaudio==2.11.0`, `torchvision==0.28.0`,
  `transformers>=5.5.3`, `huggingface_hub>=1.27.0`, `tokenizers>=0.21.1`,
  `numba==0.65.0`, `flashinfer-python==0.6.16.post3`, `compressed-tensors==0.17.0`,
  plus ~60 more.
- `transformers 5.16.1` → `huggingface-hub<2.0,>=1.5.0`, `tokenizers<0.24.0,>=0.23.1`,
  `safetensors>=0.8.0`, `regex>=2025.10.22`, `typer`, `numpy>=1.17`. **No torch
  dependency** — torch is optional, so nothing here fights the `+cu130` pin.
- `peft 0.20.0` → `torch>=1.13.0`, `transformers` (unpinned), `accelerate>=0.21.0`,
  `huggingface-hub>=0.25.0`.
- `datasets 4.6.0` → `pyarrow>=21.0.0`, `fsspec[http]<=2026.2.0,>=2023.1.0`,
  `huggingface-hub<2.0,>=0.25.0`, `dill<0.4.1,>=0.3.0`, `multiprocess<0.70.19`.
- `accelerate 1.13.0` → `torch>=2.0.0`, `huggingface_hub>=0.21.0`,
  `safetensors>=0.4.3`.
- `kernels 0.16.0` → `huggingface-hub>=1.10.0`, `kernels-data>=0.14.0.dev1`,
  `sigstore<5,>=4`, `tomlkit>=0.13.3`.
- `matplotlib 3.11.0` → `numpy>=1.25`, `contourpy>=1.0.1`, `pillow>=9`, etc.
- `torch 2.13.0` → `cuda-toolkit[...]==13.0.3`, `nvidia-cudnn-cu13==9.20.0.48`,
  `nvidia-cusparselt-cu13==0.8.1`, `nvidia-nccl-cu13==2.29.7`,
  `nvidia-nvshmem-cu13==3.4.5`, `triton==3.7.1` (all Linux-gated).
- `flash-linear-attention 0.5.2` → `fla-core==0.5.2`, `transformers>=4.45.0`.

### Conflict scan

No unsatisfiable intersection:

| Shared dep | Tightest constraints | Latest available | Resolves to |
|---|---|---|---|
| `huggingface-hub` | `>=1.27.0` (vllm) ∩ `<2.0,>=1.5.0` (transformers) ∩ `<2.0` (datasets) | 1.30.0 | 1.30.0 ✅ |
| `tokenizers` | `<0.24.0,>=0.23.1` (transformers) ∩ `>=0.21.1` (vllm) | 0.23.2 | 0.23.2 ✅ |
| `numpy` | `<2.5,>=1.22` (vllm's `numba==0.65.0`) ∩ `>=1.25` (matplotlib) | 2.5.3 | a 2.4.x ✅ (a cap, not a conflict) |
| `fsspec` | `<=2026.2.0` (datasets) ∩ `>=0.8.5` (torch) | 2026.7.0 | 2026.2.0 ✅ (a downgrade, not a conflict) |
| `kernels-data` | `>=0.14.0.dev1` (kernels) | 0.16.1 | 0.16.1 ✅ (a final release exists; no pre-release opt-in needed) |

Two things `requirements.txt` does not mention but that the resolver will pull,
because vLLM hard-pins them:

- **`torchvision==0.28.0` and `torchaudio==2.11.0`.** Both exist on PyPI with
  cp312 manylinux wheels. Because `--extra-index-url` makes the cu130 index a
  peer of PyPI and PEP 440 sorts a local label *above* the bare public version,
  pip should prefer the `+cu130` builds from that index if they exist there —
  which is the outcome you want. Either way both indexes serve CUDA-13 builds
  for this release.
- **Every exotic CUDA pin in vLLM's `requirements/cuda.txt`** —
  `flashinfer-python==0.6.16.post3`, `tokenspeed-mla==0.1.8`,
  `humming-kernels==0.1.12`, `quack-kernels==0.6.4`, `apache-tvm-ffi==0.1.11`,
  `tilelang==0.1.12`, `nvidia-cutlass-dsl==4.6.2`, `PyNvVideoCodec==2.0.4`,
  `numba==0.65.0`, `compressed-tensors==0.17.0`, `depyf==0.20.0`,
  `outlines_core==0.2.14`, `lm-format-enforcer==0.11.3`. I checked each: **all
  present on PyPI with cp312-compatible wheels.** vLLM's published wheel
  deliberately excludes `flashinfer-cubin` from `install_requires` (it is not on
  PyPI since 0.6.14 — `requirements/cuda.txt:12-14`), so no extra index beyond
  the pytorch one is needed. This is the part that would have broken a naive
  `pip install vllm` and it does not.

**Verdict on installability: metadata-consistent, and the real answer arrives
from issue #2.** Nothing here can prove the ~3 GB of wheels actually build a
working environment on `/mnt/data` — only `setup.sh` can. What this section does
establish is that there is no *resolution* failure waiting, which is the failure
mode that would have wasted the most time.

---

## 2. Does vLLM 0.28.0 hard-pin torch, and does `+cu130` satisfy it?

**Yes to both.**

PyPI metadata for `vllm 0.28.0` contains a literal `Requires-Dist: torch==2.13.0`,
and the sdist's `requirements/cuda.txt:7` is the source of it:

```
# Dependencies for NVIDIA GPUs
torch==2.13.0
torchaudio==2.11.0
# These must be updated alongside torch
torchvision==0.28.0
```

On the local version: the PyPA version-specifier specification says

> If the specified version identifier is a public version identifier (no local
> version label), then the local version label of any candidate versions MUST be
> ignored when matching versions.

`==2.13.0` carries no local label, so `2.13.0+cu130` matches. The wheel the
cluster needs exists —
`torch-2.13.0+cu130-cp312-cp312-manylinux_2_28_x86_64.whl` is listed in
`https://download.pytorch.org/whl/cu130/torch/`.

vLLM's own release notes also describe its PyPI wheel as the CUDA 13.0 build
("PyPI (CUDA 13.0, uv) | `uv pip install vllm --torch-backend=auto`"), so vLLM
and torch agree on the CUDA major here.

See **C1** for what is wrong about the *reasoning* in `requirements.txt`.

---

## 3. FSDP2 in transformers 5.16.1

Everything `sft_common.fsdp_config()` depends on holds. Details in the verdict
table (3.1–3.7); the short version:

- `version` defaults to `2` — `training_args.py:2777`
  `fsdp_version = int(self.fsdp_config.get("version", 2))`. FSDP1 is not merely
  non-default: `training_args.py:2817-2818` warns it "is deprecated and will be
  removed in Transformers v5.20".
- `backward_prefetch`, `forward_prefetch`, `use_orig_params` appear **only** in
  the FSDP1 branch (`training_args.py:2823-2829`, after `if fsdp_version == 2:`
  at `:2810` and its `else:` at `:2814`). No validation, no unknown-key check, no
  warning — genuinely silent. Dropping them was the right call. This is a
  static-analysis conclusion; a two-GPU run with `backward_prefetch` set and logs
  captured would confirm it empirically, and issue #2's smoke run will do that
  for free.
- Every key `fsdp_config()` *does* set is read before the version split and is
  applied by accelerate under FSDP2 (`accelerate/utils/fsdp_utils.py:665`,
  `:744-748`, `:810`, `:830`).
- On `auto_wrap`: the documented no-op is the legacy **`fsdp="auto_wrap"`**
  string option (`training_args.py:2886-2887`, `elif item == FSDPOption.AUTO_WRAP: pass`),
  not an `fsdp_config` key, and it is version-agnostic rather than an FSDP2
  quirk. `auto_wrap_policy` — what the repo actually sets — is honoured and
  validated, raising `ValueError` on an unknown value (`:2784-2787`).
  `"TRANSFORMER_BASED_WRAP"` is in fact the default (`accelerate/utils/constants.py:39`),
  so setting it explicitly is harmless documentation.
- On `activation_checkpointing`: `docs/source/en/fsdp.md:78` says "Use this
  instead of gradient checkpointing in `TrainingArguments`. Setting both raises
  an error", and `trainer.py:843-849` enforces that with a `ValueError`. The
  AllGather rationale is not in the docs, but it is in transformers' own source,
  citing the same issue the repo cites:

  ```python
  # training_args.py:2747-2752
  if self.gradient_checkpointing:
      logger.warning(
          "When using FSDP, prefer `activation_checkpointing` in `fsdp_config` over "
          "`gradient_checkpointing`; the latter introduces a redundant AllGather in the backward pass. "
          "Reference: https://github.com/huggingface/transformers/issues/30404"
      )
  ```

- On the issue itself: [#30404](https://github.com/huggingface/transformers/issues/30404)
  is titled "[FSDP] redundant additional allgather during backward when using
  FSDP FULL_SHARD with gradient checkpointing", opened 2024-04-22 by
  `yundai424`, closed 2024-07-09. Its body reports "2 allgather ops per
  transformer block backward … resulting in 2 -> 3 collectives per backprop" and
  recommends `apply_activation_checkpointing` instead. **The citation in
  `sft_common.py` is accurate.**

Checked on the **exact pinned accelerate 1.13.0 sdist**, not just latest, since
transformers pins `accelerate>=1.1.0` with no upper bound:

- `utils/constants.py:39` — `FSDP_AUTO_WRAP_POLICY = ["TRANSFORMER_BASED_WRAP", "SIZE_BASED_WRAP", "NO_WRAP"]`,
  so the repo's value is valid.
- `accelerator.py:1692-1696` — under FSDP2, `set_auto_wrap_policy(model)` runs
  first ("Needs to be done first, to make sure AC + fully_shard will work as
  expected"), then `fsdp2_apply_ac(self, model)` if
  `activation_checkpointing` is set. `utils/fsdp_utils.py:604,641,689,749,771`
  builds and applies the transformer wrap policy. So `auto_wrap_policy` +
  `transformer_layer_cls_to_wrap` + `activation_checkpointing` are all live under
  FSDP2 in 1.13.0.
- And confirming what the repo dodged by dropping the FSDP1 knobs — had they
  reached accelerate 1.13.0, `dataclasses.py:1879` warns
  "backward_prefetch is not supported in FSDP2", `:1916` warns "use_orig_params
  is obsolete in FSDP2", and `:1931` **raises**
  `ValueError("forward_prefetch is not yet implemented in FSDP2, set to None or use `fsdp_version=1`")`.
  transformers shields you from that by never forwarding them.

---

## 4. `use_kernels`, `processing_class`, `dtype`

- `use_kernels=True` is a real `from_pretrained` keyword —
  `modeling_utils.py:4150` pops it, `:4396` calls `model.set_use_kernels(...)`,
  which requires the `kernels` package and then calls `kernelize(self, ...)`
  (`:3838-3878`). Documented in `docs/source/en/kernels.md:35`, though *not* in
  the `from_pretrained` docstring.
- The model family is real in 5.16.1: `src/transformers/models/qwen3_5/` exists,
  registered as `("qwen3_5", "Qwen3_5Config")`. Its linear-attention layer is
  the one carrying the layer-level hub decorator:

  ```python
  # modeling_qwen3_5.py:383-387
  @use_kernel_forward_from_hub("Qwen3_5GatedDeltaNet")
  @use_kernelized_func([torch_recurrent_gated_delta_rule, torch_chunk_gated_delta_rule,
                        causal_conv1d_fn, causal_conv1d_update])
  class Qwen3_5GatedDeltaNet(nn.Module):
  ```

  — and see **C4** for why the layer mapping does not fire on sm_103 while the
  function mappings do.
- One extra switch to know about: `hub_kernels.py:61-62` reads a `USE_HUB_KERNELS`
  env var (default `"YES"`). If it is ever set false in a Slurm environment,
  every kernel decorator silently becomes a no-op with a warning, and
  `use_kernels=True` buys nothing. Worth checking in the smoke-run log.
- `Trainer.__init__` in 5.16.1 (`trainer.py:368-387`) takes `processing_class`
  and has neither `tokenizer` nor `**kwargs`, so a leftover `tokenizer=` is a
  `TypeError` at construction, not a deprecation warning.
- `dtype=` is the documented keyword. `torch_dtype=` is still accepted by
  `from_pretrained` for backward compatibility and — notably — **without a
  warning** there (`modeling_utils.py:4130-4131`, `:4158-4160`); the warning
  lives on `_from_config` (`:1488-1493`) and the config property
  (`configuration_utils.py:461-474`). No removal version is stated anywhere. So
  `sft_common.load_model`'s comment ("`torch_dtype` is deprecated in transformers
  v5") is accurate; "removed" would not be.

---

## 5. `--language-model-only` in vLLM 0.28.0

**It exists. `serve.sbatch` does not need to drop it, and the README hedge can be
resolved.**

The chain, all in the 0.28.0 sdist:

1. `vllm/config/multimodal.py:101` — the field:
   ```python
   language_model_only: bool = False
   """If True, disables all multimodal inputs by setting all modality limits to 0.
   Equivalent to setting `--limit-mm-per-prompt` to 0 for every modality."""
   ```
2. `vllm/engine/arg_utils.py:1297` — the CLI registration:
   ```python
   multimodal_group.add_argument(
       "--language-model-only", **multimodal_kwargs["language_model_only"]
   )
   ```
3. `vllm/config/multimodal.py:492` — `get_limit_per_prompt()` returns `0` for
   every modality when set.
4. `vllm/model_executor/models/interfaces.py:296-338` — `_mark_tower_model`,
   whose docstring says "Tower model components are automatically skipped when
   `--limit-mm-per-prompt` is set to zero for all of their modalities", and which
   swaps in `StageMissingLayer` under `no_init_weights` when
   `all(mm_config.get_limit_per_prompt(m) == 0 for m in modalities)`.
5. `vllm/model_executor/models/qwen3_5.py:495-501` — the Qwen3.5 vision tower is
   constructed inside exactly that context:
   ```python
   with self._mark_tower_model(vllm_config, {"image", "video"}):
       self.visual = Qwen3_VisionTransformer(...)
   ```

And the docs name the family: `docs/models/supported_models.md:508` —

> For hybrid-only models such as Llama-4, Step3, Mistral-3 and Qwen-3.5, a
> text-only mode can be enabled by setting all supported multimodal modalities to
> 0 (`--language-model-only`) so that their multimodal modules will not be loaded
> to free up more GPU memory for KV cache.

That is the repo's comment, almost word for word.

**The nuance in C5.4 matters for what you'll observe.** `serve.sbatch`'s default
target is the *merged* model. `transformers.PreTrainedModel.save_pretrained` sets
`config.architectures = [model_to_save.__class__.__name__.removeprefix("FSDP")]`
(`modeling_utils.py:3418` at v5.16.1), so the merged checkpoint advertises
`Qwen3_5ForCausalLM`, which `vllm/model_executor/models/registry.py:203`
registers in the **text-only** table. `ModelConfig.__post_init__` only builds a
`multimodal_config` when `self._model_info.supports_multimodal`
(`vllm/config/model.py:752`), so for the merged model the flag's `InitVar` is
simply discarded — no error, no effect, and nothing to skip anyway since
`Qwen3_5ForCausalLM` never loaded `model.visual.*`. Serving the **base** model
for A/B (`serve.sbatch <base> 8001`) hits the multimodal path
(`architectures: ["Qwen3_5ForConditionalGeneration"]`, `vision_config.depth: 27`)
and there the flag does real work.

vLLM 0.28.0's release notes also list "Qwen3.5 fixes for text-only checkpoints
(#50734, #50355)", which is the exact path this repo takes.

While in the file: `--reasoning-parser qwen3` is registered
(`vllm/reasoning/__init__.py:15,127`) and `--default-chat-template-kwargs` exists
with JSON parsing (`vllm/entrypoints/openai/cli_args.py:93,167,200-201`), so the
other two flags in `serve.sbatch` are also real.

---

## 6. The sm_103 determinism bug

**The bug is real, it is genuinely worse on sm_103 than on sm_100, and it is
fixed in 0.5.2. But `requirements.txt`'s scoping of it is backwards, and the
practical answer is that this repo is not exposed — because it does not install
`flash-linear-attention` at all.**

### 6a. The bug — CONFIRMED

[fla-org/flash-linear-attention#945](https://github.com/fla-org/flash-linear-attention/issues/945):
"`chunk_gated_delta_rule_fwd_kernel_h_blockdim64` produces non-deterministic
output on NVIDIA B300 (sm_103) when autotune selects `num_warps=4`". Opened
2026-06-12, labelled `bug`, closed 2026-06-20. Reported against
`flash-linear-attention 0.5.0` on B300 SXM6, torch 2.10/2.12+cu130, triton
3.6.0/3.7.0. The report:

> returns **bitwise-different `h` / `v_new` on every invocation with
> bit-identical inputs** on NVIDIA B300 (sm_103) when compiled with
> `num_warps=4` and `num_stages=2` or `3`. The autotuner's config space for this
> kernel includes those configs, and on B300 the autotuner **selects one of the
> racy configs as best**

It carries a per-config matrix (25 identical calls, distinct bitwise hashes), a
stage bisection isolating divergence to that one kernel, and an
allocator-poisoning control ruling out uninitialised memory. Independently
corroborated in three other trackers:
[linkedin/Liger-Kernel#1255](https://github.com/linkedin/Liger-Kernel/issues/1255)
("it turns out the issue is upstream in FLA and Triton"),
[triton-lang/triton#10590](https://github.com/triton-lang/triton/issues/10590),
and [ByteDance-Seed/VeOmni#1101](https://github.com/ByteDance-Seed/VeOmni/pull/1101),
which pins `flash-linear-attention>=0.5.2,<0.6` with the comment
"FLA 0.5.2 fixes nondeterministic Blackwell GDN forward configs during
checkpoint recomputation."

**Root cause is a Triton miscompile, not FLA logic** — resolved on Triton
`main` / 3.8.0. The sm_100 sibling was
[triton#9871](https://github.com/triton-lang/triton/issues/9871) (closed
2026-04-20). What is sm_103-specific is that it survived into *released* Triton
with a wider envelope: after the sm_100 fix shipped in Triton 3.7.0 — the
version `torch 2.13.0` pins — `BV=64,w4,s2` and `BV=32,w8,s2/s3` still failed on
sm_103 and did not fail on sm_100 or H200/sm_90.

**The blast radius is larger than "nondeterministic loss".**
`fla/ops/gated_delta_rule/chunk.py:310` does not `save_for_backward` `h`/`v_new`;
the backward re-runs `chunk_gated_delta_rule_fwd_h` (`chunk.py:160`). On a racy
config the backward therefore recomputes a *different* state than the forward
produced, so gradients are inconsistent with the loss on every step — with or
without activation checkpointing. That is worse than irreproducibility; it is
silently wrong training.

### 6b. Fixed in 0.5.2 — CONFIRMED, with four caveats

[PR #953](https://github.com/fla-org/flash-linear-attention/pull/953) "[GDN]
Restrict Blackwell fwd h kernel to 2 warps", merged 2026-06-20 (after v0.5.1 on
2026-06-18, before v0.5.2 on 2026-07-27), `Closes #945`. The diff, taken from
the sdists:

```diff
--- fla-core-0.5.1/fla/ops/common/chunk_delta_h.py
+++ fla-core-0.5.2/fla/ops/common/chunk_delta_h.py
-        for num_warps in [2, 4]
+        for num_warps in GATED_DELTA_RULE_FWD_H_NUM_WARPS
```

with `GATED_DELTA_RULE_FWD_H_NUM_WARPS = [2] if IS_NVIDIA_BLACKWELL else [2, 4]`
and `IS_NVIDIA_BLACKWELL = IS_NVIDIA and torch.cuda.get_device_capability()[0] in (10, 12)`
(`fla/utils/_device.py:134`) — so sm_103 is covered.

Note **the kernel lives in `fla-core`, not `flash-linear-attention`**;
`flash_linear_attention-0.5.2/pyproject.toml:12` pins `fla-core==0.5.2` exactly,
so the commented-out pin in `requirements.txt` would in fact pull the fixed
kernel. Caveats:

1. It is a **config-space workaround**, not a fix. The code comment says as much:
   "Keep this kernel on `num_warps=2` for Blackwell until Triton 3.8 is released
   and we re-validate the wider config space." It also costs ~10% on sm_100 and
   sm_120, which are gated the same way.
2. **The sibling backward kernel was never guarded.**
   `chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64` still autotunes over
   `[2, 4]` in 0.5.2 and on `main`. Issue #945 asked for that audit; nobody swept
   it per-config. Whether it is racy on sm_103 is **UNVERIFIABLE** from source.
3. **The v0.5.2 release notes never say "determinism".** Grepping the release
   body for `determinis|sm_103|b300` matches only PR titles. Anyone triaging by
   release notes would miss this.
4. `FLA_CACHE_MODE` (default `DISABLED`) can load a cached autotune config that
   bypasses the guard. No config files ship in the sdist, so this only bites
   someone who opts in with a stale cache.

### 6c. Which path it reaches you by — the repo's scoping is CONTRADICTED

`requirements.txt` says the bug is "only relevant if you install the classic
fast path instead of using Hub kernels". Three things are wrong with that:

1. **The Hub kernel *is* FLA.** `hub_kernels.py:197-210` maps
   `chunk_gated_delta_rule` to `repo_id="kernels-community/fla"` — vendored
   flash-linear-attention Triton source, not an independent implementation.
   "Hub kernels instead of FLA" is not a real distinction.
2. **The Hub copy already contains the fix.** The kernel-builder source in
   `huggingface/kernels-community` at `fla/torch-ext/fla/ops/common/chunk_delta_h.py`
   carries `GATED_DELTA_RULE_FWD_H_NUM_WARPS = [2] if IS_NVIDIA_BLACKWELL else [2, 4]`
   with the same `capability[0] in (10, 12)` test. Its snapshot post-dates the
   2026-06-20 fix. So the Hub path is the *safe* one, not the exposed one.
3. **The pip `fla` module is used even without `use_kernels=True`.**
   `use_kernel_func_from_hub_with_fallback` (`hub_kernels.py:822-865`) documents
   the order as "1. Hf kernels (if requested) 2. Original package 3. Torch only
   path", and tier 2 is resolved **at import time**, unconditionally:

   ```python
   module = importlib.import_module(package)          # package == "fla"
   implementation = resolve_internal_import(module, full_path)
   ```

   with `full_path == "ops.gated_delta_rule.chunk_gated_delta_rule"`
   (`_KERNELS_INTERNAL_PATH_MAPPINGS`, `hub_kernels.py:66-76`). So an importable
   `fla` displaces the torch path whether or not you asked for Hub kernels.

**The correct rule is therefore the inverse of the comment.** It is not "install
FLA and you might hit the bug, use Hub kernels and you won't". It is: **never let
`fla-core < 0.5.2` be importable in this venv.** As written, `requirements.txt`
installs no FLA at all, and nothing in the pinned stack declares `fla-core` as a
dependency (I checked every `Requires-Dist` in §1), so **this repo is not exposed
today**. The commented-out `flash-linear-attention==0.5.2` is a correct floor if
anyone uncomments it; what would be dangerous is any older FLA arriving
transitively.

The remedy line in `requirements.txt` — "Uncomment if you hit nondeterministic
loss or activation-checkpoint recompute failures" — reads as though installing
FLA is the cure. With no FLA installed the delta-rule path is either the patched
Hub kernel or pure torch; in neither case does installing FLA fix
nondeterminism. It is a floor to respect, not a lever to pull.

### 6d. What could not be verified

- **The published `kernels-community/fla` build artifact.** It returns HTTP 401
  anonymously, and so does a name I invented
  (`kernels-community/nonexistent-xyz`), so 401 does not distinguish "gated"
  from "absent". It is also missing from the org's public 61-repo listing, while
  `kernels-community/mamba-ssm` and `kernels-community/causal-conv1d` return 200
  and *are* listed. Everything above about the Hub copy comes from the public
  kernel-builder *source*, not from the published build. Reportedly the build is
  `[torch-noarch]` — pure-Python Triton, JIT-compiled on the user's GPU — so
  there are no per-SM build variants and no sm_103 variant to be missing. The
  same 401 applies to `Atlas-Inference/gdn`, which transformers loads with
  `trust_remote_code=True` and a "TODO: drop once Atlas-Inference is an
  allow-listed trusted publisher" comment.
  **What would settle it:** the first `use_kernels=True` load on the cluster,
  which has HF egress. Its kernel-resolution log line says whether the Hub fetch
  succeeded, needed a token, or fell through.
- **Whether the Hub `fla` kernel supports training at all.** transformers
  PR #48185 is reported to say its backward raises `ModuleNotFoundError`, making
  it inference-only. I could not read that PR to confirm. If true, a training run
  either errors or falls through to tier 2/3 — which changes which
  implementation the LoRA run actually uses. Same log line settles it.
- **Whether the bug reproduces on *this* B300 with torch 2.13.0 / triton 3.7.1.**
  Nothing short of running it settles this. Issue #945's repro is 30 identical
  calls, counting distinct bit-hashes (expect 1; B300 gave 30). Note the fix is
  moot on Triton ≥3.8 where the root cause is gone — and torch 2.13.0 pins
  `triton==3.7.1`, so this stack is on the affected Triton.
- **Whether the unguarded backward kernel is racy on sm_103.** Never swept by
  anyone.

---

## 7. The tensor-parallel constraint

See **C2** and **C3**. Summary: the head counts in the README are correct
(verified against `Qwen/Qwen3.8-27B/config.json`), the conclusion
`TP ∈ {1,2,4,8}` is correct for a single 8-GPU B300 node, and the stated reason
for TP=16 failing is wrong — it fails on the 24 query heads
(`vllm/model_executor/models/qwen3_next.py:242`), not on the 16 linear-key heads,
which divide 16 exactly.

This is still an inference from reading assertions, not a test. It is a *strong*
inference — `divide()` is a bare `assert` that fires during model construction,
so a bad TP fails in seconds, before any weights load — but the map already
notes that whether TP>1 is worth testing at all is unsettled. If someone does
test it, TP=3 is the interesting case: it satisfies every head-count check the
README reasons about and still fails.

---

## What only the cluster can settle

Deliberately not guessed here, and arriving anyway from issue #2 and the smoke
runs:

1. **Whether the ~3 GB install actually works.** Metadata consistency is proven;
   wheel-level ABI compatibility across `torch+cu130` / `torchvision` /
   `flashinfer` / `triton 3.7.1` is not. `setup.sh` step 5 is the test.
2. **Whether torch 2.13.0+cu130 runs on sm_103.** PyTorch's `DEVICE_REQUIREMENT`
   table and NVIDIA's minor-revision rule both say sm_100 SASS covers cc 10.3,
   and no published 2.13.0 wheel has sm_103 SASS, so that rule is the only thing
   standing between this repo and a "no kernel image is available" error.
   `setup.sh`'s bf16 matmul check answers it in one 5-minute job. Watch for a
   `Found GPU0 ... compute capability (CC) 10.3` warning in that log — its
   presence or absence is the whole answer.
3. **Whether `use_kernels=True` resolves anything on sm_103.**
   `kernels-community/fla` is not readable anonymously (§6d) and
   `kernels-community/mamba-ssm` is, so the two halves of the GDN fast path may
   not resolve the same way. The first kernel-resolution line in the training log
   settles it — including whether the Hub `fla` kernel supports the backward pass
   at all, or whether the run silently falls through to the pure-torch path.
4. **Determinism.** See §6 — no amount of reading settles bitwise reproducibility
   on this hardware. Two identical 25-step smoke runs with the same seed, diffed
   on loss, does. Worth doing regardless of §6's "not exposed" verdict, because
   the smoke run is cheap and the failure mode (gradients disagreeing with the
   loss) is invisible in a loss curve that merely looks plausible.
5. **TP>1.** Untested by anyone here. Cheap to test if wanted (fails at
   construction), out of scope per the map.
