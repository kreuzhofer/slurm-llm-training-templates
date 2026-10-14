# LTX-2.5 on B300: weights, hardware, limits, consistency

Research notes and measurements from 2026-10-14, made while building the
[`models/ltx-2.5`](../models/ltx-2.5/README.md) template (commit `398f198`).

- Model: [Lightricks/LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5), a 22B
  audio-video diffusion transformer.
- Code: [Lightricks/LTX-2](https://github.com/Lightricks/LTX-2) at tag
  `v1.4.2`. Local paths below are relative to that checkout
  (`$TEMPLATES_DIR/src/LTX-2`).
- Cluster: 2 × 8 NVIDIA B300 SXM6 (compute capability 10.3), driver 580.159.04
  (CUDA 13.0), x86_64. torch 2.13.0+cu130, transformers 5.14.1, natten
  0.21.7, ltx-kernels built for sm_103.

Unless noted, a measured run used the distilled pipeline at 1536×1024 and
24 fps. Job IDs refer to Slurm jobs on this cluster. Their logs and videos have
since been deleted; the numbers below were collected before that.

## 1. Available weights

### Lightricks/LTX-2.5

The repo is 201 GB in total. Each pipeline needs only some of the files
([model card](https://huggingface.co/Lightricks/LTX-2.5); sizes from the
Hugging Face API).

| component | file | size | used by |
|---|---|---|---|
| transformer, distilled | `diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors` | 42.02 GB | distilled, DFR, IC-LoRA, Dub-It pipelines |
| transformer, dev (full) | `diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors` | 42.02 GB | guided two-stage, keyframe interpolation, audio-to-video, training |
| transformer, distilled NVFP4 | `diffusion_models/ltx-2.5-22b-distilled-transformer-nvfp4.safetensors` | 18.72 GB | Blackwell; ComfyUI or `--quantization nvfp4-prequant` |
| transformer, int8 | `…-dev-…-comfy-int8-convrot`, `…-distilled-…-comfy-int8-convrot` | 21.50 GB each | ComfyUI only |
| text encoder (Gemma 4 12B + projection) | `text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors` | 26.26 GB | every pipeline |
| text encoder, int8 | `text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors` | 15.37 GB | ComfyUI only |
| video VAE, diffusion decoder | `vae/ltx-2.5-video-vae-bf16.safetensors` | 1.47 GB | better quality, slower |
| video VAE, convolutional | `vae/ltx-2.5-video-vae-conv-bf16.safetensors` | 1.45 GB | faster, lighter |
| audio VAE + vocoder | `vae/ltx-2.5-audio-vae-bf16.safetensors` | 0.36 GB | sound |
| spatial upscaler ×2 | `latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors` | 1.00 GB | all two-stage pipelines |
| temporal upscaler ×2 | `latent_upscale_models/ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors` | 0.26 GB | DFR `--temporal-upscalings` |
| distilled LoRA | `loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors` | 8.90 GB | dev-model two-stage pipelines (TI2Vid, Keyframe, A2Vid) |
| duration head | `model_patches/ltx-2.5-duration-head-bf16.safetensors` | 4 MB | auto duration |

The template downloads the distilled transformer, the bf16 text encoder, both
bf16 VAEs, both upscalers, the duration head and the DFR detailing IC-LoRA:
71 GB in total (67 GiB on disk). Job 1261 downloaded it in about 5 minutes.

The text encoder is a Gemma 4 12B fine-tuned for LTX. Stock Gemma 4 does not
work: loading checks for `gemma4-12b-ltx-v1` (`README.md`, Full Model List).
The int8 and NVFP4 files are listed as ComfyUI or Blackwell options on the
[model card](https://huggingface.co/Lightricks/LTX-2.5).

### Other LTX-2.5 repos

Sizes from the Hugging Face API, 2026-10-14. All are gated with automatic
approval.

| repo | size | what it is |
|---|---|---|
| `Lightricks/LTX-2.5-Pre-Trained` | 70.1 GB | pretrained checkpoint for deep fine-tuning |
| `Lightricks/LTX-2.5-Diffusers` | 174.4 GB | Diffusers format, includes a prompt enhancer |
| `…-22b-IC-LoRA-Ingredients` | 1.4 GB | reference sheet → consistent characters, props, location (section 5) |
| `…-22b-IC-LoRA-Pixel-Spatial-Upscaler` | 0.3 GB | detailing pass for the DFR pipeline |
| `…-22b-IC-LoRA-Refine-Details` | 1.3 GB | |
| `…-22b-IC-LoRA-Restore` | 1.7 GB | |
| `…-22b-IC-LoRA-Deblur` | 0.9 GB | |
| `…-22b-IC-LoRA-Decompression` | 0.9 GB | |
| `…-22b-IC-LoRA-Colorization` | 0.9 GB | |
| `…-22b-IC-LoRA-SDR-To-HDR` | 1.3 GB | |
| `…-22b-IC-LoRA-Day-To-Night` | 0.3 GB | |
| `…-22b-IC-LoRA-Clean-Plate` | 0.3 GB | |
| `…-22b-IC-LoRA-Alpha-Gen` | 1.3 GB | |
| `…-22b-IC-LoRA-Layout-To-Render` | 1.3 GB | 3D viewport or blocking animation → finished shot ([card](https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Layout-To-Render)) |
| `…-22b-IC-LoRA-Water-Simulation` | 1.0 GB | |
| `…-22b-LoRA-Cinemagraph` | 0.2 GB | |
| `…-22b-LoRA-Slow-Motion-Control` | 0.7 GB | |

Only the two IC-LoRAs with a description were read in detail. The others are
listed by name only.

### License and access

Every repo above is gated. The Hugging Face account behind the token must
accept the license on each repo's page; until then a download fails with
"Access denied. This repository requires approval."

Weights and code are under the
[LTX-2.x Community License](https://github.com/Lightricks/LTX-2/blob/main/LICENSE-2_x)
(`LICENSE-2_x`):

- Section 2.1: entities with annual revenue of at least $10,000,000 need a paid
  Commercial Use Agreement.
- Section 2.2: such an entity may use LTX-2.x without one only for a
  Non-Commercial Purpose. That includes "testing, evaluation, or
  non-commercial research and development in a non-production or development
  environment". Revenue-generating use, use that affects end users, and
  training models for commercial use are excluded.

## 2. Hardware requirements

### What LTX documents

- **No minimum GPU memory for inference is published.** Peak memory follows
  from the order of loading. The pipeline runs the text encoder (26 GB) first,
  then frees GPU memory between stages by default
  (`packages/ltx-pipelines/docs/optimization.md`, Memory Cleanup Between
  Stages). It then loads the transformer (42 GB in bf16) and the decoders.
  Working memory comes on top and grows with resolution × length. Peak memory
  was not measured on B300.
- **Quantization** (`docs/optimization.md`, FP8 Quantization):
  - `fp8-cast`: about 21 GB of transformer weights, any GPU with FP8
  - `fp8-scaled-mm`: Hopper and newer
  - `nvfp4-cast` and `nvfp4-prequant`: Blackwell, needs the `nvfp4_cpp`
    extension from ltx-kernels
- **NVFP4 kernel targets:** ltx-kernels builds them for `100a`, `110a` and
  `120a` only (`packages/ltx-kernels/setup.py:191`). CUDA's `a` targets run
  only on exactly that compute capability:
  - RTX 5090 (12.0) is covered.
  - B300 (10.3) and DGX Spark (12.1) are not.
  - This was not tested on any of the three.
- **Offloading:** `--offload cpu` or `--offload disk` streams weights from host
  memory or disk. The memory figures in its docstring (`~36 GB RAM + ~5 GB VRAM`)
  are written for LTX-2's ~28 GB model, not for the 22B LTX-2.5
  (`packages/ltx-pipelines/src/ltx_pipelines/utils/types.py:188-198`).
- **Training (ltx-trainer):**
  - "80GB+ VRAM" recommended.
  - 32 GB GPUs, "e.g., RTX 5090", use the low-VRAM config with INT8
    quantization (`packages/ltx-trainer/README.md:44-45`).
  - LTX's own training skill puts the minimum at 32 GB
    (`.claude/skills/train-model/references/hardware-profiles.md:9`).
- **Multi-GPU is single-node only.** The `*_mgpu` runners hold a full
  transformer replica on every GPU, so they cut latency but do not save memory
  (`packages/ltx-pipelines/docs/multigpu/README.md:7-25`).

### B300 against the home machines

"RTX Spark" is read as NVIDIA DGX Spark (GB10).

Sources for the vendor specs:
[DGX Spark](https://www.nvidia.com/en-us/products/workstations/dgx-spark/),
[RTX 5090](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/),
[RTX 50 announcement](https://www.nvidia.com/en-us/geforce/news/rtx-50-series-graphics-cards-gpu-laptop-announcements/)
(5090 bandwidth),
[HGX](https://www.nvidia.com/en-us/data-center/hgx/),
[GB300 NVL72](https://www.nvidia.com/en-us/data-center/gb300-nvl72/),
[CUDA GPUs](https://developer.nvidia.com/cuda-gpus) (compute capability).

| | B300 SXM6 (this cluster) | RTX 5090 | DGX Spark |
|---|---|---|---|
| GPU memory | 275,040 MiB (268.6 GiB) per GPU, from `nvidia-smi`; HGX B300 lists 2.1 TB per 8 GPUs | 32 GB GDDR7 | 128 GB LPDDR5x unified with the CPU (a 64 GB model also exists) |
| memory bandwidth | 8 TB/s per GPU, derived (GB300 NVL72: 576 TB/s ÷ 72 GPUs) | 1792 GB/s | 273 GB/s |
| compute capability | 10.3 (Blackwell Ultra) | 12.0 (Blackwell) | 12.1 (Blackwell) |
| published FP4 peak | 18 PFLOPS sparse per GPU, derived (HGX B300: 144 PFLOPS ÷ 8) | 3352 AI TOPS | up to 1 PFLOP FP4 |
| CPU | x86_64 | host-dependent, typically x86_64 | 20-core Arm (10 Cortex-X925 + 10 Cortex-A725) |
| bf16 weights fit (42 GB transformer, 26 GB text encoder, loaded one after the other) | yes, measured | no; needs NVFP4 (18.7 GB), fp8-cast (~21 GB), ComfyUI int8 or offloading | yes on the 128 GB model, with no offloading (not tested) |
| NVFP4 via ltx-kernels | no (10.3 not in the target list) | yes (`120a`), not tested | no (12.1 not in the target list) |
| one 5 s clip, whole job | measured 162–216 s on 1 GPU with up to 8 jobs per node; 306–329 s on 8 GPUs | not measured | not measured |
| 20 s / 30 s clips | measured on 1 GPU: 20 s in 269 s, 30 s in 325 s | not tested; working memory on top of quantized weights may need offloading | not tested; memory suffices on the 128 GB model |
| LoRA training tier (ltx-trainer) | 80 GB+ config | 32 GB low-VRAM config, named by LTX | 80 GB+ config fits in memory on the 128 GB model; slow (estimate) |
| software for this template | as pinned in `models/ltx-2.5/requirements.txt` | x86 cu130 pins may carry over, untested; ComfyUI is the documented consumer path | needs aarch64 builds of torch, natten and ltx-kernels; ltx-core's natten pin includes aarch64; untested |

How to read the speed rows: per-clip speed at home is an estimate, not a
measurement. On published FP4 peaks the B300 has about 5× the compute of an
RTX 5090 and about 18× that of a DGX Spark. Its memory bandwidth is about 4.5×
and about 29× theirs. A whole 1-GPU job on B300 takes 3–3.5 minutes for 5 s;
how much of that is loading versus generation was not measured. If generation
scales with these ratios, expect several to tens of minutes per 5 s clip at
home.

## 3. Measured results on B300

### Time per job

Job time is measured from job start to "Done", and includes about 2 minutes of
Python start-up and model loading from NFS. On 8 GPUs the breakdown comes from
the MGPU log timestamps; on 1 GPU there is no breakdown.

| layout | clip | jobs | time per job |
|---|---|---|---|
| 8 GPUs (`distilled_mgpu`, FP8 cast) | 5 s, 121 frames | 1265, 1266, 1293–1299 | 306–329 s: Python start ~2 min, worker load 86–103 s, generation + decode + encode 91–119 s |
| 8 GPUs, `--chunked` | 30 s, 721 frames | 1304 | 365 s (load 100 s, rest 143 s) |
| 1 GPU (bf16) | 5 s | 1322, 1329–1336 | 162–216 s |
| 1 GPU | 5 s, 1920×1088 | 1328 | 189 s |
| 1 GPU, DFR | 5 s | 1327 | 210 s |
| 1 GPU | 10 s, 241 frames | 1323, 1324 | 224 s |
| 1 GPU | 20 s, 481 frames | 1320 | 269 s |
| 1 GPU | 30 s, 721 frames | 1321 | 325 s |
| 1 GPU | 20 s vertical, 1088×1920 | 1339, 1340 | 310 s, 306 s |
| 1 GPU, `NUM_FRAMES=auto` | the model chose 233 frames (9.71 s) | 1325 | 216 s |

The 1-GPU jobs ran up to eight at a time on one node (1320–1327 on
`worker-b300-0`, 1328–1336 on `worker-b300-1`).

The 8-GPU runner was not faster per job than one GPU, because eight workers
each load the full weights. It also failed for 10 s at 1536×1024 (job 1300; see
section 4).

Every output was H.264 with AAC stereo audio at 48 kHz. A 5 s clip at
1536×1024 is about 2.1 MB.

### Speech: German against English

`models/ltx-2.5/check_speech.py` transcribes the audio with Whisper large-v3,
detects the language, and computes the word error rate (WER) against the quoted
dialogue in the prompt. Whisper writes numbers as digits, so "vier" against "4"
counts as one error. Every clip was detected as the intended language.

| prompt | seed 42: EN | seed 42: DE | seed 7: EN | seed 7: DE |
|---|---|---|---|---|
| bakery, 5 s | 0.05 (1265, digit) | 0.06 (1266, digit) | 0.00 (1329) | 0.06 (1330, digit) |
| news anchor, 5 s | 0.00 (1293) | 0.00 (1294) | 0.00 (1331) | 0.00 (1332) |
| tram announcement, 5 s | 0.00 (1295) | 0.10 (1296) | 0.00 (1333) | 0.10 (1334) |
| two speakers, kitchen, 5 s | 0.00 (1297) | 0.00 (1298) | 0.00 (1335) | 0.00 (1336) |

- **The one real German error:** in both tram takes (1296, 1334), Whisper heard
  "Hauptbindhof" for "Hauptbahnhof". The rest of the line, including
  "Straßenbahnhaltestelle", came out correctly.
- **The German news line came out word for word in both seeds:** "Guten Abend.
  Der Bundestag hat heute ein Gesetz beschlossen, das Bahnfahren für
  Studierende günstiger macht."
- **Prompt written entirely in German** (1299): the same result as the
  English-written prompt with German dialogue, 0.06 from the digit.
- **Two speakers:** in 1297 and 1298 (seed 42), eight frames per clip show the
  man asking and the woman answering, as prompted. The seed-7 pair was checked
  for words only.
- **Long monologues:**
  - 10 s: EN 0.00 (1323), DE 0.00 (1324)
  - 20 s, German, 52 words: 0.00 (1320)
  - 30 s, German, 79 words: 0.01 (1321). The only error is a trailing "you",
    which Whisper hallucinated on the closing silence.
- **German multishot**, prompted with three shots, 5 s (1326): 0.20. The only
  error is "zwanzig" written as "20". The cuts themselves were not checked.
- **DFR, German** (1327): 0.06, from the digit.
- **Same prompt and seed on two layouts** (1266 on 8 GPUs, 1322 on 1 GPU):
  identical words, and frames that are nearly identical.

### Picture over long clips

The frame-to-frame difference (mean absolute difference on downsampled
grayscale frames) gives these jumps:

- **1321** (30 s, one pass): steady until about 28.9 s, then the largest jumps
  of the clip. Her head drops and the framing changes in the last second.
- **1304** (30 s, `--chunked`): jumps at 4.0 s (the end of the first 97-frame
  window) and at 23.8 s, with visible changes of pose and framing.

## 4. Limits

- **Length.**
  - LTX states "up to 20 seconds per generation" ([ltx.io/llm-info](https://ltx.io/llm-info)).
  - The LTX-2 paper (§6.3) states "up to 20 seconds of continuous video".
  - §7 Limitations: "generating coherent audiovisual sequences longer than
    roughly 20 seconds" "can lead to temporal drift, degraded synchronization,
    or reduced scene diversity"
    ([arXiv 2601.03233](https://arxiv.org/html/2601.03233v1)).
  - The open-source code has no hard cap. `--num-frames` must be 8k+1, and
    width and height must be divisible by 64 for the two-stage pipelines.
  - Observed: 20 s clean; 30 s with all words correct and drift in the last
    second (1320, 1321).
- **Auto duration** clamps to 1–20 s (`utils/types.py:175-185`). For a one-line
  bakery prompt that is spoken in about 5 s it chose 9.71 s (1325).
- **Languages.**
  - The model card tags nine: en, de, es, fr, ja, ko, zh, it, pt (Hugging Face
    API tags).
  - LTX's prompt guide lists "Languages currently validated: English, French,
    Spanish, German, Russian". That list is for the Dub-It IC-LoRA, not for
    text-to-video ([prompt guide](https://ltx.io/blog/ltx-2-5-prompt-guide)).
  - The paper warns that languages or dialects underrepresented in training
    "may yield less accurate speech synthesis or weaker audio–visual
    alignment".
  - German measured well (section 3).
- **Multi-GPU token cap.**
  - `distilled_mgpu` and `ti2vid_two_stages_mgpu` preallocate their
    sequence-parallel buffers for 32768 video tokens
    (`distilled_mgpu.py:44`, `ti2vid_two_stages_mgpu.py:46`); `dfr_mgpu` uses
    524288 (`dfr_mgpu.py:52`). The CLI cannot raise it.
  - Video tokens = ((frames − 1)/8 + 1) × (H/32) × (W/32)
    (`docs/optimization.md`). The cap is therefore 161 frames (6.7 s) at
    1536×1024 and 121 frames (5 s) at 1920×1088.
  - Job 1300 (241 frames) failed with "Total video token count (47616) exceeds
    attention_manager max_tokens (32768)".
  - The MGPU runners also always cast the transformer to FP8
    (`distilled_mgpu.py:65`, `dfr_mgpu.py:149`).
- **`--chunked` is not for speech.** It renders 97-frame windows with 25 frames
  of overlap (`docs/installation.md:104`). Every window re-reads the whole
  prompt, so the speaker restarts the dialogue. Job 1304 (30 s) reached WER
  0.85: "Mein Großvater hat mir beigebracht … Mein Großvater hat mir beidmisch
  geholter …".
- **Several speakers.** The paper: "in multi-speaker scenarios, the model may
  inconsistently assign spoken content to characters". Our two
  two-speaker clips that were checked were assigned correctly.
- **On-screen text.** "exact spelling and consistency across frames are not
  guaranteed" ([prompt guide](https://ltx.io/blog/ltx-2-5-prompt-guide)).
- **Multishot.** The prompt guide recommends 2–4 shots per generation, each cut
  named in prose ("A hard cut transitions to …").

## 5. Consistency across a long series

The planned use is about 100 episodes of 1–2 minutes with recurring
characters, locations, dialogue and voiceover. Each generation is at most about
20 s, so consistency has to come from inputs that every shot reuses.

| tool | what it carries | status |
|---|---|---|
| fixed description block per character and location | appearance through text only; weakest | prompting practice from the [prompt guide](https://ltx.io/blog/ltx-2-5-prompt-guide), not tested here |
| start frame `--image PATH FRAME_IDX STRENGTH` (template: `IMAGE=`) | look, set and light at the conditioned frame, drifting over the clip | supported by every pipeline (`docs/conditioning.md`); not tested here |
| keyframe interpolation (`keyframe_interpolation`) | fixed images at several points of a shot | needs the dev model (`README.md`, Full Model List); not tested |
| native multishot | identity, set and voice across 2–4 cuts within one generation | 1326: prompted with three shots, line correct; cuts and identity across them not checked |
| character LoRA via ltx-trainer | identity learned from the character's clips; text-to-video training denoises video and audio together, so the voice is learned too (`packages/ltx-trainer/docs/training-modes.md:48`) | not tested; needs the dev model and 32 GB+ GPUs |
| IC-LoRA Ingredients | a reference sheet (one composite image: per character a face close-up and turnaround, props, one location panel), looped as a static video; prompt split into "Reference sheet:" and "Generated video:" | from the [model card](https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Ingredients): base model LTX-2.5-22B dev, trained at 768×448, 121 frames, 24 fps only; not tested |
| audio-to-video (`a2vid_two_stage --audio-path`) | a fixed voice: the picture is generated to a given audio track, e.g. from per-character TTS | needs the dev model plus the distilled LoRA (`README.md:120`); not tested |
| Dub-It (`dubit --reference-video`) | the speaker's identity while changing the words | the documented IC-LoRA is `Lightricks/LTX-2.3-22b-IC-LoRA-DubIt` (`docs/pipelines.md:186`); whether it works on 2.5 is unverified |
| retake (`retake`) | regenerates one time range of an existing clip and keeps the rest | not tested |
| extend | continues a clip | an API endpoint (`ltx-2-3-pro`) per [LTX's guide](https://ltx.io/blog/how-to-make-longer-ai-videos); v1.4.2 has no local `extend` module; locally the nearest are `--chunked` (not for speech), retake and keyframe interpolation |

The pilot proposed is two characters, one location and one scene of about
60 s. It would compare prompt-only, start frames, character LoRAs, the
Ingredients IC-LoRA and audio-first generation on face similarity and voice
consistency across about 10 shots.

## 6. Seeds

- The seed seeds a `torch.Generator`, which draws the initial video and audio
  noise (`packages/ltx-pipelines/src/ltx_pipelines/distilled.py:266`).
- LTX-2.5 checkpoints sample with Euler ancestral (`eta=1.0`)
  (`CHANGELOG.md:43`, `docs/pipelines.md:53`). That sampler adds fresh noise at
  every step, from a generator seeded with `seed + offset + call_index`
  (`distilled.py:95-118`).
- **Same prompt, seed, size, frame count and pipeline give the same clip.**
  1266 (8 GPUs, FP8 cast) and 1322 (1 GPU, bf16) used the same prompt and seed
  42. Their frames are nearly identical; the small differences in light and
  background come from the different precision.
- **A different seed is a different take:** different gestures, timing and
  voice, often a different face. WER for seeds 42 and 7 was the same in every
  pair except where Whisper wrote a number as a digit in one take and as a word
  in the other (section 3).
- **What changes the result even with the same seed:**
  - the prompt text
  - resolution or frame count, which change the shape of the noise
  - pipeline or precision
- **Use seeds to:** render several takes of one shot and pick one; reproduce a
  chosen take (the job log records the seed); hold everything constant while
  tuning prompt wording.
- **Seeds do not carry identity into a different prompt.**

## 7. Setup defects found

| defect | symptom | fix in the template |
|---|---|---|
| pip pairs `nvidia-cuda-nvcc==13.0.88` with unpinned `nvidia-cuda-crt` / `nvidia-nvvm` 13.4 | ltx-kernels build: "ptxas fatal: Unsupported .version 9.4; current version is '9.0'" | pin crt and nvvm 13.0.88 (`models/ltx-2.5/requirements.txt`) |
| no `ninja` in the venv | torch builds the extensions with the slow serial backend | `ninja==1.13.0` |
| system nvcc is 12.9, torch is cu130 | torch refuses a mismatched nvcc | nvcc 13.0 from pip; `CUDA_HOME` left unset so `setup.py` finds it |
| `transformers>=5.15` breaks the Gemma 4 encoder (`packages/ltx-core/pyproject.toml:22`) | the shared venv has 5.16.1 for vLLM | separate `venv-ltx`; `cluster_setup venv-ltx` |
| `DefMemPerNode=UNLIMITED` | a 1-GPU job without `--mem` took all 2487G and blocked the node's other 7 GPUs | `--mem=250G`, held for 30 s clips and DFR |
| prompt file under the login node's `/tmp` | the compute node could not see it and rendered the path string as the prompt (job 1268) | `generate.sbatch` refuses a single word that is not a file |
| Xid 13 (08:03) and Xid 79 (09:06) on GPU 0 of `worker-b300-0` | the Soperator health check drained the node (`on_fail: drain`, `on_ok: none`, so it never resumes on its own) | `sudo scontrol reboot ASAP nextstate=RESUME`; the node was back after 29 minutes with a clean kernel log |

The two cuDNN builds present (pip 9.20.0.48 and the system's 9.10.1) both lack
`libcudnn_engines_tensor_ir`, so the mismatch LTX warns about for cu132 does not
arise here. The setup check ran cuDNN and Flash attention successfully on
sm_103.
