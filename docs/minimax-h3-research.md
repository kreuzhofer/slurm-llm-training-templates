# MiniMax H3 on B300: weights, job times, limits, references, scenes

Research notes and measurements from 2026-10-14 and 2026-10-15, made while
building the [`models/minimax-h3`](../models/minimax-h3/README.md) template.

- Model: [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) at
  revision `42ed227`. Its transformer is "a 33B-parameter dense, single-stream
  Transformer" (model card, H3-Omni-Transformer).
- Code: diffusers 0.41.0, modular pipeline. Local paths below are relative to
  `diffusers/modular_pipelines/minimax_h3/` in `$TEMPLATES_DIR/venv-h3`.
- Cluster: 2 × 8 NVIDIA B300 SXM6 (compute capability 10.3), driver 580.159.04
  (CUDA 13.0), x86_64. torch 2.13.0+cu130, transformers 5.18.0.

Unless noted, a run used one GPU, bf16, 50 steps and seed 42. Job IDs refer to
Slurm jobs on this cluster; their logs are in `$TEMPLATES_DIR/logs`.
Speech and voice numbers come from `models/minimax-h3/check_lines.py`.

## 1. Weights

### MiniMaxAI/MiniMax-H3

The repo holds the same model twice: the original checkpoints for SGLang and
vLLM, and the diffusers layout at the top level (sizes from the Hugging Face
API).

| component | path | size | used by |
|---|---|---|---|
| transformer, text and keyframes | `transformer/` | 66.28 GB | `t2va`, `fl2va` |
| transformer, references | `transformer_ref/` | 66.28 GB | `ref2va` |
| text encoder (Qwen3-VL-32B) | `text_encoder/` | 66.73 GB | all |
| video VAE | `vae/` | 10.42 GB | all |
| audio VAE | `audio_vae/` | 0.61 GB | all |
| processor, tokenizer, schedulers | | 0.02 GB | all |
| original checkpoint, keyframes | `FL2VA/` | 144.05 GB | SGLang, vLLM |
| original checkpoint, references | `Ref2VA/` | 144.05 GB | SGLang, vLLM |

The repo is 498.5 GB. The template downloads the diffusers layout, the
`LICENSE` and the `docs/` prompt guides: 210.3 GB. A workflow-pruned pipeline
loads only one of the two transformers (`generate.py`).

The released checkpoints are the H3-Base model at 768p. Two other parts of
MiniMax's system are not open
([model card](https://huggingface.co/MiniMaxAI/MiniMax-H3), System Overview):

- **H3-Context-IR** rewrites a short request into H3's structured prompt.
  MiniMax "strongly recommend[s]" it or a context system of your own built on
  their prompting guide.
- **H3-Regenerate-2K** regenerates the 768p result at 2K.

Both are available only through MiniMax's API.

### License and access

The repo is not gated (Hugging Face API: `gated: false`). Weights and code are
under the MiniMax H3 Community License Agreement (`LICENSE`, dated
2026-08-02):

- **Territory.** The license covers the "Applicable Territory": worldwide,
  excluding the EU, the UK, South Korea and the US (Definitions 3 and 5). Use
  outside it is not authorized (V.4, Exhibit A item 1). Organizations there can
  apply for a license ([application form](https://platform.minimax.io/h3-license);
  `docs/QA-about-License.md`).
- **Revenue.** Commercial products and services with more than $20M yearly
  revenue need prior written authorization (IV.1).
- **Attribution.** Commercial products must show "MiniMax H3" in their user
  interface (IV.2). Redistribution needs a `NOTICE` file and a copy of the
  agreement (III).
- **Outputs.** MiniMax claims no rights over outputs (VI.4). Outputs may not
  be used to improve other AI models (V.3). Content published anywhere public
  must be clearly marked as machine-generated (Exhibit A item 12).
  Impersonating a person without consent is excluded (item 13).
- The text encoder, Qwen3-VL-32B, is under Apache 2.0.

## 2. Hardware

- **What MiniMax documents.** The model card serves H3 with SGLang on 4 GPUs
  with Ulysses sequence parallelism (`--num-gpus 4 --ulysses-degree 4`). It
  gives no minimum GPU memory.
- **One B300 holds everything.** In bf16 the transformer is 61.7 GiB, the
  Qwen3-VL encoder 62.1 GiB and the VAEs 10 GiB. All stay resident; nothing is
  offloaded.
- **Measured peak GPU memory:**
  - 136 GiB for a 5 s `t2va` clip (1344, 1345)
  - 147 GiB for 14.4 s (1353)
  - 141–154 GiB for `ref2va` with 1–3 pictures and 1–2 voices (1352, 1403)
  - 170 GiB with 5 pictures (1409–1413)
- **Eight jobs per node.** Each job asks for one GPU and 250 GB of host memory.
  `DefMemPerNode` is unlimited here, so without `--mem` one job takes the
  whole node.

## 3. Measured results on B300

### Time per job

A job's time is measured from start to "Done":
- load: Python start-up and model loading
- generation: 50 steps plus decoding

Loading takes 72–128 s from NFS. It takes 31–49 s when the node's page cache
already holds the weights from a previous job (1363–1365, 1368, 1369, 1422,
1423).

| workflow | clip | jobs | generation | peak |
|---|---|---|---|---|
| `t2va` | 5.17 s, 1344×768 | 1344, 1345, 1350 | 172–176 s | 136 GiB |
| `t2va` | 5.17 s, 768×1344 | 1349, 1369 | 169–170 s | 136 GiB |
| `t2va` | 8.00 s | 1346, 1368, 1396–1399 | 314–330 s | 140 GiB |
| `t2va` | 10.12 s | 1347 | 451 s | 142 GiB |
| `t2va` | 14.38 s | 1353, 1354 | 785–811 s | 147 GiB |
| `fl2va`, first frame | 5.17 s / 8.00 s | 1351 / 1367 | 184 s / 327 s | 137 / 140 GiB |
| `ref2va`, 1 picture | 8.00 s | 1389–1392 | 439–466 s | 142 GiB |
| `ref2va`, 1 picture + 1 voice | 6.58 s | 1352, 1363 | 353–361 s | 141 GiB |
| `ref2va`, 2 pictures + 1 voice | 7.29 s | 1402, 1416 | 544–546 s | 146 GiB |
| `ref2va`, 3 pictures + 2 voices | 6.58 s / 9.42 s / 12.96 s | 1418 / 1406, 1419 / 1404 | 655 s / 912–915 s / 1287 s | 154 GiB |
| `ref2va`, 5 pictures + 2 voices | 7.29 s / 12.96 s | 1409 / 1410 | 1112 s / 1813 s | 170 GiB |
| `ref2va`, 1 video (8 s) | 8.00 s | 1366 | 1070 s | 151 GiB |

- **Generation time grows faster than clip length:** about 33 s per second of
  video at 5 s, 45 s at 10 s and 55 s at 14.4 s.
- **References add a lot.** Against text alone at the same length:
  - one picture takes 1.4× as long (8 s: 1389–1392 vs 1396–1399)
  - one picture and one voice 1.5× (6.58 s: 1352, 1363 vs 1364, 1365)
  - three pictures and two voices 2.2–2.7× (10.12 s: 1405 vs 1347; 6.58 s:
    1418 vs 1364)
- **Eight jobs side by side:** seeds 1–8 of the same 5 s prompt ran together on
  one node (1355–1362, `worker-b300-1`). Generation took 164–174 s, the same as
  a job alone. Loading took 121–128 s, against about 100 s alone.

Every output was H.264 with AAC stereo audio at 32 kHz.

### Speech

Whisper large-v3 transcribes the audio. The word error rate (WER) is measured
against the `<d>` lines in the prompt. Whisper writes numbers as digits, so
"vier" against "4" counts as one error.

| prompt | jobs | WER |
|---|---|---|
| German bakery line, 5 s, seeds 42, 7 and 1–8, 16:9 and 9:16 | 1345, 1349, 1350, 1355–1362 | 0.06 in all 11; the only error is "vier" → "4" |
| English lighthouse line, 5 s / 14.4 s | 1344 / 1353 | 0.00 / 0.07 ("Forty" → "40") |
| German, two speakers with a cut, 10 s | 1347 | 0.00 |
| German monologue, 14.4 s | 1354 | 0.00 |
| German, first frame given (`fl2va`) | 1351 | 0.00 |
| German, picture and voice reference | 1352, 1363 | 0.00 |
| German dialogue scene, 12 clips, 20 lines (section 6) | 1402–1407, 1416–1427 | every line correct apart from digits, except "Ich fahre" spoken as "Ich fahr" in both takes of one clip (1421, 1427) |

- **Speaking rate follows clip length.** The same 15-word English line ran at
  3.8 words/s in a 5 s clip (1344) and at 1.6 words/s in a 14.4 s clip (1353).
  The longer clip filled the time with slower speech and pauses.
- **Lines start late when actions come first.** A clip runs the prompt's
  description in order. In 10–12 s clips whose prompt described movement
  before the line, lines started 4.2–8.9 s in (1370–1381). With the line
  written right after the framing, they started at 0.2–0.6 s (1402–1407).

### Seeds

The pipeline draws its noise from a `torch.Generator` seeded with `SEED`
(`generate.py`).

- **The same prompt, seed, length and size give the same video, byte for
  byte.** Jobs 1345 (2026-10-14), 1439 and 1440 (2026-10-15) rendered the
  bakery prompt with seed 42. All three on `worker-b300-1` wrote mp4 files with
  the same MD5 (`5c029926…`).
- **A different seed is a different take.** Seeds 1–8 (1355–1362) gave eight
  different women matching the description, each in a different shop, with the
  same words.
- **Seeds do not carry identity.** A person stays the same across prompts only
  through references (section 5).

## 4. Limits

- **Length: 5 to 14.38 s per clip.**
  - The model card states 4–15 s. diffusers allows 5–15 s
    (`modular_pipeline.py:244-251`).
  - The frame count is rounded up to 17n+5 for the video VAE and may not exceed
    360 (`before_denoise.py:407`).
  - 15 s is 360 frames and rounds up to 362, so it fails. Job 1348 failed after
    loading for 104 s: "must be between 120 and 360, got 360 (rounded up to
    362)".
  - The longest clip is 345 frames (14.38 s). The template checks the length
    before loading.
- **Size.**
  - The short side is 768 pixels; 2K needs the closed H3-Regenerate-2K.
  - Aspect ratios run from 1:4 to 4:1 (`modular_pipeline.py:33-34`).
  - The template caps the canvas at 768×1344 (`resolve_canvas_size`).
- **Audio:** 32 kHz stereo (model card).
- **References.**
  - At most 9 images, 3 videos and 3 audio clips, 12 in total
    (`before_encoder.py:172-174`, `:411-416`).
  - An audio reference needs a picture or video next to it
    (`references.py:174`).
  - Video and audio references are 2–15 s each, 15 s in total per modality
    (model card). Two voices of 7 s fit into one clip; three do not.
- **No negative prompt.** The checkpoints are guidance-distilled: one forward
  pass per step, no guidance scale (`modular_pipeline.py:166-167`).
- **Languages.** The model card lists stable support for 11: Arabic, Chinese,
  English, French, German, Italian, Japanese, Korean, Portuguese, Russian and
  Spanish. German measured well (section 3).
- **Prompt format.** H3 expects MiniMax's structured prompt
  (`docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md`, `..._ref_en.md`). Writing it
  is up to the user, because H3-Context-IR is not released.

## 5. References and voices

Voice similarity is the cosine between WavLM speaker embeddings
(`microsoft/wavlm-base-plus-sv`) of a line and of a 7 s voice clip. The voice
clips were cut from earlier renders: the woman's from 1346, the man's from
1368. Section 6 later used a second man's voice, cut from 1399. Across the
scene in section 6, a line scored 0.73–0.99 against its own speaker and
0.36–0.73 against the other one. The lowest own-speaker scores were lines of
1–2 s.

| experiment | jobs | similarity to the woman's or man's voice clip |
|---|---|---|
| same scene, picture + voice reference vs text only, seeds 42 and 7 | 1352, 1363 vs 1364, 1365 | 0.974, 0.966 vs 0.907, 0.908 |
| the man's picture, no voice reference, four different descriptions of voice and delivery | 1389–1392 | 0.977–0.982 to his own voice clip |
| four different-looking men, text only, each with a different voice description (deep bass, nasal tenor, hoarse, velvety baritone) | 1396–1399 | 0.909–0.971 to the man's voice clip |

- **A voice reference holds the voice.** Text alone drifted to 0.91.
- **A face brings its own voice.** With the picture but without a voice
  reference, four descriptions ("with a cold edge, speaking slowly", "of a
  long-time smoker, unhurried", …) changed the delivery but not the timbre:
  0.977–0.982 to the voice the man had in his own clip.
- **A different voice comes from casting.** Rendering different-looking people
  from text and cutting the voice of the one you want gave clearly different
  voices (down to 0.91). That clip then serves as the voice reference.

### Making a take longer than one clip

All three continued the same 8 s clip by 8 s. The jump at the join is the mean
frame difference (downsampled grayscale) between the last frame and the next
one. Within normal motion, the median is 0.56 and the 99th percentile 2.06.

| method | jobs | jump at the join | voice | generation |
|---|---|---|---|---|
| last frame as first frame (`IMAGE`, `fl2va`) | 1367 | 1.85, seamless | 0.944, drifts | 327 s |
| previous clip as a video reference | 1366 | 9.1 | 0.984 | 1070 s |
| last frame as `<Picture 1>` "first frame" + voice reference | 1382 | 12.1 | 0.982 | 453 s |

No method kept both the picture and the voice. For dialogue, cut between clips.

## 6. Scenes of several clips

The test scene: two people (a woman and a man), German dialogue, vertical
768×1344, 12 clips, 1:40. Every clip used `ref2va` with:
- a picture of each person in it
- a 7 s voice clip of each speaker
- a picture of the set

Versions, in order:

| version | jobs | change | first line starts | gap to the answer | framing |
|---|---|---|---|---|---|
| 1 | 1370–1381 | one line per 10–12 s clip; the set picture `fully_preserved` | 4.2–8.9 s | one clip per line | shots open on the empty set and people walk in |
| 2 | 1383–1388 | one exchange per clip with a timed cut; people "already" in place "from the very first frame"; the line right after the framing; the set picture as background only | 0.2–2.7 s | 0.8–3.5 s | no walk-ins; one person drifts to the centre |
| 3a | 1402–1407 | clip length from the words; "at a natural conversational pace" | 0.2–0.6 s | 0.6–2.0 s | two-shots keep their sides; single close-ups flip sometimes |
| 3b | 1408–1413 | 3a plus a text-free storyboard sketch per shot as a picture reference | as 3a | as 3a | the woman's close-ups on the correct side; an over-the-shoulder shot lost; one continuity error; generation +41–60 % |
| 4 | 1416–1427 | six more clips, two takes each: seed 42 text only, and seed 7 with sketches for single-person close-ups | 0.3 s where a clip opens on a line | 1.8–3.1 s | each losing take had a visible error |

In every version, every line was closest to its own speaker's voice. In 3a and
4 the median speaking rate was 2.5–2.6 words/s.

- **Two takes per clip.** Picking the better take per clip caught every
  visible error (walk-in, missing prop, wrong side). Wording changes to the
  prompt did not catch them all.
- **Storyboard sketches as references** put single close-ups on the correct
  side but broke other shots. They also cost 25–60 % more generation time
  (e.g. 655 → 1017 s, 1418 vs 1424). In version 4 the sketch takes also had a
  different seed, so sketch and seed are not separated there; 3a against 3b is
  the clean comparison.
- **Cost:** 1.82 GPU-hours for the 56.5 s of 3a, and 1.39 for the 43 s of
  version 4 at seed 42. Both are about 0.032 GPU-hours per second of finished
  film per take, loading included.

Still open:

- Single close-ups sometimes swap sides.
- Shots that track a walking person open on the empty set for a beat.
- H3 occasionally changes a word ("fahre" → "fahr").
- Room tone differs from clip to clip; level jumps at cuts and lip sync were
  not measured.

The template's two-clip bakery scene (README section 5) was rendered as
written:
- the two cast clips: 1437, 1438 (8 s, `t2va`)
- each clip with seeds 42 and 7: 1443–1446 (7.29 s, 2 pictures + 2 voices).
  Generation took 536–551 s, a whole job 755–768 s, peak 146 GiB.

All 8 lines came out word for word, each closest to its own speaker (0.94–0.98
against 0.64–0.76). The answer followed the line after 0.6–1.7 s. The two-shots
kept the sides as written. In both seeds, the baker's single close-up was
centred rather than in the right half. Without a picture of the set, the shop
looked different in each clip.

## 7. Setup defects found

| defect | symptom | fix in the template |
|---|---|---|
| `hf download --include a b c` | only `a` is an include pattern; `b` and `c` become file names and the CLI then ignores `--include`, so `model_index.json` was never fetched | one `--include` per pattern (`download.sh`) |
| `DURATION=15` | 360 frames round up to 362; diffusers refuses after the 100 s load (1348) | `generate.py` caps at 345 frames and checks 5–15 s before loading |
| `DefMemPerNode=UNLIMITED` | a 1-GPU job without `--mem` takes the whole node | `--mem=250G` |
| diffusers 0.41.0 was released against transformers 5.18; the shared venv pins 5.16.1 for vLLM | | separate `venv-h3` |
| no torchaudio 2.13 on any index | | torchaudio 2.11 (abi3, no torch pin); diffusers uses it only to resample reference audio |
| no `ffmpeg` on the login node | | `cut_ref.py` and `join.py` use PyAV |
