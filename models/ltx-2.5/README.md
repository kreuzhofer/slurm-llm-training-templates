# LTX-2.5 video generation on Slurm

Generate video with synchronized sound (speech, effects, ambience) from a text
prompt with [LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5), Lightricks'
22B audio-video model. One Slurm job renders one video on one node and exits.

| file | what it does |
|---|---|
| `setup.sh` | one-time install of the LTX-2 code and its own Python environment |
| `download.sbatch` | Slurm job that downloads the weights to shared storage (CPU only) |
| `download.sh` | the download itself; also runs directly on a login node |
| `generate.sbatch` | Slurm job that renders one video from a prompt |
| `vertical_20s.sh` | renders a 20 s vertical video from a prompt, follows the job and prints the video's path |
| `prompts/` | example prompts in English and German |
| `check_speech.py` | transcribes a video's audio and compares it with the prompt's dialogue |

Requirements:

- NVIDIA B300 GPUs. Each video uses one GPU, so a node with 8 renders 8 videos
  at once.
- A filesystem shared by the login node and all compute nodes, here
  `/mnt/data`, with 80 GB free for weights and environment.
- Internet access from the login node and the compute nodes.
- A Hugging Face account that has accepted the LTX-2.x Community License (step 1).

`$TEMPLATES_DIR` is `/mnt/data/slurm-llm-templates`. All commands below run on
the login node.

## 0. One-time setup

```bash
git clone https://github.com/kreuzhofer/slurm-llm-training-templates.git
cd slurm-llm-training-templates
bash cluster/setup.sh                 # shared venv + copy of the repo on shared storage
bash models/ltx-2.5/setup.sh          # LTX venv ($TEMPLATES_DIR/venv-ltx), ends with a 1-GPU check
source /mnt/data/slurm-llm-templates/activate.sh
cd $TEMPLATES_DIR/repo/models/ltx-2.5
```

`setup.sh` takes 15 to 25 minutes. It ends with `=== LTX-2.5 setup complete ===`.

## 1. Accept the license and download the weights

Both model repositories are gated. Log in to Hugging Face with the account
whose token is on the cluster (`hf auth whoami`) and accept the license on:

- https://huggingface.co/Lightricks/LTX-2.5
- https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Pixel-Spatial-Upscaler

Approval is immediate. Then download (71 GB, a few minutes):

```bash
sbatch download.sbatch
tail -f $TEMPLATES_DIR/logs/ltx25_download_<JOBID>.out      # ends with "Done."
```

## 2. Generate a video

```bash
sbatch generate.sbatch prompts/en-bakery.txt
```

The job runs on a compute node, so a prompt file must be on shared storage,
for example under `$TEMPLATES_DIR` or your home directory, not under `/tmp`.

Or pass the prompt inline:

```bash
sbatch generate.sbatch "A red fox trots through fresh snow at dawn, its breath steaming. Birds chirp in the distance."
```

Follow the job:

```bash
squeue --me
tail -f $TEMPLATES_DIR/logs/ltx25_generate_<JOBID>.out      # ends with "Done in ... s"
```

The video is `$TEMPLATES_DIR/output/ltx-2.5/ltx25_<JOBID>.mp4` (H.264 with AAC
stereo audio), the prompt is saved next to it as `ltx25_<JOBID>.prompt.txt`.
A 5 s clip takes about 3 to 4 minutes, most of it loading the model.

Each job uses one GPU. Submit as many as you like; Slurm runs eight per node
at the same time:

```bash
for f in prompts/*.txt; do sbatch generate.sbatch "$f"; done
```

### A 20 s vertical video in one command

`vertical_20s.sh` submits the job at 1088×1920 and 481 frames, shows the job
log while it runs and prints the path of the video at the end:

```bash
bash vertical_20s.sh "A street musician in Vienna plays the violin, then says in German: \"Danke fürs Zuhören.\""
```

```
Submitted job 1340. Waiting for it to start ...
...
Video: /mnt/data/slurm-llm-templates/output/ltx-2.5/ltx25_1340.mp4
```

Ctrl-C stops following the log; the job keeps running. `SEED=7 bash
vertical_20s.sh "..."` renders another take of the same prompt.

## 3. Copy the video to your laptop

On your laptop:

```bash
scp -i ~/.ssh/<your-private-key> \
    <user>@<login-node-address>:/mnt/data/slurm-llm-templates/output/ltx-2.5/ltx25_<JOBID>.mp4 .
```

## Options

Set these as environment variables in front of `sbatch`:

| variable | default | meaning |
|---|---|---|
| `PIPELINE` | `distilled` | `distilled` is fast; `dfr` adds a detailing pass for production quality and takes longer |
| `NUM_FRAMES` | `121` | length in frames, must be 8k+1: `121` = 5 s, `241` = 10 s, `481` = 20 s at 24 fps; `auto` lets the model choose 1–20 s from the prompt. See [Clip length](#clip-length) |
| `WIDTH`, `HEIGHT` | `1536`, `1024` | output size, both divisible by 64; e.g. `1920`×`1088` landscape, `1088`×`1920` portrait |
| `FRAME_RATE` | `24` | frames per second |
| `SEED` | `42` | change it to get a different take of the same prompt |
| `IMAGE` | empty | path to a still image to use as the first frame (image-to-video) |
| `EXTRA_ARGS` | empty | extra flags for the LTX pipeline, passed as-is |

Examples:

```bash
NUM_FRAMES=241 SEED=7 sbatch generate.sbatch prompts/de-bakery.txt
PIPELINE=dfr WIDTH=1920 HEIGHT=1088 sbatch generate.sbatch prompts/de-multishot.txt
IMAGE=$HOME/first-frame.png sbatch generate.sbatch "The woman in the photo turns and waves."
```

## Clip length

LTX-2.5 is made for clips of up to 20 seconds (`NUM_FRAMES=481`). Longer clips
still render (`721` frames = 30 s), but the picture starts to drift towards the
end. For longer scenes, cut them into shots of 20 seconds or less and edit them
together, or use several shots in one prompt (see below).

Do not use `EXTRA_ARGS=--chunked` for clips with speech. It renders the clip in
short windows that each start the dialogue from the beginning.

## Writing prompts

- Describe one scene as a flowing paragraph in present tense: shot and
  framing, subject, action, camera movement, light, then the sound.
- Put spoken lines in quotation marks and name the language and voice:
  `she says in German with a warm, clear voice: "..."`.
- Write the dialogue in the target language. The model speaks what is in the
  quotes and does not translate. The model card lists English, German, French,
  Spanish, Italian, Portuguese, Japanese, Korean and Chinese. The rest of the
  prompt can be in English or in the dialogue's language.
- One speaker per sentence. With two or more people talking, the model can
  give a line to the wrong person.
- For several shots in one video, name each cut: `A hard cut transitions to a
  close-up of ...`, and say whether the sound continues across it. Two to four
  shots per video work best.
- Keep text on screen short; spelling is not reliable.

`prompts/` has worked examples, including a German one with three shots.

## Check what was said

`check_speech.py` transcribes the audio with Whisper large-v3 (downloads 3 GB
on first use), detects the spoken language and compares the words with the
quoted dialogue in the prompt as a word error rate (WER). It runs on the login
node in about a minute:

```bash
$TEMPLATES_DIR/venv-ltx/bin/python check_speech.py $TEMPLATES_DIR/output/ltx-2.5/ltx25_<JOBID>.mp4
```

```
ltx25_<JOBID>.mp4
  language  : de
  expected  : Wir backen jedes Brot von Hand, jeden Morgen ab vier Uhr, nur mit Mehl, Wasser und Salz.
  transcript: Wir backen jedes Brot von Hand, jeden Morgen ab 4 Uhr, nur mit Mehl, Wasser und Salz.
  WER       : 0.06
```

Whisper writes numbers as digits, so "vier" against "4" counts as one wrong word.

## Logs

| job | log |
|---|---|
| download | `$TEMPLATES_DIR/logs/ltx25_download_<JOBID>.out` |
| generate | `$TEMPLATES_DIR/logs/ltx25_generate_<JOBID>.out`, `.err` (progress bars, warnings) |

## License

The LTX-2.5 weights and the LTX-2 code are released under the
[LTX-2.x Community License](https://github.com/Lightricks/LTX-2/blob/main/LICENSE-2_x).
Organizations with annual revenue of $10M or more need a paid license from
Lightricks (licensing@ltx.io) for any use other than testing, evaluation or
non-commercial research and development in a non-production environment.
