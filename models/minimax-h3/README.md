# MiniMax H3 video generation on Slurm

Generate video with synchronized stereo sound (speech, effects, music) with
[MiniMax H3](https://huggingface.co/MiniMaxAI/MiniMax-H3), a 33B audio-video
model, from a prompt, from a first and/or last frame, or from reference
images, videos and voice clips. One Slurm job renders one video on one GPU
and exits.

| file | what it does |
|---|---|
| `setup.sh` | one-time install of the Python environment for H3 |
| `download.sbatch` | Slurm job that downloads the weights to shared storage (CPU only) |
| `download.sh` | the download itself; also runs directly on a login node |
| `generate.sbatch` | Slurm job that renders one video |
| `generate.py` | the generation, run by `generate.sbatch` |
| `cut_ref.py` | cuts a reference picture and voice clip out of a video |
| `check_lines.py` | checks each spoken line: words, timing, speaking rate, voice |
| `join.py` | cuts clips together into one video |
| `prompts/` | example prompts in English and German |

Requirements:

- NVIDIA B300 GPUs. Each video uses one GPU, so a node with 8 renders 8 videos
  at once.
- A filesystem shared by the login node and all compute nodes, here
  `/mnt/data`, with 220 GB free for weights and environment.
- Internet access from the login node and the compute nodes.
- A license that covers where you run it. The MiniMax H3 Community License
  covers every country except the EU member states, the UK, the US and South
  Korea. There you need MiniMax's written authorization first
  ([application form](https://platform.minimax.io/h3-license)).

`$TEMPLATES_DIR` is `/mnt/data/slurm-llm-templates`. All commands below run on
the login node.

## 0. One-time setup

```bash
git clone https://github.com/kreuzhofer/slurm-llm-training-templates.git
cd slurm-llm-training-templates
bash cluster/setup.sh                 # shared venv + copy of the repo on shared storage
bash models/minimax-h3/setup.sh       # H3 venv ($TEMPLATES_DIR/venv-h3), ends with a 1-GPU check
source /mnt/data/slurm-llm-templates/activate.sh
cd $TEMPLATES_DIR/repo/models/minimax-h3
```

`setup.sh` takes about 10 minutes. It ends with `=== MiniMax H3 setup complete ===`.

## 1. Download the weights

```bash
sbatch download.sbatch
tail -f $TEMPLATES_DIR/logs/h3_download_<JOBID>.out      # ends with "Done."
```

This fetches 210 GB (a few minutes) to `$TEMPLATES_DIR/models/MiniMax-H3`:
the two transformers (one for text and keyframes, one for references), the
Qwen3-VL text encoder and the VAEs.

## 2. Generate a video

```bash
sbatch generate.sbatch prompts/en-lighthouse.txt
```

The job runs on a compute node, so a prompt file must be on shared storage,
for example under `$TEMPLATES_DIR` or your home directory, not under `/tmp`.

Follow the job:

```bash
squeue --me
tail -f $TEMPLATES_DIR/logs/h3_generate_<JOBID>.out      # ends with "Done in ... s"
```

The video is `$TEMPLATES_DIR/output/minimax-h3/h3_<JOBID>.mp4` (H.264 with AAC
stereo audio, 1344×768 at 24 fps), the prompt is saved next to it as
`h3_<JOBID>.prompt.txt`. A job takes about 6 minutes for 5 s, 11 minutes for
10 s and 17 minutes for 14 s; about 2 of those minutes are loading the model.
A picture and a voice as references add about half to the generation time,
three pictures and two voices more than double it.

Each job uses one GPU. Submit as many as you like; Slurm runs eight per node
at the same time:

```bash
for f in prompts/en-*.txt prompts/de-*.txt; do sbatch generate.sbatch "$f"; done
```

Longer, vertical, another take:

```bash
DURATION=10 sbatch generate.sbatch prompts/de-office-two-shots.txt
ASPECT=9:16 sbatch generate.sbatch prompts/de-bakery.txt
SEED=7 sbatch generate.sbatch prompts/de-bakery.txt
```

## 3. Start or end on an image

`IMAGE` is the first frame, `LAST_IMAGE` the last one; with both, the video
moves from one to the other. The video takes the aspect ratio of `IMAGE`.

```bash
IMAGE=$HOME/office-start.png sbatch generate.sbatch prompts/<your-prompt>.txt
IMAGE=$HOME/a.png LAST_IMAGE=$HOME/b.png sbatch generate.sbatch prompts/<your-prompt>.txt
```

The prompt starts with a line that tells H3 where the image sits (see
[Writing prompts](#writing-prompts)).

## 4. Keep a character: reference image and voice

`REFS` takes up to 9 images, 3 videos and 3 audio clips (each 2 to 15 s). The
prompt calls them `<Picture 1>`, `<Video 1>`, `<Audio 1>`, … in the order
given. With a picture of a person and a recording of their voice, H3 puts that
person, with that voice, into a new scene:

```bash
REFS="$HOME/character.png $HOME/character-voice.wav" sbatch generate.sbatch prompts/ref-de-character.txt
```

Paths in `REFS` are separated by spaces, so they cannot contain spaces.

## 5. Longer scenes

A clip is at most 14.4 s. A longer scene is several clips cut together, and
every clip gets the same pictures and voices as references, so the people look
and sound the same across the cuts. The example is a scene of two clips in a
bakery.

Render each person once, alone, speaking to the camera, without music:

```bash
DURATION=8 sbatch generate.sbatch prompts/de-bakery.txt      # the baker
DURATION=8 sbatch generate.sbatch prompts/de-customer.txt    # the customer
```

`check_lines.py` shows where each one speaks:

```bash
PY=$TEMPLATES_DIR/venv-h3/bin/python
$PY check_lines.py $TEMPLATES_DIR/output/minimax-h3/h3_<BAKER_JOBID>.mp4 $TEMPLATES_DIR/output/minimax-h3/h3_<CUSTOMER_JOBID>.mp4
```

```
h3_<BAKER_JOBID>.mp4  8.0 s  language de  WER 0.06
  S1  0.4-5.2 s  3.3 words/s
      expected: Wir backen jedes Brot von Hand, jeden Morgen ab vier Uhr, nur mit Mehl, Wasser und Salz.
      heard   : Wir backen jedes Brot von Hand, jeden Morgen ab 4 Uhr, nur mit Mehl, Wasser und Salz.
h3_<CUSTOMER_JOBID>.mp4  8.0 s  language de  WER 0.00
  S1  0.4-7.2 s  2.6 words/s
      expected: Ich komme jeden Freitag hierher, immer kurz vor Ladenschluss, und nehme das letzte Roggenbrot mit, seit fast zehn Jahren.
      heard   : Ich komme jeden Freitag hierher, immer kurz vor Ladenschluss und nehme das letzte Roggenbrot mit. Seit fast zehn Jahren.
```

Cut a picture (the first frame) and the spoken line out of each:

```bash
mkdir -p $HOME/bakery
$PY cut_ref.py $TEMPLATES_DIR/output/minimax-h3/h3_<BAKER_JOBID>.mp4 $HOME/bakery/baker --length 5.5
$PY cut_ref.py $TEMPLATES_DIR/output/minimax-h3/h3_<CUSTOMER_JOBID>.mp4 $HOME/bakery/customer --length 7.5
```

This writes `baker.png` and `baker-voice.wav` (0 to 5.5 s), and the same for
the customer. `--frame` and `--start` choose another frame or start. Each voice
clip should hold that person's voice only, and the voices in one clip may add
up to 15 s.

Each clip is one exchange: a line, a cut, the answer. Both clip prompts use the
same four references in the same order, so `<Picture 1>` is always the
customer. Render each clip with two seeds and keep the better take:

```bash
export REFS="$HOME/bakery/customer.png $HOME/bakery/baker.png $HOME/bakery/customer-voice.wav $HOME/bakery/baker-voice.wav"
for seed in 42 7; do
    SEED=$seed DURATION=7 sbatch generate.sbatch prompts/ref-de-bakery-1.txt
    SEED=$seed DURATION=7 sbatch generate.sbatch prompts/ref-de-bakery-2.txt
done
unset REFS
```

A job takes about 13 minutes. Check what each take says, when, and in whose
voice:

```bash
$PY check_lines.py $TEMPLATES_DIR/output/minimax-h3/h3_<JOBID>.mp4 \
    --voice customer=$HOME/bakery/customer-voice.wav --voice baker=$HOME/bakery/baker-voice.wav
```

```
h3_<JOBID>.mp4  7.3 s  language de  WER 0.00
  S1  1.3-2.5 s  3.3 words/s  closest customer 0.96 (baker 0.71)
      expected: Haben Sie noch ein Roggenbrot?
      heard   : Haben Sie noch ein Roggenbrot?
  S2  3.1-5.4 s  3.0 words/s  closest baker 0.94 (customer 0.64)
      expected: Das letzte. Ich habe es für Sie zurückgelegt.
      heard   : Das letzte. Ich habe es für Sie zurückgelegt.
```

Each line should be closest to its own speaker's voice. `check_lines.py` runs
on the login node in one to two minutes (Whisper large-v3 and a speaker model,
3.5 GB, download on first use). Whisper writes numbers as digits, so "vier"
against "4" counts as one wrong word.

Cut the chosen takes together:

```bash
$PY join.py $HOME/bakery/scene.mp4 \
    $TEMPLATES_DIR/output/minimax-h3/h3_<CLIP1_JOBID>.mp4 $TEMPLATES_DIR/output/minimax-h3/h3_<CLIP2_JOBID>.mp4
```

What keeps a scene together:

- Every clip describes each person the same way, next to their picture, in
  `subject_definitions`.
- People are in place from the first frame: `From the very first frame, ...
  already standing ... Nobody enters or leaves the frame`. Without it, shots
  tend to open on the empty room and people walk in.
- The line comes right after the framing. Actions written before a line are
  played before it, as silence.
- One exchange per clip, with the cut at a set time:
  `[Shot 2] At 00:02.800, the camera cuts to ... Immediately, ... says`.
- Clip length: about 2.5 words per second plus 0.6 s per line. A clip much
  longer than its lines slows the speech down and adds pauses.
- `at a natural conversational pace` and `immediately` keep the dialogue
  moving; `slowly` and `measured` slow it down.
- Each person keeps one side of the frame: here the customer on the left
  looking right, the baker on the right looking left. Two-shots keep it;
  single close-ups sometimes drift to the centre or the other side.
- The room changes from clip to clip unless it is a reference too. Add a
  picture of the empty set as one more `<Picture>`, and mark it as background
  only: `<Subject 3>: reference - the bakery serves only as the background
  behind the people; the target video never shows it empty.`
- Show who speaks. A voice over someone else's face looks like a mistake.
- Write numbers as words (`zehn Jahren`).
- A face brings its own voice. A different voice comes from a voice
  reference; describing the voice in the prompt changes the delivery, not the
  voice.
- Do not continue a clip from its last frame (`IMAGE`) to make a longer
  dialogue: the picture joins seamlessly, but the voice drifts away. Cut
  between clips instead.

## 6. Copy the video to your laptop

On your laptop:

```bash
scp -i ~/.ssh/<your-private-key> \
    <user>@<login-node-address>:/mnt/data/slurm-llm-templates/output/minimax-h3/h3_<JOBID>.mp4 .
```

## Options

Set these as environment variables in front of `sbatch`:

| variable | default | meaning |
|---|---|---|
| `DURATION` | `5` | length in seconds, 5 to 15; the longest clip is 14.4 s (`15` gives that) |
| `ASPECT` | `16:9` | `9:16`, `1:1`, `4:3`, `21:9`, …; the short side is 768 pixels. With `IMAGE` the default is the image's aspect |
| `SEED` | `42` | change it to get a different take of the same prompt |
| `STEPS` | `50` | denoising steps |
| `IMAGE`, `LAST_IMAGE` | empty | first and last frame (section 3) |
| `REFS` | empty | reference images, videos and audio, space-separated (section 4). Not together with `IMAGE` |

## Writing prompts

H3 expects a structured prompt. MiniMax's guides are downloaded with the
weights:
`$TEMPLATES_DIR/models/MiniMax-H3/docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md`
(text and keyframes) and `..._ref_en.md` (references). The examples in
`prompts/` follow them.

A text prompt has three fields, written in English:

```
integrated_multimodal_description: [Shot 1] Live-action, cinematic, a medium close-up frames ...
The baker with a warm, clear voice (S1) looks into the camera and says: <d>[German] Wir backen jedes Brot von Hand.</d> She closes her lips ...

overall_soundscape: Quiet street noise and the hum of an oven ...

non_diegetic_music: N/A
```

- Describe the shot along its timeline: framing, people, action, camera
  movement (type, amplitude, speed), light.
- Each speaker gets an ID, `(S1)`, `(S2)`, and a voice description when they
  first speak. The spoken words go inside `<d>[Language] ... </d>`, in that
  language; H3 does not translate them. MiniMax lists stable support for
  Arabic, Chinese, English, French, German, Italian, Japanese, Korean,
  Portuguese, Russian and Spanish.
- End a line by saying what the speaker does when it stops, e.g. `She closes
  her lips and nods.`, as MiniMax's examples do.
- For a cut, start a new `[Shot 2]` with its time: `At 00:04.000, the camera
  cuts to ...`.
- `overall_soundscape` holds ambience and effects, `non_diegetic_music` the
  score only the audience hears, or `N/A`.
- With `IMAGE`, the first line is
  `For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.`
  followed by a blank line.
- With `REFS`, the prompt has six sections (`subject_definitions`, `summary`,
  `retention_analysis`, `detailed_description`, `overall_soundscape`,
  `non_diegetic_music`); see `prompts/ref-de-character.txt`, and
  `prompts/ref-de-bakery-1.txt` for two people.

MiniMax's hosted service rewrites a short request into this format (H3-Context-IR)
and can upscale the result to 2K (H3-Regenerate-2K). Neither is part of the open
weights; this template renders the 768p base output from the prompt you write.

## Logs

| job | log |
|---|---|
| download | `$TEMPLATES_DIR/logs/h3_download_<JOBID>.out` |
| generate | `$TEMPLATES_DIR/logs/h3_generate_<JOBID>.out`, `.err` (progress bars, warnings) |

## License

MiniMax H3 is released under the
[MiniMax H3 Community License Agreement](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE).
It does not cover the EU, the UK, the US or South Korea; organizations there
need a license from MiniMax ([application form](https://platform.minimax.io/h3-license)).
Products and services with more than $20M yearly revenue need MiniMax's written
authorization, and commercial products must show "MiniMax H3" in their user
interface. Videos published anywhere public must be clearly marked as
machine-generated (Acceptable Use Policy, item 12). The text encoder is
Qwen3-VL-32B under Apache 2.0.
