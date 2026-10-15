"""Render one MiniMax H3 video with sound and exit.

    python generate.py --model-dir $TEMPLATES_DIR/models/MiniMax-H3 \
        --prompt-file prompt.txt --output out.mp4 [--duration 5] [--aspect 16:9] \
        [--seed 42] [--steps 50] [--image first.png] [--last-image last.png] \
        [--ref face.png --ref voice.wav ...]

The inputs pick the workflow, as in diffusers:
  t2va    text only
  fl2va   --image and/or --last-image: the video starts and/or ends on them
  ref2va  --ref: up to 9 images, 3 videos and 3 audio clips (12 in total) the
          prompt refers to as <Picture 1>, <Video 1>, <Audio 1> in the order given

Normally run by generate.sbatch, which sets the paths.
"""

import argparse
import os
import time

import torch
from diffusers import ModularPipeline
from diffusers.modular_pipelines.minimax_h3 import (
    MiniMaxH3AudioReference,
    MiniMaxH3ImageReference,
    MiniMaxH3VideoReference,
)
from diffusers.modular_pipelines.minimax_h3.modular_pipeline import MINIMAX_H3_FPS, resolve_canvas_size
from diffusers.utils import load_image
from diffusers.utils.export_utils import encode_video

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm", ".avi"}
AUDIO_EXT = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac"}


def frame_count(duration):
    """Frames for `duration` seconds (5 to 15). The VAE decodes 17n+5 frames, so
    the count is rounded up to that form, but diffusers refuses more than 360
    after rounding: 15 s is 360 frames, rounds to 362 and fails with "num_frames
    ... must be between 120 and 360" after the 100 s load. 345 frames (14.4 s)
    is the longest clip, and it is checked here, before loading."""
    if not 5 <= duration <= 15:
        raise SystemExit(f"--duration must be 5 to 15 seconds, got {duration:g}")
    n = round(duration * MINIMAX_H3_FPS)
    while n % 17 != 5:
        n += 1
    return min(n, 17 * 20 + 5)


def load_reference(path):
    """A reference by file type. from_file keeps the frame and sample rates the
    container reports; H3 resamples onto its own 24 fps and 32 kHz from them."""
    ext = os.path.splitext(path)[1].lower()
    if ext in IMAGE_EXT:
        return MiniMaxH3ImageReference.from_file(path)
    if ext in VIDEO_EXT:
        return MiniMaxH3VideoReference.from_file(path)
    if ext in AUDIO_EXT:
        return MiniMaxH3AudioReference.from_file(path)
    raise SystemExit(f"--ref {path}: unknown file type {ext!r}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--prompt-file", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--duration", type=float, default=5.0,
                   help="5 to 15 seconds; becomes 124 to 345 frames (5.2 to 14.4 s), see frame_count.")
    p.add_argument("--aspect", default=None,
                   help="e.g. 16:9, 9:16, 1:1. Default: the first keyframe's aspect, else 16:9.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--image", default=None, help="first frame")
    p.add_argument("--last-image", default=None, help="last frame")
    p.add_argument("--ref", action="append", default=[], help="reference file, repeatable; order matters")
    args = p.parse_args()

    if args.ref and (args.image or args.last_image):
        raise SystemExit("--ref (ref2va) cannot be combined with --image/--last-image (fl2va)")
    workflow = "ref2va" if args.ref else "fl2va" if (args.image or args.last_image) else "t2va"
    num_frames = frame_count(args.duration)
    with open(args.prompt_file) as f:
        prompt = f.read().strip()

    t0 = time.time()
    # A workflow-pruned pipeline loads only its own transformer partition:
    # transformer/ for t2va and fl2va, transformer_ref/ for ref2va.
    pipe = ModularPipeline.from_pretrained(args.model_dir, workflow=workflow)
    pipe.load_components(dtype=torch.bfloat16)
    # Every component stays resident: 61.7 GiB transformer, 62.1 GiB Qwen3-VL
    # conditioner, 10 GiB VAEs. That fits one B300, so nothing is offloaded.
    pipe.to("cuda")
    print(f"Loaded {workflow} pipeline in {time.time() - t0:.0f} s", flush=True)

    kwargs = dict(
        prompt=prompt,
        num_frames=num_frames,
        num_inference_steps=args.steps,
        generator=torch.Generator().manual_seed(args.seed),
        output=["videos", "audio", "sampling_rate"],
    )
    if args.aspect:
        aw, ah = (float(x) for x in args.aspect.split(":"))
        # The released checkpoint's canvas: 768 short edge, at most 768x1344 pixels.
        kwargs["height"], kwargs["width"] = resolve_canvas_size(aw, ah, pipe.canvas_multiple, 768, 768 * 1344)
    if args.image:
        kwargs["image"] = load_image(args.image)
    if args.last_image:
        kwargs["last_image"] = load_image(args.last_image)
    if args.ref:
        kwargs["references"] = [load_reference(r) for r in args.ref]

    t1 = time.time()
    results = pipe(**kwargs)
    frames = results["videos"][0]
    print(f"Generated {len(frames)} frames at {frames[0].width}x{frames[0].height} "
          f"({len(frames) / MINIMAX_H3_FPS:.2f} s) in {time.time() - t1:.0f} s, "
          f"peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.0f} GiB", flush=True)

    encode_video(frames, fps=MINIMAX_H3_FPS, output_path=args.output,
                 audio=results["audio"][0].float().cpu(), audio_sample_rate=results["sampling_rate"])


if __name__ == "__main__":
    main()
