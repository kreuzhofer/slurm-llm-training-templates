"""check_lines.py -- did every line come out, when, how fast, and in whose voice?

Transcribes the audio track of H3 videos with Whisper large-v3 and splits the
transcript into the prompt's lines (every <d>[Language] ... </d> span). Per
line it prints when the line starts and ends, the speaking rate in words per
second and what Whisper heard; per video the word error rate (WER) against all
lines. With --voice NAME=WAV, each line is also compared with those voice
recordings (WavLM speaker embeddings, cosine similarity) and the closest one is
named, so a line spoken in the wrong voice stands out.

    python check_lines.py $TEMPLATES_DIR/output/minimax-h3/h3_1234.mp4 [...] \
        [--voice baker=$HOME/bakery/baker-voice.wav --voice customer=$HOME/bakery/customer-voice.wav] [--json]

The prompt is read from the h3_<JOBID>.prompt.txt that generate.sbatch writes
next to each video. Runs on the login node's CPUs, or with srun --gpus=1 on a GPU.

Whisper writes most numbers as digits ("vier" -> "4"), which counts as one
wrong word each and can shift the split between lines.
"""

import argparse
import json
import re
import string
import sys
import warnings
from pathlib import Path

import av
import numpy as np
import torch
import transformers
from transformers import AutoFeatureExtractor, WavLMForXVector, pipeline
from transformers.models.whisper.tokenization_whisper import TO_LANGUAGE_CODE

WHISPER = "openai/whisper-large-v3"
SPEAKER_MODEL = "microsoft/wavlm-base-plus-sv"
SR = 16000
DROP = str.maketrans("", "", string.punctuation + "„“”‚‘’«»–—…")


def load_audio(path) -> np.ndarray:
    """Audio track as 16 kHz mono float32."""
    resampler = av.AudioResampler(format="flt", layout="mono", rate=SR)
    out = []
    with av.open(str(path)) as container:
        for frame in container.decode(audio=0):
            out += [f.to_ndarray().reshape(-1) for f in resampler.resample(frame)]
    out += [f.to_ndarray().reshape(-1) for f in resampler.resample(None)]
    return np.concatenate(out)


def prompt_lines(prompt: str) -> tuple[str | None, list[dict]]:
    """(Whisper language code, lines). A line's speaker is the last (S<n>)
    written before it in the prompt."""
    lines, lang = [], None
    for m in re.finditer(r"<d>\s*\[([^\]]+)\]\s*(.+?)\s*</d>", prompt, re.S):
        lang = lang or TO_LANGUAGE_CODE.get(m.group(1).strip().lower())
        ids = re.findall(r"\((S\d+)\)", prompt[: m.start()])
        lines.append({"speaker": ids[-1] if ids else "", "text": m.group(2)})
    return lang, lines


def words(text: str) -> list[str]:
    return text.lower().translate(DROP).split()


def wer(ref: list[str], hyp: list[str]) -> float:
    if not ref:
        return float("nan")
    d = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        prev, d[0] = d[0], i
        for j, h in enumerate(hyp, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r != h))
    return d[-1] / len(ref)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("videos", nargs="+", type=Path)
    ap.add_argument("--voice", action="append", default=[], metavar="NAME=WAV",
                    help="a voice recording to compare every line with; repeatable")
    ap.add_argument("--json", action="store_true", help="one JSON object per video")
    args = ap.parse_args()
    voice_files = dict(spec.split("=", 1) for spec in args.voice if "=" in spec)
    if len(voice_files) != len(args.voice) or not all(voice_files) or not all(voice_files.values()):
        raise SystemExit("--voice takes NAME=WAV, one name per voice")
    transformers.logging.set_verbosity_error()
    transformers.logging.disable_progress_bar()
    warnings.filterwarnings("ignore", category=UserWarning)

    # A GPU when there is one; on a login node's CPUs a 10 s clip takes under a minute.
    device, dtype = ("cuda", torch.float16) if torch.cuda.is_available() else ("cpu", torch.float32)
    asr = pipeline("automatic-speech-recognition", model=WHISPER, dtype=dtype, device=device)
    fe = AutoFeatureExtractor.from_pretrained(SPEAKER_MODEL)
    sv = WavLMForXVector.from_pretrained(SPEAKER_MODEL).eval()

    @torch.inference_mode()
    def embed(x):
        e = sv(**fe(x, sampling_rate=SR, return_tensors="pt")).embeddings
        return torch.nn.functional.normalize(e, dim=-1)[0]

    voices = {name: embed(load_audio(path)) for name, path in voice_files.items()}

    for video in args.videos:
        prompt_file = video.with_suffix(".prompt.txt")
        if not prompt_file.exists():
            print(f"{video.name}: no {prompt_file.name} next to it", file=sys.stderr)
            continue
        lang, lines = prompt_lines(prompt_file.read_text())
        x = load_audio(video)
        generate_kwargs = {"task": "transcribe"} | ({"language": lang} if lang else {})
        # Whisper hears 30 s at a time; only a longer track needs chunking.
        chunking = {"chunk_length_s": 30} if len(x) > 30 * SR else {}
        res = asr({"raw": x, "sampling_rate": SR}, return_timestamps="word", generate_kwargs=generate_kwargs,
                  **chunking)
        heard = [(w["text"].strip(), *w["timestamp"]) for w in res["chunks"] if w["text"].strip().translate(DROP)]

        # Words go to lines in order, by each line's word count in the prompt.
        # Whisper places the first word of a clip at 0.0 s even after a pause,
        # so a line's start is its second word's start when it has one.
        k = 0
        for ln in lines:
            seg = heard[k : k + len(words(ln["text"]))]
            k += len(seg)
            ln["heard"] = " ".join(w[0] for w in seg)
            if not seg:
                continue
            start = seg[1][1] if len(seg) > 1 else seg[0][1]
            end = seg[-1][2] or seg[-1][1]
            ln["start"], ln["end"] = round(start, 2), round(end, 2)
            ln["wps"] = round((len(seg) - 1) / max(end - start, 0.1), 2) if len(seg) > 1 else None
            if voices:
                e = embed(x[int(max(start - 0.5, 0) * SR) : int(end * SR)])
                ln["voice"] = {n: round(float(v @ e), 3) for n, v in voices.items()}
                ln["closest"] = max(ln["voice"], key=ln["voice"].get)
        score = wer(words(" ".join(ln["text"] for ln in lines)), words(" ".join(w[0] for w in heard)))

        if args.json:
            print(json.dumps({"video": str(video), "language": lang, "length": round(len(x) / SR, 2),
                              "wer": score, "lines": lines}, ensure_ascii=False), flush=True)
            continue
        print(f"{video.name}  {len(x) / SR:.1f} s  language {lang or 'detected'}  WER {score:.2f}")
        for ln in lines:
            if "start" not in ln:
                print(f"  {ln['speaker']:3s} not heard: {ln['text']}")
                continue
            rate = f"{ln['wps']:.1f} words/s" if ln["wps"] else "1 word"
            voice = ""
            if voices:
                others = "  ".join(f"{n} {s:.2f}" for n, s in ln["voice"].items() if n != ln["closest"])
                voice = f"  closest {ln['closest']} {ln['voice'][ln['closest']]:.2f}" + (f" ({others})" if others else "")
            print(f"  {ln['speaker']:3s} {ln['start']:.1f}-{ln['end']:.1f} s  {rate}{voice}")
            print(f"      expected: {ln['text']}")
            print(f"      heard   : {ln['heard']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
