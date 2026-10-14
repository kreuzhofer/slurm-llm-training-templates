"""check_speech.py -- what does the generated character actually say?

Transcribes the audio track of LTX-2.5 videos with Whisper large-v3, lets
Whisper detect the spoken language, and compares the transcript with the
dialogue in the prompt (every "quoted" span) as a word error rate.

    python check_speech.py $TEMPLATES_DIR/output/ltx-2.5/ltx25_1234.mp4 [...]

Runs on the login node's CPUs or, with srun --gpus=1, on a GPU.

The prompt is read from the <video>.prompt.txt that generate.sbatch writes next
to each video. Prints one line per video and, with --json, one JSON object per
video for aggregation.

WER is computed on lowercased words with punctuation removed; umlauts and ß are
kept. It measures intelligibility of the words, not voice quality or lip sync.
Whisper writes most numbers as digits ("vier Uhr" -> "4 Uhr"), which counts as
one wrong word each.
"""

import argparse
import json
import re
import string
import sys
from pathlib import Path

import av
import numpy as np
import torch
import torchaudio
from transformers import WhisperForConditionalGeneration, WhisperProcessor

WHISPER = "openai/whisper-large-v3"
SR = 16000
WINDOW_S = 30  # Whisper's input window


def load_audio(path: Path) -> np.ndarray:
    """Audio track as 16 kHz mono float32."""
    with av.open(str(path)) as container:
        stream = container.streams.audio[0]
        frames = [f.to_ndarray() for f in container.decode(stream)]
        rate = stream.rate
    pcm = np.concatenate(frames, axis=1).astype(np.float32)  # (channels, samples), planar
    mono = torch.from_numpy(pcm.mean(axis=0))
    return torchaudio.functional.resample(mono, rate, SR).numpy()


def expected_dialogue(prompt: str) -> str:
    return " ".join(re.findall(r'["“„](.+?)["”“]', prompt))


def words(text: str) -> list[str]:
    drop = str.maketrans("", "", string.punctuation + "„“”‚‘’«»–—…")
    return text.lower().translate(drop).split()


def wer(ref: list[str], hyp: list[str]) -> float:
    if not ref:
        return float("nan")
    d = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        prev, d[0] = d[0], i
        for j, h in enumerate(hyp, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r != h))
    return d[-1] / len(ref)


@torch.inference_mode()
def transcribe(model, processor, audio: np.ndarray) -> tuple[str, str]:
    """(language code, text). Clips longer than 30 s are transcribed per window."""
    texts, lang = [], "?"
    for start in range(0, max(len(audio), 1), WINDOW_S * SR):
        chunk = audio[start : start + WINDOW_S * SR]
        feats = processor(chunk, sampling_rate=SR, return_tensors="pt").input_features
        feats = feats.to(model.device, model.dtype)
        if lang == "?":
            # generate() strips the language token from its output, so ask for it.
            token = processor.tokenizer.convert_ids_to_tokens(model.detect_language(feats).tolist())[0]
            lang = token.strip("<|>")
        ids = model.generate(feats, task="transcribe")
        texts.append(processor.batch_decode(ids, skip_special_tokens=True)[0].strip())
    return lang, " ".join(t for t in texts if t)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("videos", nargs="+", type=Path)
    ap.add_argument("--json", action="store_true", help="one JSON object per video")
    args = ap.parse_args()

    # A GPU when there is one; on a login node's CPUs a 5 s clip takes seconds.
    device, dtype = ("cuda", torch.float16) if torch.cuda.is_available() else ("cpu", torch.float32)
    processor = WhisperProcessor.from_pretrained(WHISPER)
    model = WhisperForConditionalGeneration.from_pretrained(WHISPER, dtype=dtype).to(device)

    for video in args.videos:
        prompt_file = video.with_suffix(".prompt.txt")
        prompt = prompt_file.read_text() if prompt_file.exists() else ""
        ref = expected_dialogue(prompt)
        lang, hyp = transcribe(model, processor, load_audio(video))
        score = wer(words(ref), words(hyp))
        if args.json:
            print(json.dumps({"video": str(video), "language": lang, "wer": score,
                              "expected": ref, "transcript": hyp}, ensure_ascii=False))
        else:
            print(f"{video.name}\n  language  : {lang}\n  expected  : {ref}\n"
                  f"  transcript: {hyp}\n  WER       : {score:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
