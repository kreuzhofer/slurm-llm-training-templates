"""cut_ref.py -- a reference picture and voice clip from a rendered video.

    python cut_ref.py $TEMPLATES_DIR/output/minimax-h3/h3_1234.mp4 $HOME/bakery/baker \
        [--frame 0] [--start 0] [--length 7]

Writes baker.png (the frame --frame) and baker-voice.wav (--length seconds of
the audio from --start on, 32 kHz stereo) for REFS. Pick a frame that shows
the face clearly and a span in which only that person speaks; check_lines.py
shows where the line starts and ends.

H3 takes audio references of 2 to 15 s, 15 s in total per video, so two
voices of 7 s fit into one clip; three do not.
"""

import argparse
import sys

import av
import numpy as np

SR = 32000  # H3's own audio rate


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("video")
    ap.add_argument("out", help="output path without extension, e.g. $HOME/bakery/baker")
    ap.add_argument("--frame", type=int, default=0, help="frame number for the picture (24 per second)")
    ap.add_argument("--start", type=float, default=0.0, help="start of the voice clip, seconds")
    ap.add_argument("--length", type=float, default=7.0, help="length of the voice clip, seconds")
    args = ap.parse_args()
    if not 2 <= args.length <= 15:
        raise SystemExit("--length must be 2 to 15 seconds")

    with av.open(args.video) as c:
        for i, frame in enumerate(c.decode(video=0)):
            if i == args.frame:
                frame.to_image().save(f"{args.out}.png")
                break
        else:
            raise SystemExit(f"{args.video} has only {i + 1} frames")

    resampler = av.AudioResampler(format="s16", layout="stereo", rate=SR)
    chunks = []
    with av.open(args.video) as c:
        for frame in c.decode(audio=0):
            chunks += [f.to_ndarray().reshape(-1, 2) for f in resampler.resample(frame)]
    chunks += [f.to_ndarray().reshape(-1, 2) for f in resampler.resample(None)]
    audio = np.concatenate(chunks)
    clip = audio[int(args.start * SR) : int((args.start + args.length) * SR)]
    if len(clip) < args.length * SR - 1:
        raise SystemExit(f"{args.video} has {len(audio) / SR:.2f} s of audio; "
                         f"{args.start:g} + {args.length:g} s does not fit")

    with av.open(f"{args.out}-voice.wav", mode="w") as out:
        stream = out.add_stream("pcm_s16le", rate=SR, layout="stereo")
        frame = av.AudioFrame.from_ndarray(np.ascontiguousarray(clip.reshape(1, -1)), format="s16", layout="stereo")
        frame.sample_rate = SR
        for packet in stream.encode(frame):
            out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)

    print(f"{args.out}.png: frame {args.frame}; {args.out}-voice.wav: {args.start:g} to {args.start + args.length:g} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
