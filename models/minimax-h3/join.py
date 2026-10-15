"""join.py -- cut H3 clips together into one video, in the order given.

    python join.py $HOME/bakery/scene.mp4 h3_1234.mp4 h3_1235.mp4 ...

Writes H.264 with AAC stereo at 24 fps, like the clips. Each clip's audio is
trimmed or padded to exactly the length of its picture, so sound and picture
stay in sync across the cuts. All clips must have the same size.
"""

import sys
from fractions import Fraction

import av
import numpy as np

FPS, SR = 24, 32000  # H3's frame and audio rates


def read(path):
    """(frames as RGB arrays, audio as planar stereo float32 of the same length)."""
    with av.open(path) as c:
        frames = [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]
    resampler = av.AudioResampler(format="fltp", layout="stereo", rate=SR)
    chunks = []
    with av.open(path) as c:
        for f in c.decode(audio=0):
            chunks += [g.to_ndarray() for g in resampler.resample(f)]
    chunks += [g.to_ndarray() for g in resampler.resample(None)]
    audio = np.concatenate(chunks, axis=1)
    n = round(len(frames) / FPS * SR)
    audio = audio[:, :n] if audio.shape[1] >= n else np.pad(audio, ((0, 0), (0, n - audio.shape[1])))
    return frames, audio


def main() -> int:
    if len(sys.argv) < 3:
        raise SystemExit("usage: python join.py OUT.mp4 clip1.mp4 clip2.mp4 ...")
    out_path, clips = sys.argv[1], sys.argv[2:]
    sizes = {}
    for path in clips:
        with av.open(path) as c:
            sizes[path] = (c.streams.video[0].width, c.streams.video[0].height)
    if len(set(sizes.values())) > 1:
        raise SystemExit("clips differ in size: " + ", ".join(f"{p} {w}x{h}" for p, (w, h) in sizes.items()))

    out = av.open(out_path, mode="w")
    video = audio_stream = None
    fifo, apts, total = av.AudioFifo(), 0, 0
    for path in clips:
        frames, audio = read(path)
        h, w, _ = frames[0].shape
        if video is None:
            video = out.add_stream("libx264", rate=FPS)
            video.width, video.height, video.pix_fmt = w, h, "yuv420p"
            video.options = {"crf": "18"}
            audio_stream = out.add_stream("aac", rate=SR)
            audio_stream.layout = "stereo"
        for arr in frames:
            for packet in video.encode(av.VideoFrame.from_ndarray(arr, format="rgb24")):
                out.mux(packet)
        frame = av.AudioFrame.from_ndarray(np.ascontiguousarray(audio, dtype=np.float32), format="fltp", layout="stereo")
        frame.sample_rate = SR
        fifo.write(frame)
        while fifo.samples >= 1024:  # the AAC encoder takes 1024 samples per frame
            chunk = fifo.read(1024)
            chunk.pts, chunk.time_base = apts, Fraction(1, SR)
            apts += 1024
            for packet in audio_stream.encode(chunk):
                out.mux(packet)
        total += len(frames)
        print(f"{path}: {len(frames) / FPS:.2f} s")

    if fifo.samples:
        chunk = fifo.read()
        chunk.pts, chunk.time_base = apts, Fraction(1, SR)
        for packet in audio_stream.encode(chunk):
            out.mux(packet)
    for packet in video.encode():
        out.mux(packet)
    for packet in audio_stream.encode():
        out.mux(packet)
    out.close()
    print(f"{out_path}: {total / FPS:.2f} s, {video.width}x{video.height}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
