#!/usr/bin/env python3
"""Pull frames from a YouTube billiards stream for the perception pipeline.

Downloads a short section with yt-dlp (which handles YouTube auth; passing the
raw stream URL to ffmpeg 403s), then extracts frames with ffmpeg.

    # single frame at 41:40
    python3 ingest.py https://youtu.be/5ve7BPFqrls --at 2500 --out frame.png

    # a shot sequence: 2500.0s–2506.0s at 30 fps, for tracking
    python3 ingest.py https://youtu.be/5ve7BPFqrls --clip 2500 2506 --fps 30 --out-dir frames/
"""

import argparse
import pathlib
import subprocess
import sys
import tempfile

# Progressive 360p (itag 18) is the most reliably downloadable; fall back to a
# muxed <=720p stream.
FORMAT = "18/22/bestvideo[height<=?720][ext=mp4]"


def download_section(url, start, end, dest):
    subprocess.run(
        ["python3", "-m", "yt_dlp", "-q", "-f", FORMAT,
         "--download-sections", f"*{start}-{end}", "--force-keyframes-at-cuts",
         "-o", str(dest), url],
        check=True,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("url")
    ap.add_argument("--at", type=float, help="timestamp (s) for a single frame")
    ap.add_argument("--clip", type=float, nargs=2, metavar=("START", "END"), help="section (s) for a sequence")
    ap.add_argument("--fps", type=float, default=30.0, help="frames/s to extract from a clip")
    ap.add_argument("--out", default="frame.png", help="output path for --at")
    ap.add_argument("--out-dir", default="frames", help="output dir for --clip")
    args = ap.parse_args()

    if (args.at is None) == (args.clip is None):
        ap.error("pass exactly one of --at or --clip")

    with tempfile.TemporaryDirectory() as tmp:
        clip = pathlib.Path(tmp) / "clip.mp4"
        if args.at is not None:
            download_section(args.url, args.at, args.at + 1.0, clip)
            subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(clip),
                            "-frames:v", "1", "-y", args.out], check=True)
            print(f"wrote {args.out}")
        else:
            start, end = args.clip
            download_section(args.url, start, end, clip)
            out_dir = pathlib.Path(args.out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(clip),
                            "-vf", f"fps={args.fps}", "-y", str(out_dir / "f_%04d.png")], check=True)
            n = len(list(out_dir.glob("f_*.png")))
            print(f"wrote {n} frames to {out_dir}/")


if __name__ == "__main__":
    sys.exit(main())
