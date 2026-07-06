#!/usr/bin/env python3
"""Record a live YouTube 3-cushion stream to disk, and keep it.

The MASA 4 / bilardo.com.tr feed is a continuous live tournament with a top-down
overhead inset (bottom-left). We record windows of it to `data/` so real matches
are kept on disk for tracking/reconstruction — never synthetic.

    # record 4 minutes of the live feed at 720p into data/masa4_live/
    python3 capture_live.py https://www.youtube.com/watch?v=X-CnEnG5hB4 --secs 240

Records at the live edge with stream-copy (no re-encode), so it's fast and keeps
broadcast quality. Extract frames / the inset afterward with ffmpeg or track.py.
"""

import argparse
import datetime as dt
import os
import subprocess


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("--secs", type=float, default=240.0, help="seconds to record")
    ap.add_argument("--fmt", default="95", help="yt-dlp format id (95=720p, 96=1080p, 93=360p)")
    ap.add_argument("--out-dir", default="data/masa4_live")
    ap.add_argument("--tag", default="masa4", help="filename prefix")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(args.out_dir, f"{args.tag}_{stamp}.mp4")

    m3u8 = subprocess.run(
        ["yt-dlp", "-q", "-g", "-f", args.fmt, args.url],
        check=True, capture_output=True, text=True,
    ).stdout.strip().splitlines()[0]

    print(f"recording {args.secs:.0f}s of {args.url} (fmt {args.fmt}) -> {out}")
    subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", m3u8,
         "-t", str(args.secs), "-c", "copy", "-y", out],
        check=True,
    )
    size_mb = os.path.getsize(out) / 1e6
    print(f"kept {out} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
