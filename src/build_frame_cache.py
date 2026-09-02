#!/usr/bin/env python
"""Decode the H.265 source videos once into a reusable JPEG frame cache.

Decoding is the expensive, repeated cost in this challenge: every prompt tweak
would otherwise re-decode 1080p HEVC. We pay it once here. On Apple silicon
ffmpeg's videotoolbox does the HEVC decode in hardware, which is what makes the
Mac a genuinely useful worker for this stage even though it's useless for
training.

Output layout (one directory per video, zero-padded frame names so glob sorts
lexicographically == temporally):

    cache/frames/<track>/<video_id>/000000.jpg
                                    000001.jpg
                                    ...
    cache/frames/<track>/<video_id>/.done      <- resume marker

Resumable: a video with a .done marker is skipped, so an interrupted run costs
nothing to restart.

Usage:
    python build_frame_cache.py --track egoconv --fps 1 --height 448
    python build_frame_cache.py --track egoproactive --fps 2 --height 448 --jobs 6
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VIDEO_SUFFIXES = {".mp4", ".mkv", ".mov", ".webm"}


def find_videos(track: str) -> list[Path]:
    src = ROOT / "data" / track
    if not src.exists():
        sys.exit(f"no data at {src} — run scripts/download.sh first")
    return sorted(p for p in src.rglob("*") if p.suffix.lower() in VIDEO_SUFFIXES)


def decode_one(args: tuple[Path, Path, float, int, int]) -> tuple[str, bool, str]:
    """Decode a single video to JPEGs. Returns (video_id, ok, message)."""
    video, out_dir, fps, height, quality = args
    marker = out_dir / ".done"
    if marker.exists():
        return video.stem, True, "skipped (cached)"

    # Decode into a temp dir and rename on success, so an interrupted run never
    # leaves a half-populated directory that looks complete.
    tmp_dir = out_dir.with_name(out_dir.name + ".partial")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
        "-hwaccel", "videotoolbox",
        "-i", str(video),
        # fps first, then scale: downsample the frame rate before paying for any
        # scaling work on frames we're about to throw away.
        "-vf", f"fps={fps},scale=-2:{height}:flags=bilinear",
        "-q:v", str(quality),
        "-an", "-sn",
        str(tmp_dir / "%06d.jpg"),
    ]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return video.stem, False, (proc.stderr or "ffmpeg failed").strip()[:300]

    n = len(list(tmp_dir.glob("*.jpg")))
    if n == 0:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return video.stem, False, "produced 0 frames"

    shutil.rmtree(out_dir, ignore_errors=True)
    tmp_dir.rename(out_dir)
    marker.touch()
    return video.stem, True, f"{n} frames in {time.perf_counter() - t0:.1f}s"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", required=True, choices=["egoconv", "egoproactive"])
    ap.add_argument("--fps", type=float, default=1.0,
                    help="sampled frames per second (1.0 is a sane default for VLMs)")
    ap.add_argument("--height", type=int, default=448,
                    help="output height in px; width follows aspect ratio")
    ap.add_argument("--quality", type=int, default=3,
                    help="ffmpeg -q:v, 2=best 31=worst")
    ap.add_argument("--jobs", type=int, default=4,
                    help="parallel ffmpeg processes; videotoolbox has limited "
                         "concurrent decode sessions, so >6 tends not to help")
    ap.add_argument("--limit", type=int, default=0,
                    help="decode only the first N videos (for a timing trial)")
    args = ap.parse_args()

    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg not found")

    videos = find_videos(args.track)
    if args.limit:
        videos = videos[: args.limit]
    if not videos:
        sys.exit(f"no video files under data/{args.track}")

    out_root = ROOT / "cache" / "frames" / args.track
    out_root.mkdir(parents=True, exist_ok=True)

    tasks = [
        (v, out_root / v.stem, args.fps, args.height, args.quality)
        for v in videos
    ]

    print(f"{len(tasks)} videos -> {out_root}  "
          f"(fps={args.fps} height={args.height} jobs={args.jobs})\n")

    ok = failed = 0
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(decode_one, t): t[0].stem for t in tasks}
        for i, fut in enumerate(as_completed(futures), 1):
            vid, good, msg = fut.result()
            ok += good
            failed += not good
            status = "ok " if good else "FAIL"
            print(f"[{i}/{len(tasks)}] {status} {vid}: {msg}", flush=True)

    elapsed = time.perf_counter() - t0
    print(f"\ndone: {ok} ok, {failed} failed, {elapsed / 60:.1f} min")
    if ok:
        size = sum(f.stat().st_size for f in out_root.rglob("*.jpg"))
        print(f"cache size: {size / 1e9:.1f} GB")
        print(f"throughput: {elapsed / ok:.1f}s per video")


if __name__ == "__main__":
    main()
