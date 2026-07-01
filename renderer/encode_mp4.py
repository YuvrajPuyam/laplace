"""encode_mp4 — MAIN env: PNG frame sequence -> MP4.

Runs after build_stage --animate writes frames. Kept in the main env (not the
Isaac venv) so encoding deps stay off the Isaac side. Uses imageio + the
bundled ffmpeg.

  python -m renderer.encode_mp4 --frames-dir renderer/out/frames `
      --out renderer/out/braess_motion.mp4 --fps 30
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="encode_mp4")
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--pattern", default="frame_*.png")
    args = ap.parse_args(argv)

    import imageio.v2 as imageio

    frames = sorted(Path(args.frames_dir).glob(args.pattern))
    if not frames:
        raise SystemExit(f"no frames matching {args.pattern} in {args.frames_dir}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(args.out, fps=args.fps, codec="libx264",
                                quality=8, macro_block_size=8)
    for fp in frames:
        writer.append_data(imageio.imread(fp))
    writer.close()
    secs = len(frames) / args.fps
    print(f"encoded {len(frames)} frames -> {args.out} "
          f"({secs:.1f}s @ {args.fps}fps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
