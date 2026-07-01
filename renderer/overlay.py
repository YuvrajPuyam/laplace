"""overlay — MAIN env: composite a metrics HUD + callout pointers onto frames.

Turns the raw Isaac PNG sequence into a self-explanatory demo: a live metrics
panel (throughput, p95 latency, robots carrying, shortcut occupancy) and
leader-line callouts pointing at scene elements ("Capacity-1 shortcut", "Pick
station", "Pack station"). World-space anchors are projected to screen pixels
using the camera in render_manifest.json — so the labels track the right spot.

All numbers come from hud.json (computed from the event log), so nothing is
faked. Pure post-process, no Isaac, no extra render cost.

  python -m renderer.overlay --frames-dir renderer/out/frames_work `
      --hud renderer/scenes/braess_tracks_hud.json `
      --out-dir renderer/out/frames_annotated
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFont

_VIGNETTE = {}


def _vignette(size):
    """Cached radial darkening mask (bright center -> dark corners)."""
    if size in _VIGNETTE:
        return _VIGNETTE[size]
    w, h = size
    cx, cy = w / 2.0, h / 2.0
    maxd = math.hypot(cx, cy)
    mask = Image.new("L", size)
    px = mask.load()
    for y in range(h):
        dy = (y - cy) ** 2
        for x in range(w):
            d = math.sqrt((x - cx) ** 2 + dy) / maxd
            px[x, y] = int(255 * (1.0 - 0.42 * d * d))
    rgb = mask.convert("RGB")
    _VIGNETTE[size] = rgb
    return rgb


def cinematic_grade(img):
    """Punch up a flat render: contrast, saturation, a touch warm, vignette."""
    img = ImageEnhance.Contrast(img).enhance(1.16)
    img = ImageEnhance.Color(img).enhance(1.22)
    img = ImageEnhance.Brightness(img).enhance(1.05)
    warm = Image.new("RGB", img.size, (255, 246, 228))
    img = Image.blend(img, ImageChops.multiply(img, warm), 0.10)
    img = ImageChops.multiply(img, _vignette(img.size))
    return img

# brand palette (matches the rest of the project's dark-on-light demo look)
INK = (24, 26, 32)
PANEL = (18, 20, 26, 205)
ACCENT = (226, 75, 74)       # shortcut red
INFO = (55, 138, 221)        # info blue
TEXT = (238, 240, 244)
MUTE = (170, 176, 188)
GOOD = (120, 190, 90)


def _font(size, bold=False):
    for name in (("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"),
                 "arialbd.ttf" if bold else "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _basis(eye, tgt):
    fwd = [tgt[i] - eye[i] for i in range(3)]
    fl = math.sqrt(sum(c * c for c in fwd)) or 1.0
    fwd = [c / fl for c in fwd]
    up = [0.0, 0.0, 1.0]
    if abs(sum(fwd[i] * up[i] for i in range(3))) > 0.999:
        up = [0.0, 1.0, 0.0]
    right = _norm(_cross(fwd, up))
    true_up = _cross(right, fwd)
    return right, true_up, fwd


def _cross(a, b):
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]]


def _norm(v):
    n = math.sqrt(sum(c * c for c in v)) or 1.0
    return [c / n for c in v]


def project(world, cam):
    """World xyz -> (px, py, visible). Pinhole from camera eye/target + FOV."""
    eye, tgt = cam["eye"], cam["target"]
    right, true_up, fwd = _basis(eye, tgt)
    d = [world[i] - eye[i] for i in range(3)]
    zc = sum(d[i] * fwd[i] for i in range(3))
    if zc <= 0.05:
        return 0, 0, False
    xc = sum(d[i] * right[i] for i in range(3))
    yc = sum(d[i] * true_up[i] for i in range(3))
    half_h = math.atan(cam["aperture_h"] / (2 * cam["focal_length"]))
    half_v = math.atan(cam["aperture_v"] / (2 * cam["focal_length"]))
    ndc_x = (xc / zc) / math.tan(half_h)
    ndc_y = (yc / zc) / math.tan(half_v)
    px = (ndc_x * 0.5 + 0.5) * cam["width"]
    py = (1.0 - (ndc_y * 0.5 + 0.5)) * cam["height"]
    on = -0.2 <= ndc_x <= 1.2 and -0.2 <= ndc_y <= 1.2
    return px, py, on


def _rrect(draw, box, radius, fill):
    draw.rounded_rectangle(box, radius=radius, fill=fill)


def draw_frame(img, cam, hud, i):
    W, H = img.size
    d = ImageDraw.Draw(img, "RGBA")
    f_title = _font(max(16, W // 45), bold=True)
    f_lbl = _font(max(12, W // 70), bold=True)
    f_big = _font(max(20, W // 38), bold=True)
    f_sm = _font(max(11, W // 80))

    fr = hud["frames"][i]

    # --- title strip -------------------------------------------------------
    d.rectangle([0, 0, W, max(34, H // 12)], fill=(14, 15, 19, 190))
    d.text((16, 8), "LAPLACE  ·  Warehouse Digital Twin", font=f_title, fill=TEXT)
    d.text((16, 8 + f_title.size + 2), hud.get("label", ""), font=f_sm, fill=MUTE)

    # --- metrics panel (bottom-left) --------------------------------------
    base = fr.get("baseline_throughput")
    thr_extra = ""
    thr_col = TEXT
    if base:
        delta = (fr["throughput"] - base) / base * 100.0
        thr_extra = f"  ▼{abs(delta):.0f}%" if delta < 0 else f"  ▲{delta:.0f}%"
        thr_col = ACCENT if delta < -1 else GOOD
    rows = [
        ("Sim time", f"{fr['t']:.1f} min", MUTE, None),
        ("Throughput", f"{fr['throughput']:.0f} /hr", TEXT, (thr_extra, thr_col)),
        ("p95 latency", f"{fr['p95_latency']:.1f} min", TEXT, None),
        ("Robots carrying", f"{fr['carrying']} / 9", GOOD, None),
        ("On capacity-1 edge", f"{fr['shortcut_occ']}", ACCENT if fr['shortcut_occ'] else MUTE, None),
    ]
    pw, ph = max(248, W // 4), 28 + len(rows) * 30
    x0, y0 = 16, H - ph - 16
    _rrect(d, [x0, y0, x0 + pw, y0 + ph], 10, PANEL)
    d.text((x0 + 14, y0 + 10), "LIVE METRICS  (from event log)", font=f_sm, fill=MUTE)
    yy = y0 + 32
    for name, val, col, extra in rows:
        d.text((x0 + 14, yy), name, font=f_sm, fill=MUTE)
        vw = d.textlength(val, font=f_lbl)
        ex_w = d.textlength(extra[0], font=f_sm) if extra else 0
        d.text((x0 + pw - 14 - vw - ex_w, yy - 2), val, font=f_lbl, fill=col)
        if extra:
            d.text((x0 + pw - 14 - ex_w, yy - 1), extra[0], font=f_sm, fill=extra[1])
        yy += 30

    # --- callout pointers (fanned offsets so they don't collide) ----------
    fan = [(64, 78), (-150, -66), (96, -86), (-150, 70), (96, 84)]
    for idx, a in enumerate(hud["anchors"]):
        px, py, on = project(a["xyz"], cam)
        if not on:
            continue
        col = ACCENT if a["kind"] == "alert" else INFO
        dx, dy = fan[idx % len(fan)]
        lx, ly = px + dx, py + dy
        label = a["label"]
        tw = d.textlength(label, font=f_lbl)
        d.line([(px, py), (lx, ly)], fill=col, width=2)
        d.ellipse([px - 5, py - 5, px + 5, py + 5], fill=col)
        if dx < 0:  # chip extends left from the leader end
            chip = [lx - tw - 20, ly - 14, lx + 2, ly + 14]
            tx = lx - tw - 10
        else:       # chip extends right
            chip = [lx - 2, ly - 14, lx + tw + 20, ly + 14]
            tx = lx + 10
        _rrect(d, chip, 7, (16, 18, 24, 230))
        bar = chip[0] if dx >= 0 else chip[2] - 4
        d.rectangle([bar, chip[1], bar + 4, chip[3]], fill=col)
        d.text((tx, ly - 8), label, font=f_lbl, fill=TEXT)

    return img


def main(argv=None):
    ap = argparse.ArgumentParser(prog="overlay")
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--hud", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--manifest", default=None,
                    help="render_manifest.json (default: <frames-dir>/render_manifest.json)")
    ap.add_argument("--no-grade", action="store_true", help="skip the cinematic grade")
    args = ap.parse_args(argv)

    cam = json.loads(Path(args.manifest or
                          Path(args.frames_dir) / "render_manifest.json")
                     .read_text(encoding="utf-8"))
    hud = json.loads(Path(args.hud).read_text(encoding="utf-8"))
    frames = sorted(Path(args.frames_dir).glob("frame_*.png"))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    n = min(len(frames), len(hud["frames"]))
    for i in range(n):
        img = Image.open(frames[i]).convert("RGB")
        if not args.no_grade:
            img = cinematic_grade(img)
        draw_frame(img, cam, hud, i)
        img.save(out / frames[i].name)
    print(f"annotated {n} frames -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
