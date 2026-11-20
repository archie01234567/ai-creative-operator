"""ffmpeg / ffprobe / Pillow helpers: probing, frame grabs, and final assembly.

All legible text in the ad (hook caption, brand card) is rendered here, not by the video
model. Video models still garble lettering, and a misspelt brand name is the fastest way
for an ad to look fake.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

from .errors import MediaError

FONT_CANDIDATES = [
    "C:/Windows/Fonts/seguisb.ttf",
    "C:/Windows/Fonts/segoeuib.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
]


def _run(cmd: list[str], what: str) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    except FileNotFoundError as e:
        raise MediaError(
            f"{cmd[0]} is not installed or not on PATH (needed to {what}).",
            hint="Install ffmpeg: https://ffmpeg.org/download.html",
        ) from e


# ---------------------------------------------------------------------- probing


@dataclass
class Probe:
    duration: float
    width: int
    height: int
    fps: float
    video_codec: str
    pix_fmt: str
    has_audio: bool


def probe(path: Path) -> Probe:
    result = _run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams",
         str(path)],
        "inspect video",
    )  # fmt: skip
    if result.returncode != 0:
        raise MediaError(f"ffprobe could not read {path.name}: {result.stderr.strip()[:300]}")
    data = json.loads(result.stdout or "{}")
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise MediaError(f"{path.name} has no video stream")
    num, _, den = video.get("avg_frame_rate", "0/1").partition("/")
    fps = float(num) / float(den) if float(den or 0) else 0.0
    duration = float(data.get("format", {}).get("duration") or video.get("duration") or 0)
    width, height = int(video["width"]), int(video["height"])
    # Phone footage is often stored landscape with a 90-degree rotation flag.
    rotation = next(
        (abs(int(sd.get("rotation", 0))) for sd in video.get("side_data_list", [])
         if "rotation" in sd),
        0,
    )  # fmt: skip
    if rotation in (90, 270):
        width, height = height, width
    return Probe(
        duration=duration,
        width=width,
        height=height,
        fps=round(fps, 3),
        video_codec=video.get("codec_name", "unknown"),
        pix_fmt=video.get("pix_fmt", "unknown"),
        has_audio=any(s.get("codec_type") == "audio" for s in streams),
    )


def decode_errors(path: Path) -> str:
    """Decode every frame; return ffmpeg's error output ('' means fully playable)."""
    result = _run(
        ["ffmpeg", "-v", "error", "-xerror", "-i", str(path), "-f", "null", "-"], "decode video"
    )
    if result.returncode != 0:
        return result.stderr.strip()[:300] or "decoder exited with an error"
    return ""


def leading_black_seconds(path: Path) -> float:
    """Length of any black run starting at t=0 (a black opening kills the hook)."""
    result = _run(
        ["ffmpeg", "-v", "info", "-t", "3", "-i", str(path), "-vf", "blackdetect=d=0.2:pix_th=0.10",
         "-an", "-f", "null", "-"],
        "detect black frames",
    )  # fmt: skip
    for line in result.stderr.splitlines():
        if "black_start:0 " in line or "black_start:0.0" in line:
            try:
                return float(line.split("black_duration:")[1].split()[0])
            except (IndexError, ValueError):
                return 0.0
    return 0.0


def extract_frame(path: Path, at: float, out: Path, width: int = 540) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    result = _run(
        ["ffmpeg", "-y", "-v", "error", "-ss", f"{at:.2f}", "-i", str(path), "-frames:v", "1",
         "-vf", f"scale={width}:-2", "-q:v", "3", str(out)],
        "extract a frame",
    )  # fmt: skip
    if result.returncode != 0 or not out.exists():
        raise MediaError(f"Could not extract a frame at {at:.2f}s: {result.stderr.strip()[:200]}")
    return out


# ---------------------------------------------------------------------- overlays


def _font(size: int, font_path: str | None) -> ImageFont.FreeTypeFont:
    for candidate in [font_path, *FONT_CANDIDATES]:
        if candidate and Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default(size=size)


def _wrap(draw: ImageDraw.ImageDraw, text: str, font, max_width: int) -> list[str]:
    """Wrap to the fewest lines, then balance two-line text so no word is left orphaned.

    A break after sentence punctuation is preferred ("Flight cancelled. / She's fine.").
    """
    words = text.split()
    if draw.textlength(text, font=font) <= max_width or len(words) < 2:
        return [text]
    best: tuple[float, list[str]] | None = None
    for i in range(1, len(words)):
        first, second = " ".join(words[:i]), " ".join(words[i:])
        widest = max(draw.textlength(first, font=font), draw.textlength(second, font=font))
        if widest > max_width:
            continue
        cost = widest * (0.8 if first[-1] in ".,!?:;" else 1.0)
        if best is None or cost < best[0]:
            best = (cost, [first, second])
    if best:
        return best[1]
    lines, current = [], ""  # three or more lines: plain greedy fill
    for word in words:
        trial = f"{current} {word}".strip()
        if current and draw.textlength(trial, font=font) > max_width:
            lines.append(current)
            current = word
        else:
            current = trial
    return lines + [current]


def _text_block(img: Image.Image, lines: list[str], font, top: int, fill, gap: float = 1.18) -> int:
    """Centered text with a soft shadow so it stays legible over any footage."""
    w, _ = img.size
    line_h = int(font.size * gap)
    shadow = Image.new("RGBA", img.size, (0, 0, 0, 0))
    sdraw = ImageDraw.Draw(shadow)
    for i, line in enumerate(lines):
        x = (w - sdraw.textlength(line, font=font)) / 2
        sdraw.text((x, top + i * line_h + 3), line, font=font, fill=(0, 0, 0, 170))
    img.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(10)))
    draw = ImageDraw.Draw(img)
    for i, line in enumerate(lines):
        x = (w - draw.textlength(line, font=font)) / 2
        draw.text((x, top + i * line_h), line, font=font, fill=fill)
    return top + len(lines) * line_h


def render_hook_super(text: str, out: Path, size: tuple[int, int], font_path: str | None) -> Path:
    """Opening caption, placed below the platform's top UI and above the centre of frame."""
    w, h = size
    # A soft dark gradient over the top 38% keeps white type readable on bright footage
    # (windows, sky) without boxing it in.
    scrim = Image.new("RGBA", (1, h), (0, 0, 0, 0))
    for y in range(int(h * 0.38)):
        scrim.putpixel((0, y), (6, 12, 24, int(215 * (1 - y / (h * 0.38)) ** 0.55)))
    img = scrim.resize(size)
    font = _font(int(w * 0.082), font_path)
    lines = _wrap(ImageDraw.Draw(img), text, font, int(w * 0.84))
    _text_block(img, lines, font, int(h * 0.15), (255, 255, 255, 255))
    img.save(out)
    return out


def render_end_card(
    brand: str, headline: str, cta: str, out: Path, size: tuple[int, int], font_path: str | None
) -> Path:
    """Closing brand card: a navy scrim over the last shot, wordmark, line and CTA pill.

    Everything sits between 30% and 68% of the height, clear of the caption/UI zone that
    Reels and TikTok draw over the bottom of the frame.
    """
    w, h = size
    scrim = Image.new("RGBA", (1, h))
    for y in range(h):  # heavier towards the middle band where the text sits
        t = 1 - abs((y / h) - 0.5) * 1.3
        scrim.putpixel((0, y), (8, 18, 38, int(185 * max(0.45, min(1.0, t)))))
    img = scrim.resize(size)
    draw = ImageDraw.Draw(img)

    brand_font = _font(int(w * 0.118), font_path)
    y = _text_block(img, [brand], brand_font, int(h * 0.33), (255, 255, 255, 255))
    rule_w = int(w * 0.12)
    draw.rectangle([(w - rule_w) // 2, y + 22, (w + rule_w) // 2, y + 28], fill=(120, 200, 255, 255))

    head_font = _font(int(w * 0.056), font_path)
    lines = _wrap(draw, headline, head_font, int(w * 0.8))
    y = _text_block(img, lines, head_font, y + 70, (232, 240, 250, 255))

    cta_font = _font(int(w * 0.048), font_path)
    text_w = draw.textlength(cta, font=cta_font)
    pad_x, pill_h = int(w * 0.07), int(cta_font.size * 2.1)
    left, top = (w - text_w) / 2 - pad_x, y + 70
    draw.rounded_rectangle(
        [left, top, left + text_w + 2 * pad_x, top + pill_h], radius=pill_h // 2,
        fill=(255, 255, 255, 255),
    )  # fmt: skip
    draw.text(
        (w / 2, top + pill_h / 2), cta, font=cta_font, fill=(8, 18, 38, 255), anchor="mm"
    )
    img.save(out)
    return out


# ---------------------------------------------------------------------- assembly


@dataclass
class Overlay:
    image: Path
    start: float
    end: float
    fade: float = 0.35
    fade_in: float | None = None  # defaults to `fade`; 0 means fully visible from frame one


def assemble(
    clips: list[tuple[Path, float]],
    out: Path,
    overlays: list[Overlay],
    size: tuple[int, int] = (1080, 1920),
    fps: int = 30,
) -> Path:
    """Normalise, trim and join clips, composite overlays, encode a social-ready H.264 mp4.

    `clips` is [(path, planned_seconds)]. Each clip is cropped to fill the frame (never
    letterboxed) and clips without audio get silence so the join cannot fail on a
    missing stream.
    """
    if not clips:
        raise MediaError("Nothing to assemble: no clips were generated.")
    w, h = size
    inputs: list[str] = []
    filters: list[str] = []
    total = 0.0
    for i, (path, planned) in enumerate(clips):
        info = probe(path)
        seconds = min(planned, info.duration)
        total += seconds
        inputs += ["-i", str(path)]
        filters.append(
            f"[{i}:v]trim=duration={seconds:.3f},setpts=PTS-STARTPTS,"
            f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},"
            f"fps={fps},format=yuv420p,setsar=1[v{i}]"
        )
        if info.has_audio:
            filters.append(
                f"[{i}:a]atrim=duration={seconds:.3f},asetpts=PTS-STARTPTS,"
                f"aresample=48000,aformat=channel_layouts=stereo,apad,"
                f"atrim=duration={seconds:.3f}[a{i}]"
            )
        else:
            filters.append(f"anullsrc=r=48000:cl=stereo,atrim=duration={seconds:.3f}[a{i}]")

    n = len(clips)
    filters.append("".join(f"[v{i}][a{i}]" for i in range(n)) + f"concat=n={n}:v=1:a=1[vcat][acat]")

    last = "vcat"
    for j, overlay in enumerate(overlays):
        index = n + j
        end = min(overlay.end, total)
        inputs += ["-loop", "1", "-t", f"{total:.3f}", "-i", str(overlay.image)]
        fade_in = overlay.fade if overlay.fade_in is None else overlay.fade_in
        fades = f"fade=t=in:st={overlay.start:.2f}:d={fade_in}:alpha=1" if fade_in > 0 else "null"
        if end < total - 0.05:  # an overlay that runs to the last frame should not fade out
            fades += f",fade=t=out:st={end - overlay.fade:.2f}:d={overlay.fade}:alpha=1"
        filters.append(f"[{index}:v]format=rgba,{fades}[ov{j}]")
        filters.append(
            f"[{last}][ov{j}]overlay=0:0:enable='between(t,{overlay.start:.2f},{end:.2f})'[vo{j}]"
        )
        last = f"vo{j}"
    filters.append(f"[acat]afade=t=out:st={max(total - 0.6, 0):.2f}:d=0.6[aout]")

    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-v", "error", *inputs,
        "-filter_complex", ";".join(filters),
        "-map", f"[{last}]", "-map", "[aout]",
        "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
        "-r", str(fps), "-c:a", "aac", "-b:a", "160k", "-t", f"{total:.3f}",
        "-movflags", "+faststart", str(out),
    ]  # fmt: skip
    result = _run(cmd, "assemble the final video")
    if result.returncode != 0 or not out.exists():
        raise MediaError(f"ffmpeg assembly failed: {result.stderr.strip()[-500:]}")
    return out
