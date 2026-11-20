"""QA gate: never assume a generated asset is good.

Two independent checks:

* technical - measured with ffprobe/ffmpeg. Objective, cheap, always runs.
* creative  - a vision model looks at sampled frames against the brief. Subjective, so the
  model only *observes and scores*; the pass/fail decision is made here with fixed
  thresholds, which keeps the gate stable and testable.
"""

from __future__ import annotations

from pathlib import Path

from .config import Settings
from .errors import MediaError, StageError
from .llm import LLM
from .media import decode_errors, extract_frame, leading_black_seconds, probe
from .models import (
    Brief,
    CreativeQA,
    CreativeScores,
    QAReport,
    Storyboard,
    Strategy,
    TechnicalQA,
)

SCORE_FIELDS = (
    "realism", "visual_consistency", "brief_adherence", "hook_strength", "advertising_quality"
)  # fmt: skip

REVIEWER = """You are a sceptical creative QA reviewer for paid social ads. You are shown
still frames sampled from a finished AI-generated vertical ad, each labelled with its
timestamp. The first frames are sampled densely because the opening decides whether a
viewer keeps watching.

Work in this order:
1. Describe what is literally visible in each frame. Do not infer from the brief.
2. List defects you can point to in a specific frame: warped hands or faces, garbled or
   misspelt text, the protagonist changing appearance between shots, impossible objects,
   plastic skin, mismatched colour grade.
3. Only then score each dimension from 1 to 10.

Scoring anchors: 9-10 would pass for professionally shot footage; 7-8 is clearly usable in
a paid campaign with minor flaws; 5-6 is noticeably AI or off-brief; below 5 is unusable.
A normal viewer spotting one obvious AI artifact caps realism at 6. Be strict: a wrongly
approved ad costs media spend, a wrongly rejected one costs a re-render.

You only see stills, so do not credit or penalise motion or audio."""


def technical_qa(video: Path, brief: Brief) -> TechnicalQA:
    report = TechnicalQA()
    if not video.exists() or video.stat().st_size == 0:
        report.failures.append("video file does not exist or is empty")
        return report
    report.exists = True

    try:
        info = probe(video)
    except MediaError as e:
        report.failures.append(e.message)
        return report
    broken = decode_errors(video)
    report.playable = not broken
    if broken:
        report.failures.append(f"not fully decodable: {broken}")

    report.duration = round(info.duration, 2)
    report.resolution = f"{info.width}x{info.height}"
    report.fps = info.fps
    report.video_codec = info.video_codec
    report.has_audio = info.has_audio

    want_w, want_h = (int(x) for x in brief.aspect_ratio.split(":"))
    ratio = info.width / info.height
    if abs(ratio - want_w / want_h) <= 0.01:
        report.aspect_ratio = brief.aspect_ratio
    else:
        report.aspect_ratio = f"{ratio:.3f}:1"
        report.failures.append(f"aspect ratio is {ratio:.3f}, expected {brief.aspect_ratio}")

    # Small tolerance: container durations land a frame or two either side of the target.
    if not brief.min_seconds - 0.1 <= info.duration <= brief.max_seconds + 0.1:
        report.failures.append(
            f"duration {info.duration:.2f}s is outside {brief.min_seconds}-{brief.max_seconds}s"
        )
    if min(info.width, info.height) < 720:
        report.failures.append(f"resolution {report.resolution} is below 720p")
    elif min(info.width, info.height) < 1080:
        report.warnings.append(f"resolution {report.resolution} is below the 1080x1920 target")

    if info.video_codec != "h264" or info.pix_fmt != "yuv420p":
        report.warnings.append(
            f"{info.video_codec}/{info.pix_fmt} may not play everywhere; platforms expect "
            "h264/yuv420p"
        )
    if not info.has_audio:
        report.warnings.append("no audio track")
    black = leading_black_seconds(video)
    if black >= 0.3:
        report.warnings.append(f"opens on {black:.1f}s of black, which weakens the hook")

    report.valid = not report.failures
    return report


def sample_times(duration: float) -> list[float]:
    """Dense in the first two seconds (the hook), then evenly spaced to the end."""
    hook = [t for t in (0.1, 0.8, 1.8) if t < duration]
    rest_start, rest_end = 3.0, max(duration - 0.4, 3.0)
    steps = 5
    rest = [rest_start + i * (rest_end - rest_start) / (steps - 1) for i in range(steps)]
    return hook + [round(t, 2) for t in rest if t < duration]


def creative_qa(
    llm: LLM,
    video: Path,
    brief: Brief,
    strategy: Strategy,
    storyboard: Storyboard,
    frames_dir: Path,
    settings: Settings,
) -> CreativeQA:
    if not llm.supports_vision:
        return CreativeQA(skipped_reason="No vision model available (offline mode).")
    try:
        duration = probe(video).duration
        frames = [
            (t, extract_frame(video, t, frames_dir / f"frame_{i:02d}_{t:05.2f}s.jpg"))
            for i, t in enumerate(sample_times(duration))
        ]
    except MediaError as e:
        return CreativeQA(skipped_reason=f"Could not sample frames: {e.message}")

    angle = strategy.selected
    user = (
        f"BRAND: {brief.brand} - {brief.product}\n"
        f"AUDIENCE: {brief.target_audience}\n"
        f"CORE MESSAGE: {brief.core_message}\n"
        f"BRAND FEEL: {', '.join(brief.brand_feel)}\n"
        f"INTENDED CONCEPT: {angle.concept}\n"
        f"INTENDED HOOK: {angle.hook}\n"
        f"INTENDED PROTAGONIST: {storyboard.protagonist}\n"
        f"The opening caption and closing brand card were added in post-production; judge "
        f"whether they are legible and well placed.\n\n"
        f"Review the {len(frames)} frames above and fill in the report."
    )
    try:
        scores = llm.structured(
            system=REVIEWER,
            user=user,
            schema=CreativeScores,
            images=[(f"Frame at {t:.1f}s of {duration:.1f}s", path) for t, path in frames],
            effort="medium",
        )
    except StageError as e:
        # A reviewer outage must not discard a finished render: report it and let a human look.
        return CreativeQA(
            skipped_reason=f"Creative review failed: {e.message}",
            frames=[str(p) for _, p in frames],
        )
    return gate(scores, settings, [str(p) for _, p in frames])


def gate(scores: CreativeScores, settings: Settings, frames: list[str] | None = None) -> CreativeQA:
    """Turn model scores into a decision. The model never gets to say 'passed'."""
    values = {name: getattr(scores, name) for name in SCORE_FIELDS}
    mean = sum(values.values()) / len(values)
    failures = [
        f"{name} scored {value} (minimum {settings.qa_min_dimension})"
        for name, value in values.items()
        if value < settings.qa_min_dimension
    ]
    if mean < settings.qa_min_mean:
        failures.append(f"mean score {mean:.1f} is below {settings.qa_min_mean}")
    return CreativeQA(
        scores=scores,
        mean_score=round(mean, 2),
        passed=not failures,
        failures=failures,
        frames=frames or [],
    )


def evaluate(
    llm: LLM,
    video: Path,
    brief: Brief,
    strategy: Strategy,
    storyboard: Storyboard,
    frames_dir: Path,
    settings: Settings,
) -> QAReport:
    technical = technical_qa(video, brief)
    if not technical.playable:
        creative = CreativeQA(skipped_reason="Video is not playable; nothing to review.")
    else:
        creative = creative_qa(llm, video, brief, strategy, storyboard, frames_dir, settings)
    # A skipped creative review is not a pass: the job ends in NEEDS_REVIEW, not FINALIZED.
    passed = technical.valid and creative.passed is True
    return QAReport(technical=technical, creative=creative, passed=passed)
