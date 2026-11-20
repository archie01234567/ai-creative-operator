"""Storyboard -> provider requests. Deterministic: no model call happens here.

The storyboard is provider-neutral. This module adapts it to what the chosen provider can
do, which is the part that changes when a provider is swapped:

* single-take  - the whole spot fits in one generation, so it is requested as one clip with
  a timed shot list. One generation means one face, one grade and one continuous audio bed,
  which is the cheapest way to avoid "unrelated AI clips stitched together".
* per-scene    - the provider's clips are shorter than the spot, so each shot is its own
  generation. Shot 1 is rendered first and a frame from it is passed to the others as an
  identity reference.
"""

from __future__ import annotations

from .models import ClipTask, Storyboard
from .providers.base import Capabilities

CONSTRAINTS = (
    "Photorealistic live-action commercial footage, natural skin texture, anatomically correct "
    "hands and faces, physically plausible motion. No on-screen text, captions, subtitles, "
    "logos, watermarks or user-interface graphics anywhere in frame; any signage or screens "
    "are out of focus and unreadable. No spoken dialogue or voice-over."
)


def choose_mode(storyboard: Storyboard, caps: Capabilities, requested: str = "auto") -> str:
    if requested in ("single", "per_scene"):
        return requested
    return "single" if storyboard.total_seconds <= caps.max_clip_seconds else "per_scene"


def build_tasks(
    storyboard: Storyboard,
    caps: Capabilities,
    mode: str = "auto",
    feedback: list[str] | None = None,
) -> tuple[str, list[ClipTask]]:
    mode = choose_mode(storyboard, caps, mode)
    notes = _feedback_block(feedback or [])

    if mode == "single":
        prompt = _single_take_prompt(storyboard) + notes
        return mode, [ClipTask(index=0, prompt=prompt, duration=storyboard.total_seconds)]

    tasks = []
    for i, scene in enumerate(storyboard.scenes):
        reference = i > 0 and caps.supports_reference_images
        prompt = (
            f"Vertical 9:16 shot from a premium commercial, {scene.duration} seconds, one "
            f"continuous take.\n"
            f"LOOK: {storyboard.visual_style}\n"
            f"PROTAGONIST: {storyboard.protagonist}"
            + (
                " This is the same person shown in the reference image: keep their face, hair "
                "and wardrobe identical. Use the image for identity and grade only, not framing.\n"
                if reference
                else "\n"
            )
            + f"ACTION: {scene.video_prompt}\n"
            f"CAMERA: {scene.camera}\n"
            f"SOUND: {scene.audio}\n"
            f"{CONSTRAINTS}{notes}"
        )
        tasks.append(
            ClipTask(
                index=i,
                prompt=prompt,
                # Never request less than the provider can make; assembly trims to the plan.
                duration=min(max(scene.duration, caps.min_clip_seconds), caps.max_clip_seconds),
                uses_reference=reference,
            )
        )
    return mode, tasks


def scene_durations(storyboard: Storyboard, mode: str) -> list[int]:
    """Planned on-screen length of each generated clip, used when trimming in assembly."""
    if mode == "single":
        return [storyboard.total_seconds]
    return [s.duration for s in storyboard.scenes]


def _single_take_prompt(sb: Storyboard) -> str:
    lines = [
        f"A {sb.total_seconds}-second vertical 9:16 premium commercial made of "
        f"{len(sb.scenes)} shots joined by clean hard cuts. It follows one person through one "
        f"short story.",
        f"LOOK (identical in every shot): {sb.visual_style}",
        f"PROTAGONIST (the same person in every shot): {sb.protagonist}",
        "",
    ]
    start = 0
    for i, scene in enumerate(sb.scenes, 1):
        end = start + scene.duration
        lines.append(
            f"SHOT {i} ({start}s-{end}s): {scene.video_prompt} Camera: {scene.camera} "
            f"Sound: {scene.audio}"
        )
        start = end
    lines += [
        "",
        "The final shot holds on a calm, uncluttered composition until the very last frame.",
        CONSTRAINTS,
    ]
    return "\n".join(lines)


def _feedback_block(feedback: list[str]) -> str:
    if not feedback:
        return ""
    return (
        "\nNotes from the review of the previous render. Fix them in the footage itself; "
        "still render no text, captions or readable screens:\n- " + "\n- ".join(feedback)
    )
