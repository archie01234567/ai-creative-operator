"""Creative planning: brief -> angles + selection -> storyboard.

Structured outputs guarantee the *shape* of what comes back. The business rules (three
angles, a real selection, 10-15 seconds, a short opening shot) are checked here, and a
violation is sent back to the model once as a repair request before the job fails.
"""

from __future__ import annotations

from typing import Callable, TypeVar

from pydantic import BaseModel

from .errors import PlanningError
from .llm import LLM
from .models import Brief, Storyboard, Strategy

T = TypeVar("T", bound=BaseModel)

STRATEGIST = """You are a senior creative strategist for paid social (Instagram Reels, TikTok).

You are planning a 10-15 second vertical ad that will be produced entirely with an AI video
model and watched mostly with sound off by people who did not ask to see it.

Judge ideas the way a media buyer would: the first two seconds decide everything, one idea
per ad, the product's benefit must be visible rather than explained, and the story must
survive with no audio. Be honest about what AI video cannot do well today: legible text,
phone/app screens, logos, fine hand actions, crowds, split screens and any shot that needs
the same person twice in one frame all tend to break realism, so an idea that depends on
them is a weaker idea no matter how clever it is.

Propose angles that are genuinely different from each other (different emotion and different
opening image), score each one candidly, then pick one. Keep every field short."""

DIRECTOR = """You are a commercial director writing a shot list for an AI video model.

The spot is 9:16, photorealistic, and must read as one continuous piece of film: one
protagonist, one location logic, one look. Open on the most arresting image you have; the
first shot is at most 3 seconds and must work as a thumbnail. Tell the story with pictures:
problem, turn, relief. The brand resolves the tension in the final shot.

Rules for the video prompts:
- Describe what the camera sees, concretely: subject, action, setting, light, lens.
- Never ask for on-screen text, captions, readable signs, logos, UI or the brand name. The
  brand card and captions are added in post where they can be rendered perfectly.
- Keep actions simple and physically plausible for a few seconds of footage.
- One person, one frame: no split screens, composites, mirrors or duplicated subjects.
- The opening shot must show the problem in body language a stranger reads in one second
  (posture, face, gesture), not in a sign or a screen.
- The product cannot be filmed, so make its effect visible: the protagonist's situation
  changes on camera in the turn shot (she stops, reads her phone, her shoulders drop, she
  walks the other way). `benefit_super` is the short caption added in post over that shot
  that names what just happened for her; set `benefit_shot` to that shot's number.
- Wardrobe and props never change: nobody takes a coat off, gains a bag or loses one.
- Prefer slow, simple camera moves and keep the face sharp; fast motion smears.
- End on a calm, bright, uncluttered composition: the closing brand card is overlaid on it.
- `protagonist` and `visual_style` are pasted into every shot, so write them once, precisely."""


def _with_repair(call: Callable[[str], T], validate: Callable[[T], list[str]], what: str) -> T:
    result = call("")
    problems = validate(result)
    if not problems:
        return result
    repair = (
        "\n\nYour previous answer broke these rules. Return a corrected version:\n- "
        + "\n- ".join(problems)
    )
    result = call(repair)
    problems = validate(result)
    if problems:
        raise PlanningError(
            f"The {what} still violates the brief after one repair attempt: " + "; ".join(problems),
            hint="Re-run the job; if it repeats, loosen the brief constraints.",
        )
    return result


def _brief_block(brief: Brief) -> str:
    return (
        f"BRAND: {brief.brand}\n"
        f"PRODUCT: {brief.product}\n"
        f"AUDIENCE: {brief.target_audience}\n"
        f"CUSTOMER PROBLEM: {', '.join(brief.customer_problem)}\n"
        f"CORE MESSAGE: {brief.core_message}\n"
        f"BRAND FEEL: {', '.join(brief.brand_feel)}\n"
        f"FORMAT: {brief.aspect_ratio}, {brief.min_seconds}-{brief.max_seconds}s, "
        f"{' / '.join(brief.platforms)}"
    )


def validate_strategy(s: Strategy) -> list[str]:
    problems = []
    names = [a.name for a in s.angles]
    if len(s.angles) < 3:
        problems.append(f"need at least 3 angles, got {len(s.angles)}")
    if len(set(names)) != len(names):
        problems.append("angle names must be unique")
    if s.selected_angle not in names:
        problems.append(f"selected_angle '{s.selected_angle}' is not one of {names}")
    for a in s.angles:
        for field in ("hook_strength", "sound_off_clarity", "producibility"):
            if not 1 <= getattr(a, field) <= 10:
                problems.append(f"{a.name}.{field} must be 1-10")
    return problems


def validate_storyboard(sb: Storyboard, brief: Brief) -> list[str]:
    problems = []
    if not 2 <= len(sb.scenes) <= 5:
        problems.append(f"need 2-5 scenes, got {len(sb.scenes)}")
    if not brief.min_seconds <= sb.total_seconds <= brief.max_seconds:
        problems.append(
            f"scene durations sum to {sb.total_seconds}s; must be "
            f"{brief.min_seconds}-{brief.max_seconds}s"
        )
    for i, scene in enumerate(sb.scenes, 1):
        if scene.duration < 2:
            problems.append(f"scene {i} is {scene.duration}s; minimum is 2s")
        if brief.brand.lower() in scene.video_prompt.lower():
            problems.append(f"scene {i} video_prompt names the brand; video models garble text")
    if sb.scenes and sb.scenes[0].duration > 3:
        problems.append("the opening shot must be 3s or shorter")
    if len(sb.hook_super.split()) > 6:
        problems.append("hook_super must be 6 words or fewer")
    if not sb.benefit_super.strip():
        problems.append("benefit_super is required: the caption that names what the product did")
    elif len(sb.benefit_super.split()) > 6:
        problems.append("benefit_super must be 6 words or fewer")
    if not 2 <= sb.benefit_shot <= len(sb.scenes):
        problems.append("benefit_shot must be the number of the turn shot (2 or later)")
    if len(sb.end_card_headline.split()) > 7:
        problems.append("end_card_headline must be 7 words or fewer")
    if len(sb.end_card_cta.split()) > 4:
        problems.append("end_card_cta must be 4 words or fewer")
    return problems


def plan_strategy(llm: LLM, brief: Brief) -> Strategy:
    user = (
        f"{_brief_block(brief)}\n\n"
        "Propose at least three distinct creative angles, score them, and select one."
    )
    return _with_repair(
        lambda repair: llm.structured(system=STRATEGIST, user=user + repair, schema=Strategy),
        validate_strategy,
        "strategy",
    )


def plan_storyboard(llm: LLM, brief: Brief, strategy: Strategy) -> Storyboard:
    angle = strategy.selected
    user = (
        f"{_brief_block(brief)}\n\n"
        f"SELECTED ANGLE: {angle.name}\n"
        f"EMOTION: {angle.target_emotion}\n"
        f"HOOK: {angle.hook}\n"
        f"CONCEPT: {angle.concept}\n"
        f"CTA: {angle.cta}\n"
        f"AUDIENCE INSIGHT: {strategy.audience_insight}\n\n"
        f"Write the storyboard: 3-4 shots, whole-second durations summing to "
        f"{brief.min_seconds}-{brief.max_seconds} seconds."
    )
    return _with_repair(
        lambda repair: llm.structured(system=DIRECTOR, user=user + repair, schema=Storyboard),
        lambda sb: validate_storyboard(sb, brief),
        "storyboard",
    )
