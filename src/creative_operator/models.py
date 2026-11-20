"""Data contracts shared by every stage.

Models the LLM fills in (Strategy, Storyboard, CreativeScores) deliberately avoid numeric
bounds in the schema: structured outputs guarantee shape, not business rules, so the rules
live in `planner.py` / `qa.py` where a violation can be fed back to the model.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum

from pydantic import BaseModel, Field


def now() -> datetime:
    return datetime.now(timezone.utc)


# --------------------------------------------------------------------------- brief


class Brief(BaseModel):
    brand: str
    product: str
    target_audience: str
    customer_problem: list[str]
    core_message: str
    brand_feel: list[str]
    platforms: list[str] = ["Instagram Reels", "TikTok"]
    aspect_ratio: str = "9:16"
    min_seconds: int = 10
    max_seconds: int = 15


# ------------------------------------------------------------------------ strategy


class Angle(BaseModel):
    name: str
    target_emotion: str
    hook: str = Field(description="What the viewer sees in the first 2 seconds, as a picture.")
    concept: str
    cta: str
    hook_strength: int = Field(description="1-10: would a cold viewer stop scrolling?")
    sound_off_clarity: int = Field(description="1-10: is the story clear with no audio or text?")
    producibility: int = Field(
        description="1-10: can a current AI video model render this photorealistically? "
        "Penalise legible text, screens/UI, hands doing fine work, crowds, logos."
    )


class Strategy(BaseModel):
    angles: list[Angle]
    selected_angle: str = Field(description="Exact `name` of the chosen angle.")
    selection_rationale: str = Field(description="Why this angle over the others. 2 sentences.")
    audience_insight: str = Field(description="The audience truth it rests on. 1-2 sentences.")
    why_it_performs: str = Field(description="Why it should work as a paid social ad. 2 sentences.")

    @property
    def selected(self) -> Angle:
        return next(a for a in self.angles if a.name == self.selected_angle)


# ---------------------------------------------------------------------- storyboard


class Scene(BaseModel):
    duration: int = Field(description="Whole seconds.")
    description: str = Field(description="What happens, in plain language.")
    camera: str = Field(description="Framing, lens feel and movement.")
    video_prompt: str = Field(
        description="Self-contained prompt for the video model for this shot only: subject, "
        "action, setting, light. No on-screen text, no brand names."
    )
    audio: str = Field(description="Diegetic sound / music direction for this shot.")


class Storyboard(BaseModel):
    title: str
    protagonist: str = Field(
        description="One fixed physical description (age, hair, wardrobe) reused in every shot."
    )
    visual_style: str = Field(
        description="One fixed look reused in every shot: light, palette, lens, grade."
    )
    scenes: list[Scene]
    hook_super: str = Field(description="<=6 word caption burned over the opening shot.")
    benefit_super: str = Field(
        default="",
        description="<=6 word caption over the turn shot saying what the product just did "
        "for the protagonist, e.g. 'Rebooked. Hotel covered.'",
    )
    benefit_shot: int = Field(
        default=0, description="1-based number of the shot the benefit caption sits on."
    )
    end_card_headline: str = Field(description="<=7 word line on the closing brand card.")
    end_card_cta: str = Field(description="<=4 word call to action.")

    @property
    def total_seconds(self) -> int:
        return sum(s.duration for s in self.scenes)


# ------------------------------------------------------------------------------ QA


class TechnicalQA(BaseModel):
    exists: bool = False
    playable: bool = False
    duration: float | None = None
    resolution: str | None = None
    aspect_ratio: str | None = None
    fps: float | None = None
    video_codec: str | None = None
    has_audio: bool = False
    valid: bool = False
    failures: list[str] = []
    warnings: list[str] = []


class CreativeScores(BaseModel):
    """Filled in by the vision model. Defects are listed before scores on purpose."""

    observed_defects: list[str] = Field(
        description="Concrete visual defects seen in specific frames (cite timestamps): warped "
        "hands/faces, garbled text, identity drift between shots, morphing objects."
    )
    shot_by_shot: list[str] = Field(description="One line per frame: what is actually visible.")
    realism: int
    visual_consistency: int
    brief_adherence: int
    hook_strength: int
    advertising_quality: int
    fix_suggestions: list[str] = Field(
        description="Prompt-level changes that would fix the worst problems. Empty if none."
    )


class CreativeQA(BaseModel):
    skipped_reason: str | None = None
    scores: CreativeScores | None = None
    mean_score: float | None = None
    passed: bool | None = None
    failures: list[str] = []
    frames: list[str] = []


class QAReport(BaseModel):
    technical: TechnicalQA
    creative: CreativeQA
    passed: bool


# ----------------------------------------------------------------------------- job


class JobState(StrEnum):
    CREATED = "CREATED"
    PLANNING = "PLANNING"
    GENERATING = "GENERATING"
    ASSEMBLING = "ASSEMBLING"
    EVALUATING = "EVALUATING"
    FINALIZED = "FINALIZED"  # asset produced and passed QA
    NEEDS_REVIEW = "NEEDS_REVIEW"  # asset produced, QA failed, regeneration budget spent
    FAILED = "FAILED"  # no usable asset; see job.error

    @property
    def terminal(self) -> bool:
        return self in (JobState.FINALIZED, JobState.NEEDS_REVIEW, JobState.FAILED)


class TaskStatus(StrEnum):
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"


class Attempt(BaseModel):
    number: int
    provider_job_id: str | None = None
    started_at: datetime = Field(default_factory=now)
    ended_at: datetime | None = None
    error_kind: str | None = None
    error: str | None = None
    hint: str | None = None


class ClipTask(BaseModel):
    """One provider generation. A job has one (single-take) or one per scene."""

    index: int
    prompt: str
    duration: int
    uses_reference: bool = False
    status: TaskStatus = TaskStatus.PENDING
    attempts: list[Attempt] = []
    not_before: datetime | None = None  # backoff gate for the next attempt
    output_path: str | None = None


class JobError(BaseModel):
    stage: JobState
    kind: str
    message: str
    hint: str | None = None


class Event(BaseModel):
    at: datetime = Field(default_factory=now)
    state: JobState
    message: str


class Job(BaseModel):
    id: str
    brief: Brief
    provider: str
    state: JobState = JobState.CREATED
    created_at: datetime = Field(default_factory=now)
    updated_at: datetime = Field(default_factory=now)
    strategy: Strategy | None = None
    storyboard: Storyboard | None = None
    render_mode: str | None = None
    tasks: list[ClipTask] = []
    qa_round: int = 0  # how many QA-driven regenerations have been spent
    qa_feedback: list[str] = []
    final_video: str | None = None
    qa: QAReport | None = None
    error: JobError | None = None
    events: list[Event] = []
