"""Job lifecycle.

    CREATED -> PLANNING -> GENERATING -> ASSEMBLING -> EVALUATING -> FINALIZED
                               ^                           |
                               +---- QA-driven re-render --+      NEEDS_REVIEW / FAILED

Every transition is written to disk before the next side effect, and every stage is safe to
re-enter, so `run()` is also `resume()`: it looks at the persisted state and continues.
Expected failures end the job in FAILED with a kind, message and hint; they never escape
as a traceback.
"""

from __future__ import annotations

import random
import time
import uuid
from datetime import timedelta
from pathlib import Path
from typing import Callable

from . import media, planner, prompt_builder, qa
from .config import Settings
from .errors import GenerationError, StageError
from .llm import LLM
from .models import (
    Attempt,
    Brief,
    ClipTask,
    Event,
    Job,
    JobError,
    JobState,
    TaskStatus,
    now,
)
from .providers.base import (
    ErrorKind,
    GenerationRequest,
    ProviderError,
    ProviderState,
    VideoProvider,
)
from .store import JobStore


class Pipeline:
    def __init__(
        self,
        store: JobStore,
        llm: LLM,
        provider: VideoProvider,
        settings: Settings | None = None,
        on_event: Callable[[Event], None] | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.store = store
        self.llm = llm
        self.provider = provider
        self.settings = settings or Settings()
        self.on_event = on_event or (lambda event: None)
        self.sleep = sleep

    # ------------------------------------------------------------------ public

    def create(self, brief: Brief) -> Job:
        job = Job(
            id=f"{now().astimezone():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:4]}",
            brief=brief,
            provider=self.provider.name,
        )
        self._log(job, "Job created.")
        return job

    def run(self, job_id: str) -> Job:
        """Drive a job to a terminal state. Safe to call again on any non-finalized job."""
        job = self.store.load(job_id)
        if job.state == JobState.FINALIZED:
            return job
        if job.state.terminal:
            self._reopen(job)

        stages = {
            JobState.CREATED: self._start,
            JobState.PLANNING: self._plan,
            JobState.GENERATING: self._generate,
            JobState.ASSEMBLING: self._assemble,
            JobState.EVALUATING: self._evaluate,
        }
        while not job.state.terminal:
            stage = job.state
            try:
                stages[stage](job)
            except StageError as e:
                self._fail(job, stage, e.kind, e.message, e.hint)
            except ProviderError as e:
                self._fail(job, stage, f"provider_{e.kind}", e.message, e.hint)
            except KeyboardInterrupt:
                self._log(job, "Interrupted. Run `resume` to continue from this state.")
                raise
            except Exception as e:  # a bug, not an expected failure: record it, then surface it
                self._fail(job, stage, "unexpected", f"{type(e).__name__}: {e}", None)
                raise
        return job

    # ------------------------------------------------------------------ stages

    def _start(self, job: Job) -> None:
        self._transition(job, JobState.PLANNING, "Planning creative strategy.")

    def _plan(self, job: Job) -> None:
        if job.strategy is None:
            job.strategy = planner.plan_strategy(self.llm, job.brief)
            self._log(
                job,
                f"{len(job.strategy.angles)} angles proposed; selected "
                f"'{job.strategy.selected_angle}'.",
            )
        if job.storyboard is None:
            job.storyboard = planner.plan_storyboard(self.llm, job.brief, job.strategy)
            self._log(
                job,
                f"Storyboard '{job.storyboard.title}': {len(job.storyboard.scenes)} shots, "
                f"{job.storyboard.total_seconds}s.",
            )
        self._build_tasks(job)
        self._transition(job, JobState.GENERATING, f"Generating video ({job.render_mode}).")

    def _build_tasks(self, job: Job) -> None:
        job.render_mode, job.tasks = prompt_builder.build_tasks(
            job.storyboard,
            self.provider.capabilities,
            self.settings.render_mode,
            job.qa_feedback,
        )

    def _generate(self, job: Job) -> None:
        if not job.tasks:
            self._build_tasks(job)
        while True:
            for task in job.tasks:
                if task.status == TaskStatus.SUCCEEDED and not Path(task.output_path or "").exists():
                    task.status = TaskStatus.PENDING  # the clip was deleted from disk
                if task.status == TaskStatus.PENDING:
                    self._submit(job, task)
                elif task.status == TaskStatus.SUBMITTED:
                    self._poll(job, task)
            self.store.save(job)  # heartbeat + progress

            failed = [t for t in job.tasks if t.status == TaskStatus.FAILED]
            if failed:
                last = failed[0].attempts[-1]
                raise GenerationError(
                    f"Clip {failed[0].index + 1} failed after {len(failed[0].attempts)} "
                    f"attempt(s) [{last.error_kind}]: {last.error}",
                    hint=last.hint or "Fix the cause, then `resume` the job; finished clips are kept.",
                )
            if all(t.status == TaskStatus.SUCCEEDED for t in job.tasks):
                break
            self.sleep(self.settings.poll_interval_s)
        self._transition(job, JobState.ASSEMBLING, "All clips generated; assembling.")

    def _submit(self, job: Job, task: ClipTask) -> None:
        if task.not_before and now() < task.not_before:
            return  # still backing off
        first = job.tasks[0]
        if task.uses_reference and first.status != TaskStatus.SUCCEEDED:
            return  # identity reference comes from the first clip

        clip_dir = self.store.dir(job.id) / "clips"
        number = len(task.attempts) + 1
        output = clip_dir / f"round{job.qa_round}_clip{task.index + 1}_try{number}.mp4"
        references = []
        if task.uses_reference:
            info = media.probe(Path(first.output_path))
            references = [
                media.extract_frame(
                    Path(first.output_path), info.duration / 2, clip_dir / "identity_ref.jpg", 768
                )
            ]
        request = GenerationRequest(
            prompt=task.prompt,
            duration=task.duration,
            output_path=output,
            aspect_ratio=job.brief.aspect_ratio,
            resolution=self.settings.video_resolution,
            reference_images=references,
        )
        attempt = Attempt(number=number)
        task.attempts.append(attempt)
        try:
            attempt.provider_job_id = self.provider.generate(request)
        except ProviderError as e:
            self._attempt_failed(job, task, e)
            return
        task.status = TaskStatus.SUBMITTED
        self._log(job, f"Clip {task.index + 1}: submitted (attempt {number}).")

    def _poll(self, job: Job, task: ClipTask) -> None:
        attempt = task.attempts[-1]
        status = self.provider.get_status(attempt.provider_job_id)
        if status.state == ProviderState.SUCCEEDED:
            path = self.provider.get_result(attempt.provider_job_id)
            try:
                media.probe(path)  # a "successful" job can still hand back a broken file
            except StageError as e:
                self._attempt_failed(
                    job, task, ProviderError(ErrorKind.TRANSIENT, f"unusable output: {e.message}")
                )
                return
            attempt.ended_at = now()
            task.output_path = str(path)
            task.status = TaskStatus.SUCCEEDED
            self._log(job, f"Clip {task.index + 1}: ready.")
        elif status.state == ProviderState.FAILED:
            self._attempt_failed(
                job, task, status.error or ProviderError(ErrorKind.TRANSIENT, "unknown failure")
            )
        elif (now() - attempt.started_at).total_seconds() > self.settings.poll_timeout_s:
            self._attempt_failed(
                job,
                task,
                ProviderError(
                    ErrorKind.TRANSIENT,
                    f"no result after {self.settings.poll_timeout_s:.0f}s; abandoning attempt",
                ),
            )

    def _attempt_failed(self, job: Job, task: ClipTask, error: ProviderError) -> None:
        attempt = task.attempts[-1]
        attempt.ended_at = now()
        attempt.error_kind = error.kind
        attempt.error = error.message
        attempt.hint = error.hint
        if error.retryable and len(task.attempts) < self.settings.max_attempts:
            delay = self.settings.backoff_base_s * 2 ** (len(task.attempts) - 1)
            delay += random.uniform(0, delay / 4)
            task.not_before = now() + timedelta(seconds=delay)
            task.status = TaskStatus.PENDING
            self._log(
                job,
                f"Clip {task.index + 1}: attempt {attempt.number} failed [{error.kind}] "
                f"{error.message} Retrying in {delay:.0f}s.",
            )
        else:
            task.status = TaskStatus.FAILED
            self._log(
                job,
                f"Clip {task.index + 1}: attempt {attempt.number} failed [{error.kind}] "
                f"{error.message} Not retrying.",
            )

    def _assemble(self, job: Job) -> None:
        s, sb = self.settings, job.storyboard
        job_dir = self.store.dir(job.id)
        size = (s.width, s.height)
        durations = prompt_builder.scene_durations(sb, job.render_mode)
        clips = [(Path(t.output_path), float(d)) for t, d in zip(job.tasks, durations)]
        total = sum(min(d, media.probe(p).duration) for p, d in clips)

        hook = media.render_hook_super(sb.hook_super, job_dir / "hook.png", size, s.font_path)
        card = media.render_end_card(
            job.brief.brand, sb.end_card_headline, sb.end_card_cta, job_dir / "end_card.png",
            size, s.font_path,
        )  # fmt: skip
        final = job_dir / f"final_round{job.qa_round}.mp4"
        media.assemble(
            clips,
            final,
            [
                media.Overlay(hook, 0.0, s.hook_super_seconds, fade=0.25),
                media.Overlay(card, max(total - s.end_card_seconds, 0.0), total),
            ],
            size=size,
            fps=s.fps,
        )
        job.final_video = str(final)
        self._transition(job, JobState.EVALUATING, "Assembled; running QA.")

    def _evaluate(self, job: Job) -> None:
        report = qa.evaluate(
            self.llm,
            Path(job.final_video),
            job.brief,
            job.strategy,
            job.storyboard,
            self.store.dir(job.id) / f"frames_round{job.qa_round}",
            self.settings,
        )
        job.qa = report
        if report.passed:
            self._transition(job, JobState.FINALIZED, "QA passed. Final asset ready.")
            return

        problems = report.technical.failures + report.creative.failures
        rejected = not report.technical.valid or report.creative.passed is False
        if rejected and job.qa_round < self.settings.qa_regenerations:
            job.qa_round += 1
            scores = report.creative.scores
            job.qa_feedback = (scores.fix_suggestions or scores.observed_defects)[:4] if scores else []
            self._build_tasks(job)
            self._transition(
                job,
                JobState.GENERATING,
                f"QA rejected the render ({'; '.join(problems)}). Re-rendering with feedback "
                f"(round {job.qa_round} of {self.settings.qa_regenerations}).",
            )
            return
        reason = "; ".join(problems) or report.creative.skipped_reason or "QA did not pass"
        self._transition(job, JobState.NEEDS_REVIEW, f"Asset needs human review: {reason}")

    # ------------------------------------------------------------------ helpers

    def _reopen(self, job: Job) -> None:
        """Resume a FAILED / NEEDS_REVIEW job from the stage that makes sense for its data."""
        if job.state == JobState.NEEDS_REVIEW:
            target = JobState.EVALUATING
        elif job.error:
            target = job.error.stage
        else:
            target = JobState.PLANNING
        for task in job.tasks:
            if task.status == TaskStatus.FAILED:  # a fresh retry budget for the human-fixed cause
                task.status, task.attempts, task.not_before = TaskStatus.PENDING, [], None
        job.error = None
        self._transition(job, target, f"Reopened at {target}.")

    def _transition(self, job: Job, state: JobState, message: str) -> None:
        job.state = state
        self._log(job, message)

    def _fail(self, job: Job, stage: JobState, kind: str, message: str, hint: str | None) -> None:
        job.error = JobError(stage=stage, kind=kind, message=message, hint=hint)
        self._transition(job, JobState.FAILED, f"Failed during {stage} [{kind}]: {message}")

    def _log(self, job: Job, message: str) -> None:
        event = Event(state=job.state, message=message)
        job.events.append(event)
        self.store.save(job)
        self.on_event(event)
