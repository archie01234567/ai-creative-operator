"""The job lifecycle is the highest-risk code: it spends money and must never lose a job."""

from pathlib import Path

from conftest import ROOT, needs_ffmpeg

from creative_operator.llm import FixtureLLM
from creative_operator.models import JobState, TaskStatus
from creative_operator.pipeline import Pipeline
from creative_operator.providers.base import ErrorKind
from creative_operator.providers.mock import MockProvider
from creative_operator.store import JobStore


class CountingLLM(FixtureLLM):
    def __init__(self):
        super().__init__(ROOT / "fixtures")
        self.calls = 0

    def structured(self, **kwargs):
        self.calls += 1
        return super().structured(**kwargs)


def make_pipeline(settings, provider, llm=None):
    return Pipeline(JobStore(settings.jobs_dir), llm or CountingLLM(), provider, settings,
                    sleep=lambda s: None)  # fmt: skip


@needs_ffmpeg
def test_transient_failure_is_retried_and_job_completes(brief, settings):
    provider = MockProvider(failures=[ErrorKind.TRANSIENT, None])
    pipeline = make_pipeline(settings, provider)

    job = pipeline.run(pipeline.create(brief).id)

    # Offline there is no vision model, so a technically valid asset stops at NEEDS_REVIEW
    # rather than being waved through as FINALIZED.
    assert job.state == JobState.NEEDS_REVIEW
    assert job.qa.technical.valid, job.qa.technical.failures
    attempts = job.tasks[0].attempts
    assert [a.error_kind for a in attempts] == ["transient", None]
    assert Path(job.final_video).exists()


@needs_ffmpeg
def test_fatal_failure_is_recorded_not_raised_and_resume_keeps_finished_work(brief, settings):
    # Per-scene mode (provider max 5s < 13s spot): clip 1 succeeds, clip 2 hits a fatal error.
    llm = CountingLLM()
    failing = MockProvider(failures=[None, ErrorKind.FATAL], max_clip_seconds=5)
    pipeline = make_pipeline(settings, failing, llm)

    job = pipeline.run(pipeline.create(brief).id)  # must not raise

    assert job.state == JobState.FAILED
    assert job.error.stage == JobState.GENERATING and job.error.kind == "generation"
    assert "Clip 2" in job.error.message and "fatal" in job.error.message
    assert len(job.tasks[1].attempts) == 1, "fatal errors must not be retried"
    assert job.tasks[0].status == TaskStatus.SUCCEEDED
    planning_calls = llm.calls
    first_clip = job.tasks[0].output_path

    # A new process (fresh provider with no memory of old jobs) resumes the same job.
    healthy = MockProvider(max_clip_seconds=5)
    job = make_pipeline(settings, healthy, llm).run(job.id)

    assert job.state == JobState.NEEDS_REVIEW and job.error is None
    assert llm.calls == planning_calls, "resume must not pay for planning again"
    assert job.tasks[0].output_path == first_clip, "finished clips must be reused"
    assert len(healthy.requests) == 1, "only the failed clip is generated again"
    assert healthy.requests[0].reference_images, "later shots carry the identity frame"
    assert 10 <= job.qa.technical.duration <= 15


@needs_ffmpeg
def test_job_killed_mid_generation_resubmits_lost_clip(brief, settings):
    store = JobStore(settings.jobs_dir)
    pipeline = make_pipeline(settings, MockProvider())
    job = pipeline.create(brief)

    # Simulate a crash right after submission: drive the stages by hand, then drop the process.
    pipeline._start(job)
    pipeline._plan(job)
    pipeline._submit(job, job.tasks[0])
    store.save(job)
    assert store.load(job.id).tasks[0].status == TaskStatus.SUBMITTED

    job = make_pipeline(settings, MockProvider()).run(job.id)

    assert job.state == JobState.NEEDS_REVIEW
    assert [a.error_kind for a in job.tasks[0].attempts] == ["lost", None]
