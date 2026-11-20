"""Command line interface.

    creative-operator run                      # NomadGuard brief, real providers
    creative-operator run --offline            # no API keys: canned plan + mock video
    creative-operator status <job>             # where a job is and why
    creative-operator resume <job>             # continue a failed / interrupted job
    creative-operator list
    creative-operator qa <video.mp4>           # technical QA on any file
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

from .config import Settings
from .llm import LLM, ClaudeLLM, FixtureLLM
from .models import Brief, Event, Job, JobState, now
from .pipeline import Pipeline
from .providers import PROVIDERS, ProviderError, build_provider
from .qa import SCORE_FIELDS, technical_qa
from .report import write_report
from .store import JobStore

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BRIEF = ROOT / "briefs" / "nomadguard.json"
FIXTURES = ROOT / "fixtures"
STALE_AFTER_S = 120


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    for stream in (sys.stdout, sys.stderr):  # Windows consoles default to a legacy codepage
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(prog="creative-operator", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    def add_run_options(p: argparse.ArgumentParser) -> None:
        p.add_argument("--provider", choices=PROVIDERS, help="video provider (default: omni)")
        p.add_argument("--offline", action="store_true",
                       help="no API calls: canned plan from fixtures/ and the mock provider")
        p.add_argument("--mode", choices=("auto", "single", "per_scene"), default="auto")
        p.add_argument("--clips", nargs="*", type=Path, default=[],
                       help="hand-made clips for --provider manual, in shot order")

    run = sub.add_parser("run", help="create a job from a brief and run it")
    run.add_argument("--brief", type=Path, default=DEFAULT_BRIEF, help="brief JSON (editable)")
    add_run_options(run)

    resume = sub.add_parser("resume", help="continue an interrupted, failed or rejected job")
    resume.add_argument("job_id")
    add_run_options(resume)

    status = sub.add_parser("status", help="show a job's state, plan and QA")
    status.add_argument("job_id")
    status.add_argument("--json", action="store_true", help="dump the raw job record")

    sub.add_parser("list", help="list jobs, newest first")

    check = sub.add_parser("qa", help="run technical QA on any video file")
    check.add_argument("video", type=Path)
    check.add_argument("--brief", type=Path, default=DEFAULT_BRIEF)

    args = parser.parse_args(argv)
    settings = Settings()
    store = JobStore(settings.jobs_dir)

    try:
        if args.command == "run":
            brief = Brief.model_validate_json(args.brief.read_text(encoding="utf-8"))
            pipeline = _pipeline(args, settings, store, default_provider="omni")
            job = pipeline.create(brief)
            print(f"job {job.id}")
            return _finish(pipeline.run(job.id), store)
        if args.command == "resume":
            previous = store.load(args.job_id)
            pipeline = _pipeline(args, settings, store, default_provider=previous.provider)
            return _finish(pipeline.run(args.job_id), store)
        if args.command == "status":
            job = store.load(args.job_id)
            print(job.model_dump_json(indent=2) if args.json else _summary(job))
            return 0
        if args.command == "list":
            for job in store.list():
                print(f"{job.id}  {_state_label(job):<24} {job.provider:<7} {job.brief.brand}")
            return 0
        if args.command == "qa":
            brief = Brief.model_validate_json(args.brief.read_text(encoding="utf-8"))
            report = technical_qa(args.video, brief)
            print(json.dumps({"technical": report.model_dump()}, indent=2))
            return 0 if report.valid else 1
    except (FileNotFoundError, ProviderError) as e:
        print(f"error: {e}", file=sys.stderr)
        if getattr(e, "hint", None):
            print(f"hint:  {e.hint}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\ninterrupted; the job state is saved. Continue with `resume <job>`.")
        return 130
    return 0


def _pipeline(args, settings: Settings, store: JobStore, default_provider: str) -> Pipeline:
    settings.render_mode = args.mode
    if args.offline:
        llm: LLM = FixtureLLM(FIXTURES)
        provider_name = args.provider or "mock"
    else:
        llm = ClaudeLLM(settings.llm_model)
        provider_name = args.provider or default_provider
    if args.clips:
        provider_name = "manual"
    provider = build_provider(provider_name, model=settings.video_model, clips=args.clips)
    return Pipeline(store, llm, provider, settings, on_event=_print_event)


def _print_event(event: Event) -> None:
    print(f"[{event.at.astimezone():%H:%M:%S}] {event.state:<12} {event.message}", flush=True)


def _finish(job: Job, store: JobStore) -> int:
    report = write_report(job, store.dir(job.id))
    print("\n" + _summary(job))
    print(f"\nreport: {report}")
    return 0 if job.state in (JobState.FINALIZED, JobState.NEEDS_REVIEW) else 1


def _state_label(job: Job) -> str:
    idle = (now() - job.updated_at).total_seconds()
    if not job.state.terminal and idle > STALE_AFTER_S:
        return f"{job.state} (stalled)"
    return str(job.state)


def _summary(job: Job) -> str:
    lines = [f"job      {job.id}", f"state    {_state_label(job)}", f"provider {job.provider}"]
    if "(stalled)" in lines[1]:
        lines.append("         no heartbeat for over 2 minutes: the process probably died. "
                     f"Run `resume {job.id}`.")
    if job.error:
        lines.append(f"error    [{job.error.stage} / {job.error.kind}] {job.error.message}")
        if job.error.hint:
            lines.append(f"hint     {job.error.hint}")
    if job.strategy:
        s = job.strategy
        lines.append(f"angle    {s.selected_angle} (of {len(s.angles)}): {s.selection_rationale}")
    if job.storyboard:
        sb = job.storyboard
        lines.append(f"board    {sb.title}: {len(sb.scenes)} shots, {sb.total_seconds}s, "
                     f"{job.render_mode}")
    for task in job.tasks:
        lines.append(f"clip {task.index + 1}   {task.status}, {len(task.attempts)} attempt(s)")
    if job.qa:
        t, c = job.qa.technical, job.qa.creative
        lines.append(f"tech QA  {'valid' if t.valid else 'INVALID'}: {t.duration}s, "
                     f"{t.resolution}, {t.aspect_ratio}" + "".join(f"\n         - {f}" for f in t.failures))
        if c.scores:
            scores = ", ".join(f"{n} {getattr(c.scores, n)}" for n in SCORE_FIELDS)
            lines.append(f"creative {'passed' if c.passed else 'REJECTED'} (mean {c.mean_score}): "
                         f"{scores}" + "".join(f"\n         - {f}" for f in c.failures))
        else:
            lines.append(f"creative skipped: {c.skipped_reason}")
    if job.final_video:
        lines.append(f"video    {job.final_video}")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
