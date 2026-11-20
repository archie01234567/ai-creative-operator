"""File-backed job store: one directory per job, `job.json` is the single source of truth.

A JSON file is enough here: one writer per job, and every state change is persisted before
the next side effect, so a killed process can always be resumed from disk.
"""

from __future__ import annotations

import os
from pathlib import Path

from .models import Job, now


class JobStore:
    def __init__(self, root: Path):
        self.root = Path(root)

    def dir(self, job_id: str) -> Path:
        path = self.root / job_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save(self, job: Job) -> None:
        job.updated_at = now()  # doubles as the heartbeat for stale-job detection
        target = self.dir(job.id) / "job.json"
        tmp = target.with_suffix(".json.tmp")
        tmp.write_text(job.model_dump_json(indent=2), encoding="utf-8")
        os.replace(tmp, target)  # atomic: a crash never leaves a half-written job

    def load(self, job_id: str) -> Job:
        path = self.root / job_id / "job.json"
        if not path.exists():
            raise FileNotFoundError(f"No job '{job_id}' in {self.root}")
        return Job.model_validate_json(path.read_text(encoding="utf-8"))

    def list(self) -> list[Job]:
        if not self.root.exists():
            return []
        jobs = [self.load(p.parent.name) for p in self.root.glob("*/job.json")]
        return sorted(jobs, key=lambda j: j.created_at, reverse=True)
