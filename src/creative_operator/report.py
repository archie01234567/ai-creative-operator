"""One self-contained HTML page per job: status, strategy, storyboard, prompts, QA, video."""

from __future__ import annotations

import os
from html import escape as e
from pathlib import Path

from .models import Job
from .qa import SCORE_FIELDS

CSS = """
body{font:15px/1.5 system-ui,sans-serif;max-width:1040px;margin:32px auto;padding:0 20px;
color:#16202e;background:#f6f7f9}
h1{margin:0 0 4px}h2{margin:32px 0 10px;font-size:18px}
.card{background:#fff;border:1px solid #dfe3e8;border-radius:10px;padding:16px 18px;margin:10px 0}
.sel{border-color:#1a6fd6;box-shadow:0 0 0 1px #1a6fd6}
.pill{display:inline-block;padding:2px 10px;border-radius:99px;background:#e6eaf0;font-size:13px}
.ok{background:#d9f2e1}.bad{background:#fbdcdc}.warn{background:#fdf0cf}
.grid{display:grid;grid-template-columns:320px 1fr;gap:24px;align-items:start}
video{width:320px;border-radius:12px;background:#000}
pre{white-space:pre-wrap;background:#f1f3f6;padding:10px;border-radius:8px;font-size:13px}
table{border-collapse:collapse;width:100%}td,th{text-align:left;padding:5px 10px 5px 0;
border-bottom:1px solid #eceff3;vertical-align:top}
.frames img{height:210px;border-radius:6px;margin:0 6px 6px 0}.muted{color:#647084}
@media(max-width:760px){.grid{grid-template-columns:1fr}}
"""


def write_report(job: Job, job_dir: Path) -> Path:
    def rel(path: str) -> str:
        return e(os.path.relpath(path, job_dir).replace(os.sep, "/"))

    state_class = {"FINALIZED": "ok", "FAILED": "bad", "NEEDS_REVIEW": "warn"}.get(job.state, "")
    html = [
        f"<!doctype html><meta charset='utf-8'><title>{e(job.brief.brand)} - {e(job.id)}</title>",
        f"<style>{CSS}</style>",
        f"<h1>{e(job.brief.brand)} creative job</h1>",
        f"<div class='muted'>{e(job.id)} · provider <b>{e(job.provider)}</b> · "
        f"render mode <b>{e(job.render_mode or '-')}</b> · QA re-renders used: {job.qa_round}</div>",
        f"<p><span class='pill {state_class}'>{e(job.state)}</span></p>",
    ]
    if job.error:
        html.append(
            f"<div class='card bad'><b>{e(job.error.stage)} failed [{e(job.error.kind)}]</b><br>"
            f"{e(job.error.message)}<br><i>{e(job.error.hint or '')}</i></div>"
        )

    if job.final_video or job.qa:
        html.append("<h2>Result</h2><div class='grid'><div>")
        if job.final_video and Path(job.final_video).exists():
            html.append(f"<video controls playsinline src='{rel(job.final_video)}'></video>")
        html.append("</div><div>")
        if job.qa:
            t, c = job.qa.technical, job.qa.creative
            html.append(
                f"<div class='card'><b>Technical QA</b> "
                f"<span class='pill {'ok' if t.valid else 'bad'}'>"
                f"{'valid' if t.valid else 'invalid'}</span><table>"
                f"<tr><td>Duration</td><td>{t.duration}s</td></tr>"
                f"<tr><td>Resolution</td><td>{e(str(t.resolution))} ({e(str(t.aspect_ratio))})</td></tr>"
                f"<tr><td>Codec / fps</td><td>{e(str(t.video_codec))} / {t.fps}</td></tr>"
                f"<tr><td>Playable / audio</td><td>{t.playable} / {t.has_audio}</td></tr></table>"
                + "".join(f"<div class='pill bad'>{e(x)}</div> " for x in t.failures)
                + "".join(f"<div class='pill warn'>{e(x)}</div> " for x in t.warnings)
                + "</div>"
            )
            html.append("<div class='card'><b>Creative QA</b> ")
            if c.scores:
                html.append(
                    f"<span class='pill {'ok' if c.passed else 'bad'}'>"
                    f"{'passed' if c.passed else 'rejected'} · mean {c.mean_score}</span><table>"
                    + "".join(
                        f"<tr><td>{n.replace('_', ' ')}</td><td>{getattr(c.scores, n)}/10</td></tr>"
                        for n in SCORE_FIELDS
                    )
                    + "</table>"
                    + "".join(f"<div class='pill bad'>{e(x)}</div> " for x in c.failures)
                    + "<p><b>Defects observed</b></p><ul>"
                    + "".join(f"<li>{e(x)}</li>" for x in c.scores.observed_defects)
                    + "</ul>"
                )
            else:
                html.append(f"<span class='pill warn'>skipped</span> {e(c.skipped_reason or '')}")
            html.append("</div>")
        html.append("</div></div>")
        if job.qa and job.qa.creative.frames:
            html.append(
                "<div class='frames'>"
                + "".join(f"<img src='{rel(f)}'>" for f in job.qa.creative.frames)
                + "</div>"
            )

    if job.strategy:
        s = job.strategy
        html.append("<h2>Strategy</h2>")
        for a in s.angles:
            chosen = a.name == s.selected_angle
            html.append(
                f"<div class='card {'sel' if chosen else ''}'><b>{e(a.name)}</b> "
                f"{'<span class=pill>selected</span>' if chosen else ''} "
                f"<span class='muted'>· {e(a.target_emotion)} · hook {a.hook_strength} · "
                f"sound-off {a.sound_off_clarity} · producibility {a.producibility}</span><br>"
                f"<b>Hook:</b> {e(a.hook)}<br><b>Concept:</b> {e(a.concept)}<br>"
                f"<b>CTA:</b> {e(a.cta)}</div>"
            )
        html.append(
            f"<div class='card'><b>Why this angle:</b> {e(s.selection_rationale)}<br>"
            f"<b>Audience insight:</b> {e(s.audience_insight)}<br>"
            f"<b>Why it should perform:</b> {e(s.why_it_performs)}</div>"
        )

    if job.storyboard:
        sb = job.storyboard
        html.append(
            f"<h2>Storyboard: {e(sb.title)} ({sb.total_seconds}s)</h2>"
            f"<div class='card'><b>Protagonist:</b> {e(sb.protagonist)}<br>"
            f"<b>Look:</b> {e(sb.visual_style)}<br><b>Opening caption:</b> {e(sb.hook_super)}<br>"
            f"<b>End card:</b> {e(sb.end_card_headline)} · {e(sb.end_card_cta)}</div>"
            "<div class='card'><table><tr><th>#</th><th>s</th><th>What happens</th>"
            "<th>Camera</th></tr>"
            + "".join(
                f"<tr><td>{i}</td><td>{sc.duration}</td><td>{e(sc.description)}</td>"
                f"<td>{e(sc.camera)}</td></tr>"
                for i, sc in enumerate(sb.scenes, 1)
            )
            + "</table></div>"
        )

    if job.tasks:
        html.append("<h2>Provider requests</h2>")
        for t in job.tasks:
            tries = "".join(
                f"<li>attempt {a.number}: {e(a.error_kind or 'ok')} {e(a.error or '')}</li>"
                for a in t.attempts
            )
            html.append(
                f"<div class='card'><b>Clip {t.index + 1}</b> · {t.duration}s · "
                f"<span class='pill'>{e(t.status)}</span><pre>{e(t.prompt)}</pre><ul>{tries}</ul></div>"
            )

    html.append("<h2>Timeline</h2><div class='card'><table>")
    html += [
        f"<tr><td class='muted'>{ev.at:%H:%M:%S}</td><td>{e(ev.state)}</td><td>{e(ev.message)}</td></tr>"
        for ev in job.events
    ]
    html.append("</table></div>")

    out = job_dir / "report.html"
    out.write_text("\n".join(html), encoding="utf-8")
    return out
