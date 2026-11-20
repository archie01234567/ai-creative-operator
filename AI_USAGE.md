# AI usage

> **Draft written by the coding agent from the session log. Archie: the two sections marked
> "yours to write" must be in your own words, from your own review. Delete this note after.**

## Tools

- **Claude Code** (Claude Fable 5.2): reading the brief, architecture, all code, tests, docs.
- **Claude Opus 5.5** (runtime): strategy, storyboard and creative QA inside the pipeline.
- **Gemini Omni Flash** (runtime): video generation.

## What was AI-generated

Close to all of it: every module in `src/`, the tests, the offline fixtures in `fixtures/`
(a hand-prompted strategy and storyboard used only by `--offline`), this file and the README.
Human input so far: the choice of providers, and the decisions listed below.

## Things the agent got wrong and corrected during the session

These were caught by running the code and the tests, in the same session:

1. **Provider choice was out of date.** The agent recommended Veo 3.1 on the Gemini API from
   memory. Checking the live docs before writing the adapter showed those models were shut
   down on 22 Oct 2026. The adapter was written for the documented replacement
   (`gemini-omni-1.2-flash`, Interactions API) and its request and error shapes were checked
   against the installed SDK instead of recalled.
2. **A skipped review counted as a pass.** The first QA gate treated "creative review skipped"
   as passing, so an offline run reached `FINALIZED` with nobody having looked at the video.
   Changed so only an explicit pass finalizes; a skipped review ends in `NEEDS_REVIEW`.
3. **Shared mutable state.** Provider error hints were first kept in a class-level dict on
   `Pipeline`, shared by every instance. Moved onto the per-attempt record in the job file.
4. **Missing API key crashed with a traceback.** The Anthropic SDK raises a bare `TypeError`
   when no credentials exist; the pipeline's typed error handling did not cover it. Now mapped
   to a `FAILED` job with a hint.
5. **A test asserted the wrong thing.** The resume test expected three clips to be regenerated
   after a failure; the pipeline had correctly kept the two that finished. The assertion was
   wrong, not the code, and was fixed to check that only the failed clip is re-requested.

## One AI-generated decision I changed or rejected (yours to write)

_TODO (Archie). Candidates from the session if you agree with them after reading the code:
item 1 or 2 above; or anything you change after watching the first real render (prompt
wording, thresholds, single-take vs per-scene)._

## One problem introduced by AI that I detected or corrected (yours to write)

_TODO (Archie). Review `pipeline.py`, `providers/omni.py` and `qa.py` and record what you
find. Note: the Omni adapter has not been exercised against the live API in this draft._
