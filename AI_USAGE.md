# AI usage

> **Draft written by the coding agent from the session log. Archie: the two sections marked
> "yours to write" must be in your own words, from your own review. Delete this note after.**

## Tools

- **Claude Code** (Claude Fable 5.2): reading the brief, architecture, all code, tests, docs.
- **Claude Opus 5.5** (runtime): strategy, storyboard and creative QA inside the pipeline.
- **Kling 4 Flash on Replicate** (runtime): the video generation behind the submitted render.
- **Gemini Omni Flash** (runtime): second video adapter; never produced a clip (no quota on my key).

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

6. **The Omni adapter was wrong in three ways on first contact with the real API.** It used
   the SDK's default API version (404 for the model), caught the wrong exception family (so
   the 404 was retried three times as "transient"), and used a blocking call that hung for
   over two hours. Rewritten to `v1beta`, background interactions and polling by id.
7. **QA feedback asked the video model for text.** The reviewer's fix suggestions were pasted
   into the retake prompt, including "phone showing 'Flight cancelled'"; the retake came back
   with a burned-in caption overlapping the real one. The reviewer is now told its
   suggestions must describe footage only.
8. **The first concept ignored a known model weakness.** The planner chose a split screen
   with the same person twice; the render was two near-identical plates. Composites and
   duplicated subjects are now ruled out in the strategist and director prompts.

## One AI-generated decision I changed or rejected (yours to write)

_TODO (Archie). Candidates from the session if you agree with them after reading the code:
item 1 or 2 above; item 8 (the split-screen concept), or the switch from Gemini to Replicate._

## One problem introduced by AI that I detected or corrected (yours to write)

_TODO (Archie). Candidates if you agree after reading the code: item 6 or 7 above._
