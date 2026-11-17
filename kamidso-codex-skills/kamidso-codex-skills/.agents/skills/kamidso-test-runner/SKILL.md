---
name: kamidso-test-runner
description: Run and report Kamidso Lite's authorized offline TDD regressions, mocked-provider tests, and real Docker boundary tests without triggering billable live provider traffic. Use for RED/GREEN verification, baseline comparisons, or isolated pytest diagnosis.
---

# Kamidso Test Runner

Use this skill only within the authorized phase, alongside `$kamidso-tdd`. Read local `AGENTS.md` first. Its job is to make tests reproducible, **not** to decide which phase to start.

## Before running

1. Identify exact approved test files/nodes, prior passing baseline, new RED cases, environment, and whether the phase is RED or GREEN. Identify any existing expected failures; never relabel unexpected failures as expected.
2. Verify the local repository root and active Python interpreter. Determine the correct tests from the local source, not from old prompt line numbers.
3. Keep test execution offline with respect to the LLM provider. Do not enable `KAMIDSO_RUN_LIVE_AGENT_TESTS` or `KAMIDSO_RUN_LIVE_LLM_TEST` or call OpenRouter. The diagnostic recorder may be tested with mocked HTTP only.
4. Use isolated test-process configuration (as existing fixtures demand). Do not modify `.env`, local credentials, datasets, or provider configuration. Reuse project-supported environment isolation and configure unique temporary test directories, e.g., `pytest --basetemp`.
5. Check whether Docker integration is needed and available before claiming Docker PASS. Do not simulate Docker results or silently skip required real-container tests.

## Execution order

1. Run the smallest focused test file/node proving the new behavior.
2. Run the related regression areas: Agent, grounding, provider budgets, recoveries, persistence, authorization, diagnostics, and Docker as applicable.
3. At GREEN, execute the complete **offline** suite specified by the project; do not add live tests by using broad collection patterns. At RED, the complete offline suite can legitimately contain expected RED failures.
4. Compare executed test IDs and counts with the approved baseline; report newly collected cases and unexpected skips. A printed count is not a substitute for actual output.
5. Re-run a failing offline test only when necessary to confirm reproducibility and within the phase's authorization; preserve the first error and explain additional runs. **Never** extend this permission to live tests.
6. If Docker is unavailable, report it as unverified; don't relabel container execution as mocked PASS.

## Recommended command pattern

Choose exact local paths rather than blindly copying examples:

```powershell
# Run in a clean process with the project's configured environment.
python -m pytest -q --tb=line -p no:cacheprovider --basetemp "$env:TEMP\kamidso-test-<unique-run-id>" tests/test_approved_red_module.py
python -m pytest -q --tb=line -p no:cacheprovider --basetemp "$env:TEMP\kamidso-full-<unique-run-id>" tests
```

Treat `<unique-run-id>` and `test_approved_red_module.py` as placeholders. Do not execute literal placeholders. Read [test-matrix.md](references/test-matrix.md) before preparing full coverage and report.

## Required reporting

- Phase/stage, exact commands, local files changed, test totals: PASS/expected RED/unexpected FAIL/SKIP.
- Real Docker cases actually run and passed, as distinct from mocked cases.
- Existing suite preserved, with actual baseline comparison.
- Test isolation mode and whether provider calls were made (must be zero for offline work).
- First failing assertion and root-cause evidence, with redacted logs where appropriate.
- Unverified details and the human-approval stopping point.

Never adjust assertions or test discovery to force a green report; changes to approved tests require separate explicit approval.
