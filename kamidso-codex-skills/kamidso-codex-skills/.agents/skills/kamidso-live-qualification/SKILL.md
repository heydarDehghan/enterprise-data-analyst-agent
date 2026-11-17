---
name: kamidso-live-qualification
description: Perform a separately approved, single-node, one-shot Kamidso Lite live OpenRouter/real Docker qualification with read-only preflight, strict token/call budgets, private diagnostic capture, exact evidence checks, and no automatic rerun. Use only when the user explicitly authorizes a live provider test.
---

# Kamidso Controlled Live Qualification

**Cost and authority gate:** This skill is not permission to contact a provider. Use only when the user explicitly authorized the **exact live test node**, scope, and execution count. A RED/GREEN approval or generic "test" request is not billable-live authorization.

## Preflight before any provider call

1. Read `AGENTS.md`, the local approved baseline, current local code, and selected live test. Verify the requested node exists and collect **only that node**. Match any required approval against the current user turn.
2. Verify Docker daemon and the required sandbox image locally; confirm no Python code will execute in the API process.
3. Inspect **effective local** provider/model, call count, cumulative token budget, completion cap and reasoning settings. Compare with the explicitly approved values, if provided. Never change them to fit the test; historical settings are not universal defaults.
4. Confirm tenant-aware data, date scope, strict grounding, exact evidence provenance, state persistence, artifact ownership and the selected test harness are in place. Read `$kamidso-evidence-grounding` as needed.
5. Confirm separately required opt-ins for the selected live test and private diagnostics. Set only flags explicitly authorized. Never inspect, print, commit or transmit the API key.
6. Allocate a **new** private absolute diagnostics path, verify redaction and restricted permissions, and avoid overwriting old captures. If required capture cannot be guaranteed, stop **before** billable requests.
7. Check the selected live test is not implicitly collecting unrelated tests; verify no automatic retry behavior in the wrapper. Fail preflight before contacting the provider if any required condition is unmet.

## Execute exactly once

- Run the exact approved node **one time**, no automatic pytest rerun, no post-failure rerun, no extra provider smoke test. Existing bounded Agent internal recoveries are allowed only if they were previously approved in production.
- Do not change production/tests, `.env`, datasets, budgets, model/provider, reasoning controls or prompt to make the live attempt pass.
- Preserve complete private diagnostic evidence, including first error, successful tool results, all model attempts, raw final JSON and accepted grounding decisions. Never append rejected drafts as accepted history.
- If any stage fails, **stop** and report. Do not self-authorize investigation that contacts OpenRouter again.

## Verify independently

- Differentiate upstream HTTP success from application success, and application success from pytest harness success.
- Correlate accepted model answer with actual `grounding_end` and its trace ID. Do not assume every `stop` is accepted.
- For Python+PNG scenario, match accepted revenue evidence to exact successful tool execution and its PNG (not the first artifact). Compare Python stdout to independent SQLite revenue; inspect actual sandbox exit and PNG validity.
- For a follow-up scenario, verify restart, same evidence ID and PNG, no extra Python execution, durable successful turns, and tenant-isolation (Alpha OK, Beta denied).
- Report Length/Evidence/Format recovery as `EXERCISED` or `NOT EXERCISED`; never infer a live recovery PASS from offline coverage.

## Report and stop

Read [live-report-template.md](references/live-report-template.md). Include exact node, counts, all provider finish reasons and usage (report contradictory provider accounting as reported), per-chat budget totals, actual costs, accepted and rejected answers, tool call IDs, artifact IDs, HTTP statuses, state after restart, diagnostics redaction/errors, first failing stage, and unverified checks. No credentials in user-visible output.

Stop for human review even on PASS. Future performance optimization or new live qualification requires distinct authorization.
