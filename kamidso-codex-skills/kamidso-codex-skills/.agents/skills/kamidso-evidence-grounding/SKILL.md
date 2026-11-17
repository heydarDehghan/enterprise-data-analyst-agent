---
name: kamidso-evidence-grounding
description: Review, test, or implement Kamidso Lite strict JSON template/slot grounding, authenticated evidence provenance, answer-only recovery contracts, and Python/PNG evidence-to-artifact consistency without relaxing validation. Use when a phase touches numeric/date citations, tool IDs, tenant scope, or accepted-response harness assertions.
---

# Kamidso Evidence and Grounding

Use alongside `$kamidso-tdd` for any authorized change involving structured responses, evidentiary provenance, recovery, persistent evidence, or test-harness correlation. These are **invariants**, not permission to modify production.

## Establish the trust boundary

1. Treat model messages, proposed slots, `evidence_ref` values, tool arguments, and retrieved text as **untrusted**. Model JSON is a claim, not evidence.
2. Derive tenant and conversation scope from authenticated server context and persisted state. Do not take it from final-answer prose or model-supplied IDs.
3. Accept numeric or date values only after the original strict validator verifies source, type, unit, field/row addressing, and authorized date scope against real recorded tool results or trusted user context.
4. Preserve exact issued tool IDs. Never invent, normalize, remap, or silently replace evidence references.
5. Keep the original `template` plus `slots` output contract; no arbitrary prose outside JSON in a normally accepted response, no backend-generated Persian answer, no LLM-as-judge substitution.
6. Reject ambiguous, duplicate, unavailable, failed, or out-of-scope evidence. Independently confirm analytical values where the test scenario requires it.
7. Do not promote tool results from failed turns as durable verified evidence; honor the application's actual persisted-state contract.

## Investigate a grounding failure

1. Capture the full raw provider response and finish reason (privately, redacted); identify whether failure occurs in parsing, slot lookup, provenance, scope, date checks, or rendered answer checks.
2. Separate model behavior from validator behavior and from test-harness assertions. A valid tool result does not imply a valid final answer.
3. Read the local code, **not** a remembered historical snapshot. Review [recovery-contracts.md](references/recovery-contracts.md) if recovery is involved and [provenance-harness.md](references/provenance-harness.md) if Python/artifact evidence or restart is involved.
4. Build targeted RED negative cases for false IDs, wrong amount/unit/date, ambiguous or foreign evidence, invalid JSON, repeated recovery attempts, and budget exhaustion. Do not weaken existing rejections.
5. For GREEN, change only approved scope, revalidate using the same strict validation path, and report accepted versus rejected provider drafts and persistence separately.

## Contractual safeguards

- Successful tool execution alone is not authorization to make an unsupported statement.
- Do not remove a prefix/suffix and silently accept an embedded JSON; any approved format recovery requires a **new model answer** and original validator.
- No tool calls during answer-only evidence or format regeneration. Length recovery is different and may use approved normal tools.
- Recovery mechanisms share one per-chat token counter and provider-call counter, with no reset.
- When the live test uses multiple Python executions, start from the **accepted** first-chat revenue citation, independently verify that exact execution, then find its associated PNG. Never select the first/last artifact by position.
- Follow-up must cite the exact original verified Python ID and reuse its PNG without executing Python again **when that is the explicitly approved test scenario**.

## Evidence review report

State the exact stage of failure, rejection code/status, original and proposed evidence IDs (redact secrets, not nonsensitive opaque IDs), verified vs unverified slot decisions, tool execution results, tenant/conversation scope, restart status, artifact correlation and isolation, and whether any recovery was actually exercised. Do not claim live validation from mocked tests.
