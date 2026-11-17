# Offline verification matrix

Read this only when building the test selection/report for an approved RED/GREEN stage.

| Surface | What a meaningful test establishes | Failure to avoid |
|---|---|---|
| Tool-calling Agent | Appropriate tool choice, limits, safe failure | Accepting unbounded retries or fake tool calls |
| Structured grounding | Exact `template`/`slots` contract, correct rendered values | Assuming parsable JSON alone proves grounding |
| Numeric/date provenance | Verified amount, type/unit, Gregorian period | Trusting model-supplied numbers or month text |
| Evidence ownership | Exact issued `tool_call_id` in authenticated tenant/conversation | Foreign, fabricated, duplicate or ambiguous refs |
| Answer recovery | Bound each recovery separately; one shared budget | Retry loops, tool calls in answer-only repair |
| Real Docker | Actual sandbox, relevant stdout/stderr/exit code | Mocked container masquerading as real integration |
| Persistence | Verified evidence and artifacts reloaded after restart | Promoting evidence from failed turns |
| Live harness (offline scripted) | Accepted raw response matched to trace and evidence | Choosing last `stop` or first PNG |
| Authz | Alpha/Beta isolation and artifact ownership | Relying only on response status in one happy path |
| Diagnostics | ON/OFF does not change outbound requests; secrets redacted | Leaking keys or changing observed behavior |

## Baseline handling

- Current test counts must come from **local** collection/execution, not this file.
- A RED stage may finish with approved expected failures, but all previously passing tests must remain passing.
- A GREEN stage must resolve the approved RED cases without weakening their controls and run full offline regressions.
- Distinguish the number of Docker tests *executed* from those that actually passed.
- If two commands selected overlapping tests, do not add their pass counts and claim a unique full-suite total.
