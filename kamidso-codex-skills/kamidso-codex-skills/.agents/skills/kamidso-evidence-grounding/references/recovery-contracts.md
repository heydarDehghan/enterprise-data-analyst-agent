# Existing bounded recovery contracts

These describe approved behavior observed in prior project phases. **Confirm current local code and tests** before modifying or asserting a specific path exists.

| Path | Entry condition | Re-generation tools | Safe bound | Acceptance |
|---|---|---|---|---|
| Length recovery | Well-formed `finish_reason=length`, null/empty content, no tool calls, valid usage and headroom | Normal approved tools | At most one per chat | Any final answer still passes original strict grounding; partial tools never dispatched |
| Evidence-reference repair | Eligible unknown ref, unique matching authorized advertised field/value/type/unit/row, other slots and month scope correct; reject previously issued foreign refs | `tools=[]` | One bounded answer-only regeneration under current approved implementation | Fresh exact-ID JSON passes original validator; no server-side substitution |
| JSON format repair | Normally final `stop` contains exactly one complete scoped-valid `template`/`slots` object with plain prose framing, not arbitrary malformed response | `tools=[]` | At most one per chat | Reject original entire response; fresh independent JSON passes original validator |

## Shared invariants

- All attempts, including recoveries, consume the **same chat** call and cumulative token budgets; keep the original per-call completion limit.
- No nested/unbounded retry loops. An answer-only repair that fails is terminal; it cannot re-enable tools or trigger a different repair.
- No rejected draft or invented evidence ID is sent as fresh model input in repair. Preserve original trusted context and verified manifest.
- With no eligible call allowance, preserve original grounding failure where that is the contract. If cumulative usage reaches/exceeds the configured token limit, preserve budget error (historically HTTP 429). Other failure codes follow the local tests.
- Optional diagnostics must not alter requests or accepted answers; capture attempts, decisions and usage with secrets redacted.
- **Do not silently introduce or reconfigure any recovery while doing unrelated work.** A behavior change requires its own approved RED and GREEN stages.
