# Evidence, Python and PNG provenance for the live test harness

This is a **test scenario** contract and does not license changing production validators.

1. Identify the exact raw structured JSON the original Agent accepted: match `turn`, `conversation_id`, `trace_id`, provider attempt, `grounding_start` and accepted `grounding_end`; verify the rendered answer equals `/chat` response. Do not select the last `stop` or any rejected draft.
2. Require exactly one relevant numeric `revenue` slot with `evidence_field=revenue` and `unit=scalar` for the Python+PNG qualification; read its exact `evidence_ref` unchanged.
3. Resolve it to **one** issued successful `execute_python` record in the authenticated first persisted successful turn. Independently correlate recorded tool-call message, tool result, trace step, scoped persisted evidence and expected read-only SQLite total. Reject duplicates, fabricated IDs, failed status or mismatched tenant/conversation.
4. Verify generated Python used approved tenant CSV and expected date interval as required by the test, real Docker exit 0, empty stderr, valid structured stdout, and independently correct revenue. Date/path literal checks are not a general proof of all possible generated Python behavior.
5. Locate exactly one eligible valid PNG **from the same cited execution**, also present in returned and persisted artifacts. Do not choose by filename, artifact array order, call order, or revenue amount alone; reject ambiguous multiple eligible PNGs.
6. After restart, confirm same authenticated conversation, trusted prior period, same original evidence ID and selected PNG, distinct trace ID, successful turn statuses and Alpha/Beta HTTP access restrictions. Inspect follow-up trace AND original tool-call messages for any Python attempt, including failed attempts.
7. Preserve original exact-ID assertion; a different valid issued Python reference can be grounded yet still violate the test scenario's reuse requirement.

If the source is not accessible or events cannot be correlated, fail the test explicitly rather than guessing which artifact is correct.
