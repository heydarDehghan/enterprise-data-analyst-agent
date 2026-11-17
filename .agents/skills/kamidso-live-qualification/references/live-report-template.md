# Controlled live qualification report template

**Phase / approved exact node:**

**Preflight:** local files, expected environment, Docker/image availability, selected node collection, effective model/provider, budgets, reasoning behavior, required opt-in/capture flags (never print key)

**Execution:** executed once? pytest PASS/FAIL; total wall time; app HTTP status for each chat; exact first failing stage; any unreached sections

**Provider calls (per chat):** index, route, finish reason, prompt/completion/reasoning/total tokens, cumulative tokens vs budget, latency, reported USD cost. Retain reported anomalies as-is.

**Tool evidence:** type/name, issued tool ID, arguments if safe, stdout/stderr/exit status, independently verified revenue, CSV tenant and period, generated PNG dimensions/format/ID/integrity

**Grounding:** raw accepted model response, rejected draft reasons, exact revenue `evidence_ref`/field/unit, slot decisions, rendered answer, trace correlation

**Recoveries:** Length/Evidence/Format each `EXERCISED` or `NOT EXERCISED`, eligibility and bounded result when exercised

**Restart and access:** conversation ID, distinct trace IDs, trusted period, preserved evidence ID and bytes of selected PNG, Python reexecution absent, Alpha 200, Beta 404, both persisted turns successful

**Diagnostics and security:** private capture path, key redaction verified, capture errors, exposure limitations

**Unverified claims:** anything not reached because of early failure

**Decision:** PASS for tested path only, or FAIL with first failing stage; stop pending human review, no automatic rerun
