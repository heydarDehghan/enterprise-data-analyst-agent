---
name: kamidso-tdd
description: Enforce Kamidso Lite's phase-based, approval-gated test-driven development workflow for the data analyst agent.
---

# Kamidso TDD

Use this skill for every Kamidso Lite development task that changes production code, tests, or data. It enforces a bounded workflow with explicit human approval between test-first analysis and implementation.

## Required workflow

Follow this sequence for each authorized phase:

```text
REQUIREMENTS
    ↓
ACCEPTANCE CRITERIA
    ↓
RED: Write failing tests
    ↓
STOP FOR HUMAN REVIEW
    ↓
GREEN: Minimum implementation
    ↓
REFACTOR
    ↓
REGRESSION TESTS
    ↓
STOP FOR HUMAN REVIEW
```

Each phase has two separately authorized stages.

### Phase N-A — test-first analysis

1. Analyze only the requirements within the authorized phase.
2. Define observable acceptance criteria.
3. Write tests before implementation.
4. Run the focused tests and verify the expected failures.
5. Report the actual results, failures, and limitations.
6. Stop and request human approval. Do not implement production behavior.

### Phase N-B — implementation

Begin only after explicit human approval for the corresponding Phase N-A.

1. Implement the minimum functionality needed for the approved acceptance criteria.
2. Run the approved tests and make them pass.
3. Refactor safely without weakening the behavior or security contract.
4. Run the complete regression suite.
5. Report actual results, failures, and limitations.
6. Stop and request human approval before the next phase or stage.

## Non-negotiable constraints

- No production code before the approved tests for the current phase.
- Do not weaken, delete, skip, or bypass security tests.
- Never fabricate test results, coverage, or external-system evidence.
- Do not modify original data or create replacement datasets.
- Validate numerical results independently of the implementation under test.
- Use unit tests with mocked LLM responses; do not require a live LLM for unit tests.
- Use real integration tests for security boundaries such as tenant isolation and execution isolation.
- Report failures, unverified assumptions, and limitations clearly.
- Do not implement beyond the authorized phase, even if adjacent work appears useful.
- Respect the project limits of at most 8 agent steps per request and at most 2 retries per failed operation.
- Preserve the requirements that paid orders are the sole source for revenue calculations, generated Python never runs in the API process, secrets stay out of source and logs, and tenants remain isolated.

## Standard completion report

Every stage report must include:

1. Current phase and stage.
2. Acceptance criteria.
3. Files created or modified.
4. Tests executed and their actual results.
5. Failures and limitations.
6. Approval required for the next stage.

For a Phase N-A report, explicitly identify the expected failing tests. For a Phase N-B report, distinguish focused-test results from complete-regression results and identify any evidence that was not obtained.
