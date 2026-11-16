# Kamidso Lite / Data Analyst Agent

## Project overview

Kamidso Lite is an enterprise data analyst agent. Its objective is to answer tenant-scoped business questions from approved SQLite and CSV data while preserving security, auditability, and reproducibility. This repository is being developed incrementally under strict test-driven development (TDD).

This file defines project-wide instructions for Codex. It is a design and development contract, not a claim that any component is implemented.

## Approved architecture

The approved high-level architecture is:

- Python and FastAPI for the service boundary.
- Single-Agent architecture with a custom LLM tool-calling loop.
- SQLite read-only data access.
- Tenant-aware CSV ticket retrieval.
- Isolated Docker-based Python execution for generated analysis code.
- Persistent conversation state.
- Structured JSON tracing.

These choices are approved design constraints. Do not implement or expand them unless the currently authorized phase explicitly permits it.

## Mandatory development workflow

For every development task, Codex MUST load and follow `.agents/skills/kamidso-tdd/SKILL.md` before changing production code, tests, or data. The skill defines the two-stage, phase-based TDD workflow and the required stopping points.

Development proceeds incrementally. Each phase has two separately authorized stages:

- **Phase N-A:** analyze requirements, define acceptance criteria, write tests first, run them to verify expected failures, report, and stop for human review.
- **Phase N-B:** begin only after explicit approval; implement the minimum behavior, pass approved tests, refactor safely, run the complete regression suite, report, and stop for human review.

Do not begin Phase 0-A or any later phase automatically as part of repository initialization. Do not treat approval of one stage as approval of another stage or of a different phase.

## Architectural and security requirements

The following are mandatory constraints:

- A request may execute at most 8 agent steps.
- A failed operation may be retried at most 2 times.
- Alpha and Beta tenants must remain strictly isolated.
- Revenue calculations must use paid orders only.
- Generated Python must never execute in the API process.
- Secrets must not appear in source code or logs.
- Original datasets must remain immutable.
- Example question answers must never be hardcoded.

Security boundaries must be verified with real integration tests where the boundary cannot be represented faithfully by a unit test. Do not weaken, delete, or bypass security tests to obtain a passing result.

## Quality and testing standards

- Follow TDD: no production implementation before the approved failing tests for the current stage.
- Unit tests must use mocked LLM responses and must not depend on a live model.
- Use independent numerical validation for analytical results.
- Keep tests deterministic and focused on the authorized acceptance criteria.
- Run the complete regression suite during Phase N-B after implementation and refactoring.
- Report actual commands, results, failures, and limitations. Never fabricate test results.
- Keep original user-provided datasets immutable; do not create replacement or synthetic datasets unless a later approved phase explicitly authorizes them.
- Do not add dependencies, endpoints, authentication, data access, agent behavior, execution infrastructure, or other scope that the current phase does not authorize.

## Scope control and approvals

Human approval is required between implementation stages and before proceeding to a new phase. If requirements, security implications, data contracts, or acceptance criteria are unclear, stop at the appropriate review point and report the limitation rather than inventing behavior.

The current repository-initialization task is scaffolding-only. It does not authorize FastAPI endpoints, authentication, database access, an LLM agent, Python execution, a Docker sandbox, pytest tests, datasets, or Phase 0-A.
