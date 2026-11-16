# Architecture Decisions

## Project Overview

Kamidso Lite / Data Analyst Agent is designed to answer tenant-scoped business questions over approved enterprise data. The repository is currently a scaffold; no application component is implemented by this document.

## Architecture Decisions

- **Python and FastAPI:** provide the service boundary and a maintainable Python implementation ecosystem.
- **Single-Agent architecture:** keep orchestration understandable and bounded for the initial product scope.
- **Custom LLM tool-calling loop:** make tool selection, step limits, retries, and traceability explicit rather than hiding them behind an opaque framework.
- **SQLite read-only access:** support local analytical data while preventing the agent from mutating source databases.
- **Tenant-aware CSV ticket retrieval:** support ticket analysis while enforcing Alpha/Beta data separation.
- **Isolated Docker-based Python execution:** keep generated analytical code outside the API process.
- **Persistent conversation state:** allow multi-turn analysis with controlled, durable context.
- **Structured JSON tracing:** make agent decisions, tool calls, limits, and failures inspectable.

## Technology Choices and Trade-offs

Python and FastAPI favor delivery speed, typing, and a broad data ecosystem. A single custom loop is easier to constrain and audit than a distributed multi-agent design, but it places more responsibility for orchestration and error handling in the service. SQLite is simple and portable, though it is not a substitute for a multi-user production warehouse. Docker isolation adds operational cost, but it is required for the security boundary around generated Python. CSV retrieval is accessible and portable, while tenant-aware filtering must be designed and tested carefully.

## Security Boundaries

Alpha and Beta tenants must be strictly isolated. SQLite access is read-only, original datasets are immutable, and generated Python must not run in the API process. Secrets must never be stored in source code or emitted in logs. Requests are bounded to 8 agent steps, and failed operations to 2 retries. Revenue calculations use paid orders only. These are design requirements, not implemented controls.

## Current Limitations

This repository contains no FastAPI endpoints, authentication, database layer, agent loop, execution sandbox, conversation store, tracing implementation, datasets, or pytest tests. Provider behavior, deployment, performance, and production readiness are therefore unverified.

## Improvements With One Additional Week

With one additional week, prioritize an approved test-first baseline for tenant isolation, read-only data access, paid-order revenue calculations, step/retry limits, and execution isolation. Then implement the smallest passing vertical slice, add deterministic mocked-LLM unit coverage and real security-boundary integration tests, and document reproducible local verification. Keep deployment and production-provider qualification out of scope until those local contracts are stable.
