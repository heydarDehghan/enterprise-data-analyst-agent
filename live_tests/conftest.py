"""Expose resolved LLM settings to tests that have explicitly opted in."""

from pathlib import Path

import pytest

from app.config import LLMConfigurationError, load_llm_config
from live_tests.llm_diagnostics import APPROVED_DIAGNOSTIC_NODES, capture_requested, capture_run


_SELECTION_ERROR = "Select exactly one approved live test by its full pytest node ID for LLM diagnostic capture"


def pytest_collection_finish(session):
    if not capture_requested():
        return

    # Filtering a broad selector to one item or deduplicating arguments does not count as explicit selection.
    config = session.config
    if len(config.args) != 1:
        raise pytest.UsageError(_SELECTION_ERROR)
    filename, separator, case = config.args[0].partition("::")
    if not filename or not separator:
        raise pytest.UsageError(_SELECTION_ERROR)
    try:
        file = (config.invocation_params.dir / Path(filename)).resolve().relative_to(config.rootpath)
    except (OSError, ValueError):
        raise pytest.UsageError(_SELECTION_ERROR) from None
    if f"{file.as_posix()}::{case}" not in APPROVED_DIAGNOSTIC_NODES:
        raise pytest.UsageError(_SELECTION_ERROR)


@pytest.fixture(autouse=True)
def provider_environment(monkeypatch, request):
    # The existing skip markers apply before fixture setup.
    if capture_requested():
        items = request.session.items
        if len(items) != 1 or items[0].nodeid not in APPROVED_DIAGNOSTIC_NODES:
            pytest.fail(_SELECTION_ERROR, pytrace=False)

    try:
        config = load_llm_config()
    except LLMConfigurationError as exc:
        pytest.fail(str(exc), pytrace=False)

    # The approved live tests inspect these three names in the process environment.
    monkeypatch.setenv("LLM_CHAT_COMPLETIONS_URL", config.url)
    monkeypatch.setenv("LLM_MODEL", config.model)
    monkeypatch.setenv("LLM_API_KEY", config.api_key)
    with capture_run(node_id=request.node.nodeid, monkeypatch=monkeypatch):
        yield
