"""Phase 3-A Agent integration contract; the sandbox is a test double here."""

from app.schemas import ChatRequest
from app.secure_data import SecureDataAccess

from test_agent_loop import DATA, TOKEN_MAP, SCHEMA, ScriptedLLM, call, numeric_slot, structured_turn, tool_turn, tool_result


def test_execute_python_tool_is_advertised_only_when_sandbox_configured():
    from app.agent import AgentLoop

    code = 'print(\'{"metrics":{"result":"%s"}}\' % (6 * 7))'
    stdout = '{"metrics":{"result":"42"}}\n'

    class SandboxSpy:
        calls = []

        def execute(self, code, tenant):
            self.calls.append((code, tenant))
            return type("Result", (), {"stdout": stdout, "stderr": "", "exit_code": 0,
                                       "artifacts": []})()

    store = SecureDataAccess(data_dir=DATA, token_map=TOKEN_MAP)
    tenant = store.authenticate("demo-alpha")
    sandbox = SandboxSpy()

    def check_tools(messages, tools):
        names = {tool["function"]["name"] for tool in tools}
        assert names == {"query_sql", "get_schema", "get_tickets", "execute_python"}
        assert "query" not in names and "python" not in names
        return tool_turn(call("execute_python", {"code": code}, "py_1"))

    def check_result(messages, tools):
        assert tool_result(messages, "py_1") == {
            "stdout": stdout, "stderr": "", "exit_code": 0, "artifacts": []
        }
        return structured_turn("نتیجه محاسبه {{result}} است.",
                               result=numeric_slot("۴۲", "py_1", "result", "scalar"))

    llm = ScriptedLLM(check_tools, check_result)
    response = AgentLoop(llm=llm, tenant=tenant, sandbox=sandbox, schema_text=SCHEMA,
                         trace_id_factory=lambda: "trace-sandbox-1").run(
        ChatRequest(question="۶ در ۷ چند است؟", conversation_id="conv-1")
    )
    assert sandbox.calls == [(code, tenant)]
    assert response.trace[0].tool_name == "execute_python"
    assert response.trace[0].status == "success"
    assert response.answer == "نتیجه محاسبه 42 است."


def test_execute_python_arguments_are_validated_before_dispatch():
    from app.agent import AgentLoop

    class SandboxSpy:
        calls = 0

        def execute(self, code, tenant):
            self.calls += 1
            raise AssertionError("Invalid tool input must not execute")

    tenant = SecureDataAccess(data_dir=DATA, token_map=TOKEN_MAP).authenticate("demo-alpha")
    sandbox = SandboxSpy()
    llm = ScriptedLLM(
        tool_turn(call("execute_python", {"code": "print(1)", "host_path": "/etc"})),
        lambda messages, tools: (
            tool_result(messages, "call_1")["error"] and
            structured_turn("ورودی ابزار نامعتبر بود.")
        ),
    )
    response = AgentLoop(llm=llm, tenant=tenant, sandbox=sandbox, schema_text=SCHEMA).run(
        ChatRequest(question="یک محاسبه انجام بده", conversation_id="conv-1")
    )
    assert sandbox.calls == 0
    assert response.trace[0].status == "error"
