import json
import pytest

import offline_demo as demo
from agentic_rag.factory import build_app
from agentic_rag.llm.mock import MockLLM


@pytest.mark.parametrize("name", ["agent", "plan"])
def test_offline_demo_scenarios_run_end_to_end(name):
    s, pipe = demo.make_app()
    llm = MockLLM()
    q = demo.SCENARIOS[name](llm)
    ans = build_app(s, llm=llm, pipeline=pipe).ask(q, user_id="SV001")
    llm.assert_done()
    assert ans.route == name and not ans.warnings and ans.sources
    assert ("18/01/2027" if name == "agent" else "8.320.000") in ans.text


def test_api_contract(settings, monkeypatch, pipeline):
    from fastapi.testclient import TestClient
    import agentic_rag.app as api
    from agentic_rag.llm.mock import reply

    monkeypatch.setattr(api, "_engine", build_app(settings, llm=MockLLM([reply("Chào bạn!")] * 3), pipeline=pipeline))
    c = TestClient(api.app)
    assert c.get("/health").json()["chunks"] > 0
    r = c.post("/chat", json={"message": "Xin chào"}, headers={"X-User-Id": "SV001"})
    assert r.status_code == 200 and r.json()["route"] == "direct" and r.json()["session_id"]
    sid = r.json()["session_id"]
    # session của người khác bị từ chối + lỗi có cấu trúc thống nhất
    r2 = c.post("/chat", json={"message": "hi", "session_id": sid}, headers={"X-User-Id": "SV002"})
    assert r2.status_code == 403 and r2.json()["error"]["code"] == "HTTP_403"
    assert c.post("/chat", json={"message": ""}).status_code == 422
