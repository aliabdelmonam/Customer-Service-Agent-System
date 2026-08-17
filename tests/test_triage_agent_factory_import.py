import importlib


def test_triage_agent_factory_imports_from_src_package():
    module = importlib.import_module("src.agents.triage_agent_factory")
    assert hasattr(module, "TriageAgent")
    assert hasattr(module, "TriageResult")
