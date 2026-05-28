import json

from pd_assistant.core.logging import configure_logging, get_logger, request_id_context


def test_structured_log_includes_request_id_and_fields(capsys) -> None:
    configure_logging("DEBUG")

    with request_id_context("req-test-123"):
        logger = get_logger("pd_assistant.test")
        logger.info("retrieve_complete", phase="retrieve", duration_ms=42)

    entry = json.loads(capsys.readouterr().out.strip())
    assert entry["event"] == "retrieve_complete"
    assert entry["request_id"] == "req-test-123"
    assert entry["phase"] == "retrieve"
    assert entry["duration_ms"] == 42
    assert entry.get("log_level") == "info" or entry.get("level") == "info"
    assert "timestamp" in entry
