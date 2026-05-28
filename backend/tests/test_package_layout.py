def test_import_pd_assistant_and_subpackages() -> None:
    import pd_assistant
    import pd_assistant.api
    import pd_assistant.core
    import pd_assistant.inference
    import pd_assistant.ingestion
    import pd_assistant.retrieval

    assert pd_assistant.__version__ == "0.1.0"
