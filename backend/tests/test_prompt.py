from pd_assistant.inference.prompt import build_rag_prompt



def test_build_rag_prompt() -> None:
    prompt = build_rag_prompt()
    assert prompt is not None
    assert "Research and education only. Not medical advice." in prompt
    assert "Answer only using the provided context." in prompt
    assert "Do not use outside knowledge for factual claims." in prompt
    assert "If the context is insufficient, say insufficient evidence in the provided sources." in prompt
    assert "Include citations for factual claims using source id, title, and chunk id when available." in prompt
    assert "Do not diagnose, assess personal risk, or give individualized treatment advice." in prompt
    assert "Frame responses for research and education." in prompt
    assert "Do not invent trial results, biomarkers, or statistics." in prompt