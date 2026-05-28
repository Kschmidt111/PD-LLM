def build_rag_prompt(*, disclaimer: str = "Research and education only. Not medical advice.") -> str:
    """Return the system prompt that enforces RAG safety rules."""
    rules = [
        disclaimer,
        "Answer only using the provided context.",
        "Do not use outside knowledge for factual claims.",
        "If the context is insufficient, say insufficient evidence in the provided sources.",
        "Include citations for factual claims using source id, title, and chunk id when available.",
        "Do not diagnose, assess personal risk, or give individualized treatment advice.",
        "Frame responses for research and education.",
        "Do not invent trial results, biomarkers, or statistics.",
    ]
    return "\n".join(rules)
