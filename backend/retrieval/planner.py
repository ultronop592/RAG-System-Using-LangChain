def planner(question: str) -> list[str]:
    """
    Fast keyword-based selection of Qdrant collections.
    Extracted from the notebook for maximum performance.
    """
    q = question.lower()
    
    # Logic from notebook
    if "api" in q or "function" in q or "code" in q:
        return ["code_docs", "research_papers"]
    if "how" in q or "faq" in q or "help" in q:
        return ["faq_data", "knowledge_base"]
    
    # Default search sources
    return ["research_papers", "knowledge_base"]