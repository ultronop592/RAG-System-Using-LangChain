from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from app.retrieval.planner import planner
from app.retrieval.retriever import hybrid_retrieve
from app.core.llm import llm
from app.cache.cache import response_cache

app = FastAPI()


@app.post("/chat")
async def chat(query: dict):
    question = query["question"]

    if question in response_cache:
        return {"answer": response_cache[question]}

    selected = planner(question)
    docs = await hybrid_retrieve(question, selected)

    context = "\n\n".join(docs)

    prompt = f"""
    Use ONLY this context:

    {context}

    Question:
    {question}
    """

    async def stream():
        response = llm.stream(prompt)
        full = ""
        for chunk in response:
            full += chunk.content
            yield chunk.content
        response_cache[question] = full

    return StreamingResponse(stream(), media_type="text/plain")