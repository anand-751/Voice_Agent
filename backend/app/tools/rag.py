import logging

from .base import ToolResult


log = logging.getLogger("tools.rag")


def rag_answer(app, state, args) -> ToolResult:
	"""Retrieve clinic FAQ context using Chroma vector search.
	The Responder LLM then answers strictly grounded in this context.
	"""
	settings = app.state.settings
	question = (args.get("question") or state["user_input"]).strip()
	try:
		chunks = app.state.vectors.query(question, top_k=settings.RAG_TOP_K)
	except Exception:
		log.exception("vector query failed")
		chunks = []

	chunks = [chunk for chunk in chunks if chunk and chunk.strip()]
	if not chunks:
		return ToolResult(data={"found": False, "question": question})
	return ToolResult(data={
		"found": True,
		"question": question,
		"context": "\n---\n".join(chunks),
	})
