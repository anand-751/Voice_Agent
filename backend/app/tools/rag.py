import logging

from .base import ToolResult
from ..services.doctor_service import get_doctor_recommendation_context


log = logging.getLogger("tools.rag")


def rag_answer(app, state, args) -> ToolResult:
	"""Retrieve clinic FAQ context using Chroma vector search and match
	appropriate specialist doctor and their availability days.
	The Responder LLM then answers strictly grounded in this context.
	"""
	settings = app.state.settings
	raw_input = state.get("user_input", "")
	question = (args.get("question") or raw_input).strip()
	try:
		chunks = app.state.vectors.query(question, top_k=settings.RAG_TOP_K)
	except Exception:
		log.exception("vector query failed")
		chunks = []

	chunks = [chunk for chunk in chunks if chunk and chunk.strip()]
	data = {
		"found": bool(chunks),
		"question": question,
		"context": "\n---\n".join(chunks) if chunks else "",
	}

	# Identify best specialist doctor for the user's problem and their availability days
	rec_ctx = get_doctor_recommendation_context(f"{raw_input} {question}")
	if rec_ctx:
		data.update(rec_ctx)

	if not chunks and not rec_ctx:
		return ToolResult(data={"found": False, "question": question})

	return ToolResult(data=data)
