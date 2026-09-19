import logging

from .base import ToolResult


log = logging.getLogger("tools.rag")


def rag_answer(app, state, args) -> ToolResult:
	"""Extract factual clinic knowledge using Google's langextract library with vector fallback.
	The Responder LLM then answers strictly grounded in this context.
	"""
	settings = getattr(app, "state", None) and getattr(app.state, "settings", None)
	question = (args.get("question") or state.get("user_input") or "").strip()

	# 1. Primary: Google langextract knowledge extraction
	extractor = getattr(getattr(app, "state", None), "extractor", None)
	use_extractor = getattr(settings, "USE_LANGEXTRACT", True) if settings else True

	if extractor and use_extractor:
		try:
			extract_res = extractor.extract_clinic_info(question)
			if extract_res and extract_res.get("found"):
				log.info("Knowledge extracted via %s for question: %s", extract_res.get("method"), question)
				return ToolResult(data={
					"found": True,
					"question": question,
					"direct_answer": extract_res.get("direct_answer", ""),
					"context": extract_res.get("context", ""),
					"source_quotes": extract_res.get("source_quotes", []),
					"extraction_method": extract_res.get("method", "langextract"),
				})
		except Exception as exc:
			log.warning("Extractor failed for '%s': %s — falling back to vector query", question, exc)

	# 2. Fallback: Chroma vector search
	top_k = getattr(settings, "RAG_TOP_K", 4) if settings else 4
	chunks = []
	vectors = getattr(getattr(app, "state", None), "vectors", None)
	if vectors:
		try:
			chunks = vectors.query(question, top_k=top_k)
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
		"extraction_method": "chroma_vector",
	})
