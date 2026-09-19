import logging

import chromadb

from ..config import get_settings
from .embeddings import Embedder


log = logging.getLogger("vectorstore")


class VectorStore:
	"""Chroma persistent collection over the clinic knowledge base."""

	def __init__(self):
		settings = get_settings()
		self.s = settings
		self._client = chromadb.PersistentClient(path=settings.CHROMA_DIR)
		self._col = self._client.get_or_create_collection(
			settings.CHROMA_COLLECTION, metadata={"hnsw:space": "cosine"})
		self._embedder = Embedder()
		try:
			self._embedder._load()
		except Exception as exc:
			log.warning("Embedding model pre-warm skipped: %s", exc)

	def count(self) -> int:
		return self._col.count()

	def ensure_ingested(self):
		if self.count() == 0:
			log.info("knowledge base empty — ingesting from %s", self.s.KB_DIR)
			from .ingest import ingest_kb
			ingest_kb(self)

	def reingest(self):
		log.info("wiping and re-ingesting clean knowledge base from %s", self.s.KB_DIR)
		try:
			self._client.delete_collection(self.s.CHROMA_COLLECTION)
		except Exception:
			pass
		self._col = self._client.get_or_create_collection(
			self.s.CHROMA_COLLECTION, metadata={"hnsw:space": "cosine"}
		)
		from .ingest import ingest_kb
		return ingest_kb(self)

	def add_chunks(self, ids, docs, metas):
		self._col.add(
			ids=ids,
			documents=docs,
			embeddings=self._embedder.embed(docs),
			metadatas=metas,
		)

	def query(self, text: str, top_k: int = 4):
		if self.count() == 0:
			return []
		result = self._col.query(
			query_embeddings=[self._embedder.embed_one(text)],
			n_results=min(top_k, self.count()),
		)
		return (result.get("documents") or [[]])[0]
