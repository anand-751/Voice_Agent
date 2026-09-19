import logging

from ..config import get_settings


log = logging.getLogger("embeddings")


class Embedder:
	"""Lazy-loaded fastembed model with a singleton instance."""

	_instance = None

	def __new__(cls):
		if cls._instance is None:
			cls._instance = super().__new__(cls)
			cls._instance._model = None
		return cls._instance

	def _load(self):
		if self._model is None:
			from fastembed import TextEmbedding
			model_name = get_settings().EMBEDDING_MODEL
			log.info("loading embedding model %s …", model_name)
			self._model = TextEmbedding(model_name)
		return self._model

	def embed(self, texts):
		return [vector.tolist() for vector in self._load().embed(list(texts))]

	def embed_one(self, text: str):
		return self.embed([text])[0]
