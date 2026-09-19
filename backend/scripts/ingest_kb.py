"""Rebuild the RAG index from app/knowledge/*.md."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.ingest import ingest_kb
from app.services.vectorstore import VectorStore


if __name__ == "__main__":
	store = VectorStore()
	added = ingest_kb(store)
	print(f"Ingested {added} chunks. Collection now has {store.count()} documents.")
