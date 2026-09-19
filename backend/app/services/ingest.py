import logging
from pathlib import Path

from ..config import get_settings


log = logging.getLogger("ingest")


def chunk_markdown(text: str, max_chars: int = 320, overlap_chars: int = 48):
	"""Split on headings first, then pack paragraphs into bounded chunks."""
	sections, current = [], []
	for line in text.splitlines():
		if line.startswith("#") and current:
			sections.append("\n".join(current))
			current = []
		current.append(line)
	if current:
		sections.append("\n".join(current))

	chunks, buffer = [], ""
	for section in sections:
		for paragraph in section.split("\n\n"):
			paragraph = paragraph.strip()
			if not paragraph:
				continue
			if len(buffer) + len(paragraph) + 2 > max_chars and buffer:
				chunks.append(buffer.strip())
				buffer = buffer.strip()[-overlap_chars:] + "\n\n"
			buffer += paragraph + "\n\n"
	if buffer.strip():
		chunks.append(buffer.strip())
	return chunks


def ingest_kb(store) -> int:
	settings = get_settings()
	kb_dir = Path(settings.KB_DIR)
	if not kb_dir.exists():
		kb_dir = Path(__file__).resolve().parent.parent / "knowledge"
	ids, docs, metas = [], [], []
	for path in sorted(kb_dir.glob("*.md")):
		for index, chunk in enumerate(chunk_markdown(path.read_text(encoding="utf-8"))):
			ids.append(f"{path.stem}-{index}")
			docs.append(chunk)
			metas.append({"source": path.name})
	if docs:
		store.add_chunks(ids, docs, metas)
	log.info("ingested %d chunks from %s", len(docs), kb_dir)
	return len(docs)
