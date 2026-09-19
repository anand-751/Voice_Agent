"""LangExtract Knowledge Extraction Service.

Replaces and enhances traditional vector similarity search by using Google's
langextract library to extract precise, source-grounded answers and facts
directly from the clinic knowledge base (clinic_faqs.md).
"""

import asyncio
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import get_settings

log = logging.getLogger("extractor")


class LangExtractService:
	"""Extracts grounded clinic facts from clinic_faqs.md using langextract with triple-redundant fallback."""

	def __init__(self, settings=None):
		self.s = settings or get_settings()
		self.api_key = getattr(self.s, "GEMINI_API_KEY", "") or os.environ.get("GEMINI_API_KEY", "")
		self.model_id = getattr(self.s, "LANGEXTRACT_MODEL", "gemini-2.5-flash")
		self.faqs_path = self._resolve_faqs_path()
		self.faqs_text = self._load_faqs_text()
		self.sections = self._parse_sections(self.faqs_text)
		self._cache: Dict[str, dict] = {}
		self._examples = self._build_examples()
		log.info(
			"LangExtractService initialized with %d sections from %s (model=%s)",
			len(self.sections),
			self.faqs_path,
			self.model_id,
		)

	def _build_examples(self):
		try:
			import langextract as lx
			return [
				lx.data.ExampleData(
					text="Root canal treatment (RCT): ₹3,500 to ₹6,000 per tooth depending on the tooth",
					extractions=[
						lx.data.Extraction(
							extraction_class="service_pricing",
							extraction_text="₹3,500 to ₹6,000 per tooth",
							attributes={"service": "Root canal treatment (RCT)", "cost": "₹3,500 to ₹6,000"},
						)
					],
				),
				lx.data.ExampleData(
					text="Monday to Saturday: 10:00 AM – 7:00 PM\nSunday: closed (emergencies by phone only)\nLunch break: 1:30 PM – 2:30 PM",
					extractions=[
						lx.data.Extraction(
							extraction_class="timing",
							extraction_text="10:00 AM – 7:00 PM",
							attributes={"days": "Monday to Saturday", "hours": "10:00 AM to 7:00 PM"},
						),
						lx.data.Extraction(
							extraction_class="timing",
							extraction_text="1:30 PM – 2:30 PM",
							attributes={"type": "lunch break", "no_appointments": "true"},
						),
					],
				),
				lx.data.ExampleData(
					text="Dr. Rohit Verma, BDS MDS (Orthodontics) — braces and aligners, visits on Tuesday, Thursday and Saturday.",
					extractions=[
						lx.data.Extraction(
							extraction_class="doctor_specialty",
							extraction_text="braces and aligners, visits on Tuesday, Thursday and Saturday",
							attributes={"doctor": "Dr. Rohit Verma", "specialty": "Orthodontics", "days": "Tue, Thu, Sat"},
						)
					],
				),
				lx.data.ExampleData(
					text="A ₹500 booking fee confirms every appointment and is adjusted against the treatment bill. Reschedule or cancel free of charge up to 4 hours before the appointment.",
					extractions=[
						lx.data.Extraction(
							extraction_class="policy",
							extraction_text="A ₹500 booking fee confirms every appointment and is adjusted against the treatment bill",
							attributes={"fee": "₹500", "adjusted": "true", "cancellation_notice": "4 hours"},
						)
					],
				),
			]
		except Exception as exc:
			log.warning("Could not construct langextract ExampleData: %s", exc)
			return None

	def _resolve_faqs_path(self) -> Path:
		configured = getattr(self.s, "KB_FAQS_PATH", "./app/knowledge/clinic_faqs.md")
		candidates = [
			Path(configured),
			Path(__file__).resolve().parent.parent / "knowledge" / "clinic_faqs.md",
			Path.cwd() / "backend" / "app" / "knowledge" / "clinic_faqs.md",
			Path.cwd() / "app" / "knowledge" / "clinic_faqs.md",
		]
		for c in candidates:
			if c.exists() and c.is_file():
				return c
		return candidates[1]

	def _load_faqs_text(self) -> str:
		if self.faqs_path.exists():
			try:
				return self.faqs_path.read_text(encoding="utf-8").strip()
			except Exception as exc:
				log.error("Failed to read %s: %s", self.faqs_path, exc)
		return ""

	def _parse_sections(self, full_text: str) -> Dict[str, str]:
		"""Split markdown by ## headings for targeted extraction."""
		sections: Dict[str, str] = {}
		current_heading = "general"
		current_lines: List[str] = []

		for line in full_text.splitlines():
			if line.startswith("## "):
				if current_lines:
					sections[current_heading] = "\n".join(current_lines).strip()
					current_lines = []
				current_heading = line.replace("## ", "").strip().lower()
			else:
				current_lines.append(line)

		if current_lines:
			sections[current_heading] = "\n".join(current_lines).strip()
		return sections

	def _select_relevant_text(self, question: str) -> str:
		"""Select the most relevant section of the FAQ, or full document if multi-topic."""
		q = (question or "").lower()
		matched_sections = []

		if any(k in q for k in ("timing", "time", "hour", "open", "close", "sunday", "lunch", "kab khulta", "kitne baje", "holiday")):
			if "timings" in self.sections:
				matched_sections.append(self.sections["timings"])

		if any(k in q for k in ("price", "cost", "charge", "fee", "rate", "rct", "cleaning", "scaling", "whitening", "braces", "aligner", "crown", "implant", "extraction", "teeth", "kitna kharcha", "paise")):
			if "services and prices" in self.sections:
				matched_sections.append(self.sections["services and prices"])

		if any(k in q for k in ("doctor", "dr", "ananya", "rohit", "dentist", "specialist", "qualification", "experience")):
			if "doctors" in self.sections:
				matched_sections.append(self.sections["doctors"])

		if any(k in q for k in ("location", "where", "address", "landmark", "parking", "kahan", "pahunch", "phase 7", "hdfc", "car parking")):
			if "location and parking" in self.sections:
				matched_sections.append(self.sections["location and parking"])

		if any(k in q for k in ("policy", "cancel", "reschedule", "refund", "500", "booking fee", "advance", "rules")):
			if "booking policy" in self.sections:
				matched_sections.append(self.sections["booking policy"])

		if any(k in q for k in ("insurance", "emi", "upi", "card", "cash", "payment", "reimbursement", "claim")):
			if "insurance and payments" in self.sections:
				matched_sections.append(self.sections["insurance and payments"])

		if any(k in q for k in ("prep", "empty stomach", "khana", "coffee", "tea", "blood thinner", "diabetes", "precaution")):
			if "preparation instructions" in self.sections:
				matched_sections.append(self.sections["preparation instructions"])

		if any(k in q for k in ("emergency", "severe", "bleeding", "trauma", "swelling", "accident", "urgent")):
			if "emergencies" in self.sections:
				matched_sections.append(self.sections["emergencies"])

		if matched_sections:
			return "\n\n".join(matched_sections)
		return self.faqs_text

	def extract_clinic_info(self, question: str) -> dict:
		"""Extract grounded facts and direct answers for a user question."""
		clean_q = (question or "").strip()
		if not clean_q:
			return {"found": False, "question": "", "context": ""}

		cache_key = clean_q.lower()
		if cache_key in self._cache:
			return self._cache[cache_key]

		target_text = self._select_relevant_text(clean_q)
		if not target_text:
			target_text = self.faqs_text

		# Attempt 1: Google langextract library
		res = self._try_langextract(clean_q, target_text)
		if res and res.get("found"):
			self._cache[cache_key] = res
			return res

		# Attempt 2: Direct grounded LLM extraction using FAQ context
		res_llm = self._try_grounded_llm(clean_q, target_text)
		if res_llm and res_llm.get("found"):
			self._cache[cache_key] = res_llm
			return res_llm

		# Attempt 3: Return raw matched sections as grounding context
		fallback_res = {
			"found": True,
			"question": clean_q,
			"direct_answer": target_text,
			"context": target_text,
			"source_quotes": [target_text[:300]],
			"method": "section_fallback",
		}
		self._cache[cache_key] = fallback_res
		return fallback_res

	def _try_langextract(self, question: str, target_text: str) -> Optional[dict]:
		"""Execute extraction using Google's langextract library."""
		if not getattr(self.s, "USE_LANGEXTRACT", True):
			return None
		if not self.api_key:
			return None

		prompt_desc = (
			f"You are extracting factual clinic information to answer this caller inquiry:\n"
			f"Query: \"{question}\"\n"
			f"Extract the exact facts, figures, timings, doctor names, or prices directly mentioned in the text. "
			f"Provide exact quote spans from the source text."
		)

		try:
			import langextract as lx
			# Pass api_key directly or via environment
			os.environ["GEMINI_API_KEY"] = self.api_key
			os.environ["LANGEXTRACT_API_KEY"] = self.api_key

			# Invoke langextract with schema constraints & few-shot examples
			annotated_doc = lx.extract(
				text_or_documents=target_text,
				prompt_description=prompt_desc,
				examples=self._examples,
				model_id=self.model_id,
				api_key=self.api_key,
				show_progress=False,
				extraction_passes=1,
			)

			if annotated_doc:
				quotes = []
				extracted_details = []
				# Handle Document / AnnotatedDocument extractions
				doc_items = annotated_doc if isinstance(annotated_doc, list) else [annotated_doc]
				for doc in doc_items:
					extractions = getattr(doc, "extractions", None) or getattr(doc, "annotations", None) or []
					for item in extractions:
						txt = getattr(item, "extraction_text", "") or getattr(item, "text", "") or str(item)
						if txt and txt not in quotes:
							quotes.append(txt)
							extracted_details.append(txt)

				combined_facts = "\n".join(extracted_details) if extracted_details else target_text
				return {
					"found": True,
					"question": question,
					"direct_answer": combined_facts,
					"context": f"Grounded Clinic Knowledge:\n{combined_facts}\nSource Document Section:\n{target_text}",
					"source_quotes": quotes or [target_text[:200]],
					"method": "langextract",
				}
		except Exception as exc:
			log.warning("langextract API call bypassed or raised: %s; falling back to grounded extractor", exc)
			return None

	def _try_grounded_llm(self, question: str, target_text: str) -> Optional[dict]:
		"""Grounded extractor fallback using Gemini or Groq."""
		try:
			from ..services.llm import LLMService
			# Create quick grounded prompt
			prompt = (
				f"You are the knowledge extractor for Bright Dental Clinic.\n"
				f"Based ONLY on this verified Clinic Knowledge Base:\n\"\"\"\n{target_text}\n\"\"\"\n\n"
				f"Answer this caller question accurately and concisely:\n\"{question}\"\n\n"
				f"State the exact facts (prices, timings, doctors, or policies) in 1-2 sentences with zero hallucination."
			)
			# Call LLM
			llm = LLMService()
			messages = [
				{"role": "system", "content": "You extract verified clinic facts with 100% precision."},
				{"role": "user", "content": prompt},
			]
			answer = llm.respond(messages)
			if answer and answer.strip():
				return {
					"found": True,
					"question": question,
					"direct_answer": answer.strip(),
					"context": f"Extracted Facts:\n{answer.strip()}\n---\nRelevant Clinic Section:\n{target_text}",
					"source_quotes": [target_text[:300]],
					"method": "grounded_llm",
				}
		except Exception as exc:
			log.warning("Grounded LLM extraction failed: %s", exc)
		return None
