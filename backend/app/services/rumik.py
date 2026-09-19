"""Rumik AI Text-to-Speech (TTS) Service Module.

Provides integration with Rumik AI's Silk models (Silk Mulberry 1.5 and Silk Muga 1)
for low-latency, natural Indian English and Hindi receptionist voice synthesis.
Pricing: Pay-as-you-go ~₹0.4 per minute.
"""

import base64
import hashlib
import logging
import re
import time
from pathlib import Path
from typing import Optional

import httpx

log = logging.getLogger("rumik")

RUMIK_API_BASE_URL = "https://silk-api.rumik.ai"
RUMIK_TTS_ENDPOINT = f"{RUMIK_API_BASE_URL}/v1/tts"
RUMIK_TTS_JSON_ENDPOINT = f"{RUMIK_API_BASE_URL}/v1/tts/json"

DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "cache" / "rumik_audio"


class RumikTTSService:
	"""Service for converting text to speech using Rumik AI Silk TTS API."""

	DEFAULT_DESCRIPTION = (
		"a female 20s indian receptionist voice, crisp clear articulation, brisk and lively conversational pace, warm and professional"
	)

	DEFAULT_FILLERS = [
		"Ek second, main check karti hoon...",
		"Ji, main abhi dekh kar batati hoon...",
		"Ji, ek minute...",
		"Ek minute, main details check karti hoon...",
	]

	FILLERS_BY_INTENT = {
		"payment": [
			"Ek second, main payment status check karti hoon...",
			"Ji, main abhi payment verify karti hoon...",
			"Ji, main abhi check karti hoon...",
			"Ek minute ji, main transaction confirm karti hoon...",
		],
		"slots": [
			"Wait, main check karti hoon...",
			"Ek second, main available slots check karti hoon...",
			"Ji, main calendar dekh ke batati hoon...",
			"Ji, main abhi timings check karti hoon...",
			"Ek minute ji, main calendar mein slot dekh rahi hoon...",
		],
		"info": [
			"Ek second, main details dekh ke batati hoon...",
			"Ji, main abhi check karke batati hoon...",
			"Ji, ek minute, main details confirm karti hoon...",
			"Ek minute ji, main doctor ka schedule aur details dekh rahi hoon...",
		],
		"short": [
			"Ji, ek minute...",
			"Ji, main abhi dekh ke batati hoon...",
		],
		"listening": [
			"Ji...",
			"Haan ji...",
			"Ji sun rahi hoon...",
			"Ji kahiye...",
		],
		# Backward compatibility aliases
		"heavy": [
			"Ek second, main check karti hoon...",
			"Ji, main abhi check karti hoon...",
		],
	}

	VALID_RUMIK_EMOTION_TAGS = {
		"<laugh>", "<laugh_harder>", "<sigh>", "<chuckle>", "<gasp>", "<angry>",
		"<excited>", "<whisper>", "<cry>", "<scream>", "<sing>", "<snort>",
		"<exhale>", "<gulp>", "<giggle>", "<sarcastic>", "<curious>",
	}

	def __init__(
		self,
		api_key: str = "",
		model: str = "mulberry",
		default_speaker: str = "aisha",
		default_description: Optional[str] = None,
		temperature: float = 0.6,
		default_pace: float = 1.15,
		cache_dir: Optional[Path | str] = None,
	):
		self.api_key = (api_key or "").strip()
		self.model = model or "mulberry"
		self.default_speaker = default_speaker or "aisha"
		self.default_description = default_description or self.DEFAULT_DESCRIPTION
		self.temperature = temperature
		self.default_pace = default_pace
		self._client: Optional[httpx.AsyncClient] = None
		# In-memory LRU audio cache for recurring phrases (greetings, confirmations, fillers)
		self._cache: dict[str, bytes] = {}
		self._cache_b64: dict[str, str] = {}
		self._max_cache_size: int = 128
		self.cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
		try:
			self.cache_dir.mkdir(parents=True, exist_ok=True)
		except Exception as exc:
			log.warning("Could not create Rumik audio cache directory '%s': %s", self.cache_dir, exc)

		self._balance_exhausted_until: float = 0.0

	def _handle_balance_exhaustion(self, status_code: int, text: str) -> None:
		if status_code == 402 or "insufficient_balance" in text.lower() or "insufficient prepaid balance" in text.lower():
			self._balance_exhausted_until = time.time() + 1800.0  # 30-minute cooldown
			log.warning("Rumik TTS reported insufficient balance (402). Disabling Rumik TTS for 30 minutes; switching to Sarvam TTS.")

	def _get_client(self) -> httpx.AsyncClient:
		"""Return persistent httpx.AsyncClient with connection pooling and keep-alive."""
		if self._client is None or self._client.is_closed:
			limits = httpx.Limits(max_keepalive_connections=10, max_connections=20, keepalive_expiry=30.0)
			self._client = httpx.AsyncClient(timeout=12.0, limits=limits)
		return self._client

	async def aclose(self) -> None:
		"""Close underlying persistent HTTP client cleanly."""
		if self._client is not None and not self._client.is_closed:
			await self._client.aclose()
			self._client = None

	@property
	def is_configured(self) -> bool:
		"""Return True if a non-empty API key is present and balance is not exhausted."""
		if time.time() < self._balance_exhausted_until:
			return False
		return bool(self.api_key)

	@classmethod
	def clean_text_for_speech(cls, text: str) -> str:
		"""Strip formatting, markdown, and normalize abbreviations while preserving valid Rumik emotion tags."""
		if not text:
			return ""
		# Strip think blocks
		cleaned = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE)
		cleaned = re.sub(r"<think>[\s\S]*$", "", cleaned, flags=re.IGNORECASE)

		# Strip general HTML tags (<br>, <p>, etc.) while preserving valid Rumik emotion tags
		def _filter_tags(match):
			tag = match.group(0).lower()
			if tag in cls.VALID_RUMIK_EMOTION_TAGS:
				return tag
			return ""
		cleaned = re.sub(r"<[^>]+>", _filter_tags, cleaned)

		# Strip markdown symbols (*, _, #, `, ~, etc.)
		cleaned = re.sub(r"[*_#`~\[\]]", "", cleaned)
		# Strip TOON pipe delimiters if present cleanly
		cleaned = re.sub(r"\s*\|\s*", ", ", cleaned)

		# Phonetic doctor expansion for natural Indian voice pronunciation
		cleaned = re.sub(r"\bDr\.?\s+", "Doctor ", cleaned)

		# Remove the word "baje" / "bje" completely from responses
		if re.search(r"\b(?:baje|bje)\b", cleaned, flags=re.IGNORECASE):
			cleaned = re.sub(r"\bkitne\s+(?:baje|bje)\b", "kis time", cleaned, flags=re.IGNORECASE)
			cleaned = re.sub(r"(\d{1,2}(?::\d{2})?)\s*(AM|PM)\s*(?:baje|bje)+\b", r"\1 \2", cleaned, flags=re.IGNORECASE)
			cleaned = re.sub(r"(\d{1,2}(?::\d{2})?)\s*(?:baje|bje)+\b", r"\1", cleaned, flags=re.IGNORECASE)
			cleaned = re.sub(r"\b(?:baje|bje)\b", "", cleaned, flags=re.IGNORECASE)

		# Deduplicate repeated AM/PM tokens (e.g. '10:00 AM AM' -> '10:00 AM')
		if re.search(r"\b(AM|PM)(?:\s+(?:AM|PM))+\b", cleaned, flags=re.IGNORECASE) or re.search(r"\b(\d{1,2}(?::\d{2})?\s*(?:AM|PM))\s*(?:AM|PM)+\b", cleaned, flags=re.IGNORECASE):
			cleaned = re.sub(r"\b(AM|PM)(?:\s+(?:AM|PM))+\b", r"\1", cleaned, flags=re.IGNORECASE)
			cleaned = re.sub(r"\b(\d{1,2}(?::\d{2})?\s*(?:AM|PM))\s*(?:AM|PM)+\b", r"\1", cleaned, flags=re.IGNORECASE)

		# Deduplicate repeated AM/PM across slot recitations (e.g. '10:00 AM ya 11:30 AM' -> '10:00 ya 11:30 AM')
		if re.search(r"\b(\d{1,2}(?::\d{2})?)\s*(?:AM|PM)\s*(?:aur|ya|,|and|or)\s*(\d{1,2}(?::\d{2})?)\s*(?:AM|PM)\b", cleaned, flags=re.IGNORECASE):
			cleaned = re.sub(r"\b(\d{1,2}(?::\d{2})?)\s*AM\s*(aur|ya|,|and|or)\s*(\d{1,2}(?::\d{2})?)\s*AM\b", r"\1 \2 \3 AM", cleaned, flags=re.IGNORECASE)
			cleaned = re.sub(r"\b(\d{1,2}(?::\d{2})?)\s*PM\s*(aur|ya|,|and|or)\s*(\d{1,2}(?::\d{2})?)\s*PM\b", r"\1 \2 \3 PM", cleaned, flags=re.IGNORECASE)

		# Strip redundant AM/PM following Indian day-part
		if re.search(r"\b(?:subah|dopahar|shaam)\s+\d{1,2}(?::\d{2})?\s*(?:AM|PM)\b", cleaned, flags=re.IGNORECASE):
			cleaned = re.sub(r"\bsubah\s+(\d{1,2}(?::\d{2})?)\s*AM\b", r"subah \1", cleaned, flags=re.IGNORECASE)
			cleaned = re.sub(r"\bdopahar\s+(\d{1,2}(?::\d{2})?)\s*PM\b", r"dopahar \1", cleaned, flags=re.IGNORECASE)
			cleaned = re.sub(r"\bshaam\s+(\d{1,2}(?::\d{2})?)\s*PM\b", r"shaam \1", cleaned, flags=re.IGNORECASE)

		# Collapse whitespace
		cleaned = re.sub(r"\s+", " ", cleaned).strip()
		return cleaned

	def _build_payload(
		self,
		clean_text: str,
		speaker: Optional[str] = None,
		model: Optional[str] = None,
		description: Optional[str] = None,
		audio_format: Optional[str] = None,
		temperature: Optional[float] = None,
	) -> dict:
		"""Construct valid Rumik TTS request dictionary according to Silk API specifications."""
		chosen_model = (model or self.model).lower()
		payload: dict = {
			"model": chosen_model,
			"text": clean_text,
			"temperature": temperature if temperature is not None else self.temperature,
		}

		if chosen_model == "mulberry":
			payload["description"] = description or self.default_description
			spk = (speaker or self.default_speaker).lower()
			if spk:
				payload["speaker"] = spk
		elif chosen_model == "muga":
			# For muga, tone tag steering is in the text (e.g. [happy])
			if not clean_text.startswith("["):
				payload["text"] = f"[friendly] {clean_text}"

		# audio_format: omit for 24kHz WAV, or "mp3", "opus", "pcm", "mulaw", "alaw"
		if audio_format and audio_format.lower() in ("mp3", "opus", "pcm", "mulaw", "alaw"):
			payload["audio_format"] = audio_format.lower()

		return payload

	async def synthesize(
		self,
		text: str,
		speaker: Optional[str] = None,
		model: Optional[str] = None,
		description: Optional[str] = None,
		audio_format: Optional[str] = None,
		temperature: Optional[float] = None,
	) -> bytes:
		"""Convert text to speech and return raw audio bytes (WAV or requested format).
		
		Raises:
			ValueError: If API key is missing or text is empty after sanitization.
			RuntimeError: If Rumik API returns an error or empty audio.
		"""
		if not self.is_configured:
			raise ValueError("Rumik API key is not configured. Set RUMIK_API_KEY in .env.")

		clean_text = self.clean_text_for_speech(text)
		if not clean_text:
			raise ValueError("Text payload is empty after sanitization.")

		# Rumik Silk allows up to 2000 characters per utterance
		clean_text = clean_text[:2000]

		spk = (speaker or self.default_speaker).lower()
		mod = (model or self.model).lower()
		cache_key = f"{clean_text}|{spk}|{mod}|{audio_format}"

		if cache_key in self._cache:
			log.info("Rumik TTS cache hit for '%s...' (0ms latency)", clean_text[:35])
			return self._cache[cache_key]

		# Check persistent disk cache before making network call
		key_hash = hashlib.sha256(cache_key.encode("utf-8")).hexdigest()[:32]
		wav_file = self.cache_dir / f"{key_hash}.wav"
		b64_file = self.cache_dir / f"{key_hash}.b64"

		if wav_file.exists():
			try:
				audio_bytes = wav_file.read_bytes()
				if audio_bytes:
					self._cache[cache_key] = audio_bytes
					if b64_file.exists():
						self._cache_b64[cache_key] = b64_file.read_text("utf-8")
					else:
						b64_str = base64.b64encode(audio_bytes).decode("ascii")
						self._cache_b64[cache_key] = b64_str
						try:
							b64_file.write_text(b64_str, "utf-8")
						except Exception:
							pass
					log.info("Rumik TTS disk cache hit for '%s...' (0ms API cost)", clean_text[:35])
					return audio_bytes
			except Exception as disk_err:
				log.warning("Rumik disk cache read error: %s", disk_err)

		payload = self._build_payload(
			clean_text=clean_text,
			speaker=spk,
			model=mod,
			description=description,
			audio_format=audio_format,
			temperature=temperature,
		)

		headers = {
			"Authorization": f"Bearer {self.api_key}",
			"Content-Type": "application/json",
		}

		try:
			client = self._get_client()
			resp = await client.post(RUMIK_TTS_ENDPOINT, json=payload, headers=headers)
			if resp.status_code != 200:
				self._handle_balance_exhaustion(resp.status_code, resp.text)
				log.error("Rumik TTS error %s: %s", resp.status_code, resp.text)
				raise RuntimeError(f"Rumik API returned HTTP {resp.status_code}: {resp.text}")

			audio_bytes = resp.content
			if not audio_bytes:
				raise RuntimeError("Rumik API returned empty audio content.")

			# Store in LRU cache
			if len(self._cache) >= self._max_cache_size:
				try:
					oldest_key = next(iter(self._cache))
					del self._cache[oldest_key]
					if oldest_key in self._cache_b64:
						del self._cache_b64[oldest_key]
				except Exception:
					pass

			self._cache[cache_key] = audio_bytes
			b64_str = base64.b64encode(audio_bytes).decode("ascii")
			self._cache_b64[cache_key] = b64_str

			# Persist to disk cache
			try:
				wav_file.write_bytes(audio_bytes)
				b64_file.write_text(b64_str, "utf-8")
			except Exception as disk_err:
				log.warning("Rumik disk cache write error: %s", disk_err)

			return audio_bytes
		except httpx.TimeoutException as exc:
			log.error("Rumik TTS request timed out: %s", exc)
			raise RuntimeError("Rumik TTS request timed out.") from exc
		except Exception as exc:
			log.error("Failed to synthesize speech via Rumik: %s", exc)
			raise

	async def synthesize_b64(
		self,
		text: str,
		speaker: Optional[str] = None,
		model: Optional[str] = None,
		description: Optional[str] = None,
		audio_format: Optional[str] = None,
		temperature: Optional[float] = None,
	) -> str:
		"""Convert text to speech and return base64 encoded audio string directly.
		
		Uses Rumik JSON endpoint /v1/tts/json or in-memory cache to save encoding overhead.
		"""
		if not self.is_configured:
			raise ValueError("Rumik API key is not configured. Set RUMIK_API_KEY in .env.")

		clean_text = self.clean_text_for_speech(text)
		if not clean_text:
			raise ValueError("Text payload is empty after sanitization.")

		clean_text = clean_text[:2000]
		spk = (speaker or self.default_speaker).lower()
		mod = (model or self.model).lower()
		cache_key = f"{clean_text}|{spk}|{mod}|{audio_format}"

		if cache_key in self._cache_b64:
			log.info("Rumik TTS b64 cache hit for '%s...' (0ms latency)", clean_text[:35])
			return self._cache_b64[cache_key]

		# Check persistent disk cache before making network call
		key_hash = hashlib.sha256(cache_key.encode("utf-8")).hexdigest()[:32]
		b64_file = self.cache_dir / f"{key_hash}.b64"
		wav_file = self.cache_dir / f"{key_hash}.wav"

		if b64_file.exists():
			try:
				audio_b64 = b64_file.read_text("utf-8")
				if audio_b64:
					self._cache_b64[cache_key] = audio_b64
					if wav_file.exists():
						self._cache[cache_key] = wav_file.read_bytes()
					else:
						audio_bytes = base64.b64decode(audio_b64)
						self._cache[cache_key] = audio_bytes
						try:
							wav_file.write_bytes(audio_bytes)
						except Exception:
							pass
					log.info("Rumik TTS b64 disk cache hit for '%s...' (0ms API cost)", clean_text[:35])
					return audio_b64
			except Exception as disk_err:
				log.warning("Rumik disk cache read error: %s", disk_err)

		payload = self._build_payload(
			clean_text=clean_text,
			speaker=spk,
			model=mod,
			description=description,
			audio_format=audio_format,
			temperature=temperature,
		)

		headers = {
			"Authorization": f"Bearer {self.api_key}",
			"Content-Type": "application/json",
		}

		try:
			client = self._get_client()
			resp = await client.post(RUMIK_TTS_JSON_ENDPOINT, json=payload, headers=headers)
			if resp.status_code == 200:
				data = resp.json()
				audio_b64 = data.get("audio_base64") or ""
				if audio_b64:
					audio_bytes = base64.b64decode(audio_b64)
					if len(self._cache) >= self._max_cache_size:
						try:
							oldest_key = next(iter(self._cache))
							del self._cache[oldest_key]
							if oldest_key in self._cache_b64:
								del self._cache_b64[oldest_key]
						except Exception:
							pass
					self._cache[cache_key] = audio_bytes
					self._cache_b64[cache_key] = audio_b64

					# Persist to disk cache
					try:
						b64_file.write_text(audio_b64, "utf-8")
						wav_file.write_bytes(audio_bytes)
					except Exception as disk_err:
						log.warning("Rumik disk cache write error: %s", disk_err)

					return audio_b64
			else:
				self._handle_balance_exhaustion(resp.status_code, resp.text)

			# Fallback to binary synthesize if JSON endpoint fails
			await self.synthesize(
				text=clean_text,
				speaker=spk,
				model=mod,
				description=description,
				audio_format=audio_format,
				temperature=temperature,
			)
			return self._cache_b64.get(cache_key, "")
		except httpx.TimeoutException as exc:
			log.error("Rumik TTS JSON request timed out: %s", exc)
			raise RuntimeError("Rumik TTS request timed out.") from exc
		except Exception as exc:
			log.error("Failed to synthesize speech via Rumik JSON endpoint: %s", exc)
			raise

	def get_filler_audio(self, index: int = 0) -> Optional[dict]:
		"""Return pre-warmed filler audio dictionary with text and audio_b64 for 0ms latency."""
		if not self.is_configured:
			return None
		phrase = self.DEFAULT_FILLERS[index % len(self.DEFAULT_FILLERS)]
		clean = self.clean_text_for_speech(phrase)
		spk = self.default_speaker.lower()
		mod = self.model.lower()
		cache_key = f"{clean}|{spk}|{mod}|None"
		b64 = self._cache_b64.get(cache_key)
		if b64:
			return {"text": phrase, "audio_b64": b64, "tts_provider": "rumik"}
		return None

	def get_contextual_filler_audio(self, intent: str = "slots", last_phrase: str = "") -> Optional[dict]:
		"""Return pre-warmed filler audio dictionary based on context intent, avoiding repeating last_phrase."""
		if not self.is_configured:
			return None
		candidates = self.FILLERS_BY_INTENT.get(intent) or self.FILLERS_BY_INTENT.get("slots") or self.DEFAULT_FILLERS
		if isinstance(candidates, str):
			candidates = [candidates]

		# Filter out last_phrase to avoid repeating the exact same filler back-to-back
		options = [p for p in candidates if p != last_phrase]
		if not options:
			options = candidates

		spk = self.default_speaker.lower()
		mod = self.model.lower()

		for phrase in options:
			clean = self.clean_text_for_speech(phrase)
			cache_key = f"{clean}|{spk}|{mod}|None"
			b64 = self._cache_b64.get(cache_key)
			if b64:
				return {"text": phrase, "audio_b64": b64, "tts_provider": "rumik"}

		# Look for any available cached filler in DEFAULT_FILLERS that doesn't match last_phrase
		for fallback_phrase in self.DEFAULT_FILLERS:
			if fallback_phrase == last_phrase:
				continue
			clean = self.clean_text_for_speech(fallback_phrase)
			cache_key = f"{clean}|{spk}|{mod}|None"
			b64 = self._cache_b64.get(cache_key)
			if b64:
				return {"text": fallback_phrase, "audio_b64": b64, "tts_provider": "rumik"}

		# Look for ANY cached filler audio that doesn't match last_phrase
		for k, b64 in self._cache_b64.items():
			candidate_text = k.split("|")[0]
			if candidate_text and candidate_text != last_phrase:
				return {"text": candidate_text, "audio_b64": b64, "tts_provider": "rumik"}

		return self.get_filler_audio(0)

	async def warmup(self, phrases: list[str]) -> None:
		"""Pre-warm Rumik TTS cache with frequent phrases to ensure 0ms first-turn latency.
		Leverages persistent disk cache to avoid redundant network API calls on server reloads.
		"""
		all_phrases = list(phrases)
		for filler in self.DEFAULT_FILLERS:
			if filler not in all_phrases:
				all_phrases.append(filler)
		for intent_list in self.FILLERS_BY_INTENT.values():
			for p in intent_list:
				if p not in all_phrases:
					all_phrases.append(p)

		spk = self.default_speaker.lower()
		mod = self.model.lower()

		for phrase in all_phrases:
			clean = self.clean_text_for_speech(phrase)
			if not clean:
				continue
			cache_key = f"{clean}|{spk}|{mod}|None"
			if cache_key in self._cache_b64:
				continue

			# Check disk cache first — if present, load immediately without network call
			key_hash = hashlib.sha256(cache_key.encode("utf-8")).hexdigest()[:32]
			b64_file = self.cache_dir / f"{key_hash}.b64"
			wav_file = self.cache_dir / f"{key_hash}.wav"
			if b64_file.exists():
				try:
					b64_content = b64_file.read_text("utf-8")
					if b64_content:
						self._cache_b64[cache_key] = b64_content
						if wav_file.exists():
							self._cache[cache_key] = wav_file.read_bytes()
						log.debug("Rumik warmup loaded from disk (0 API calls): '%s...'", clean[:35])
						continue
				except Exception:
					pass

			try:
				await self.synthesize_b64(clean)
				log.info("Pre-warmed Rumik TTS cache for: '%s...'", clean[:40])
			except Exception as exc:
				log.warning("Rumik TTS warmup skipped for '%s...': %s", clean[:40], exc)
				if "402" in str(exc) or "insufficient_balance" in str(exc) or "insufficient_quota" in str(exc):
					log.info("Rumik account has insufficient prepaid balance; skipping remaining warmup.")
					break
