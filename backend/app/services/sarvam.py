"""Sarvam AI Text-to-Speech (TTS) Service Module.

Provides integration with Sarvam's Bulbul V3 model for natural, studio-quality
Indian English and regional voice generation.
"""

import base64
import hashlib
import logging
from pathlib import Path
import re
from typing import Optional

import httpx

log = logging.getLogger("sarvam")

SARVAM_TTS_API_URL = "https://api.sarvam.ai/text-to-speech"
DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "cache" / "sarvam_audio"

SARVAM_BULBUL_SPEAKERS = {
	"aditya", "ritu", "ashutosh", "priya", "neha", "rahul", "pooja", "rohan", "simran",
	"kavya", "amit", "dev", "ishita", "shreya", "ratan", "varun", "manan", "sumit",
	"roopa", "kabir", "aayan", "shubh", "advait", "anand", "tanya", "tarun", "sunny",
	"mani", "gokul", "vijay", "shruti", "suhani", "mohit", "kavitha", "rehan", "soham", "rupali"
}


class SarvamTTSService:
	"""Service for converting text to speech using Sarvam AI Bulbul V3 API."""

	def __init__(
		self,
		api_key: str = "",
		default_speaker: str = "simran",
		default_language: str = "hi-IN",
		model: str = "bulbul:v3",
		default_pace: float = 1.0,
		cache_dir: Optional[Path | str] = None,
	):
		self.api_key = (api_key or "").strip()
		self.default_speaker = default_speaker or "simran"
		self.default_language = default_language or "hi-IN"
		self.model = model
		self.default_pace = default_pace
		self._client: Optional[httpx.AsyncClient] = None
		# In-memory LRU audio cache for recurring prompts (greeting, payment gate, farewells)
		self._cache: dict[str, bytes] = {}
		self._cache_b64: dict[str, str] = {}
		self._max_cache_size: int = 128
		self.cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
		try:
			self.cache_dir.mkdir(parents=True, exist_ok=True)
		except Exception as exc:
			log.warning("Could not create Sarvam audio cache directory '%s': %s", self.cache_dir, exc)

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
		"""Return True if a non-empty API key is present."""
		return bool(self.api_key)

	def _validate_speaker(self, speaker: Optional[str]) -> str:
		"""Ensure speaker is recognized by Sarvam bulbul:v3, defaulting safely to simran."""
		spk = (speaker or self.default_speaker or "simran").strip().lower()
		if spk not in SARVAM_BULBUL_SPEAKERS:
			fallback = self.default_speaker.lower() if (self.default_speaker and self.default_speaker.lower() in SARVAM_BULBUL_SPEAKERS) else "simran"
			log.info("Speaker '%s' not valid for Sarvam Bulbul; auto-substituting '%s'", spk, fallback)
			return fallback
		return spk

	@staticmethod
	def clean_text_for_speech(text: str) -> str:
		"""Strip formatting, markdown, and normalize abbreviations for natural phonetic speech."""
		if not text:
			return ""
		# Strip think blocks
		cleaned = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE)
		cleaned = re.sub(r"<think>[\s\S]*$", "", cleaned, flags=re.IGNORECASE)
		# Strip html tags
		cleaned = re.sub(r"<[^>]+>", "", cleaned)
		# Strip markdown symbols (*, _, #, `, ~, etc.)
		cleaned = re.sub(r"[*_#`~\[\]]", "", cleaned)
		# Strip TOON pipe delimiters if present cleanly
		cleaned = re.sub(r"\s*\|\s*", ", ", cleaned)

		# Phonetic doctor expansion for natural Indian voice pronunciation
		cleaned = re.sub(r"\bDr\.?\s+", "Doctor ", cleaned)

		# Phonetic doctor name pronunciation fixes for Sarvam Bulbul:v3
		# "Ananya" in Latin script triggers an unnatural elongated pause / stutter ("Anannn..yaa") in Sarvam's English phonemizer.
		# Mapping to Devanagari "अनन्या" and "शर्मा" produces natural, smooth, pause-free Hindi pronunciation.
		cleaned = re.sub(r"\bAnanya\b", "अनन्या", cleaned, flags=re.IGNORECASE)
		cleaned = re.sub(r"\bSharma\b", "शर्मा", cleaned, flags=re.IGNORECASE)

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

	async def synthesize(
		self,
		text: str,
		speaker: Optional[str] = None,
		target_language_code: Optional[str] = None,
		pace: Optional[float] = None,
	) -> bytes:
		"""Convert text to speech and return raw WAV audio bytes.
		
		Raises:
			ValueError: If API key is missing or text is empty.
			RuntimeError: If Sarvam API call fails or returns empty audio.
		"""
		if not self.is_configured:
			raise ValueError("Sarvam API key is not configured. Set SARVAM_API_KEY in .env.")

		clean_text = self.clean_text_for_speech(text)
		if not clean_text:
			raise ValueError("Text payload is empty after sanitization.")

		# Sarvam Bulbul V3 allows up to 2500 characters
		clean_text = clean_text[:2500]

		target_lang = target_language_code or self.default_language
		spk = self._validate_speaker(speaker)
		p = pace if pace is not None else self.default_pace

		# Cost & Latency Optimization: Check LRU audio cache for recurring phrases
		cache_key = f"{clean_text}|{spk}|{target_lang}|{p}"
		if cache_key in self._cache:
			log.info("Sarvam TTS cache hit for '%s...' (saving API cost & 0ms latency)", clean_text[:35])
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
					log.info("Sarvam TTS disk cache hit for '%s...' (0ms API cost)", clean_text[:35])
					return audio_bytes
			except Exception as disk_err:
				log.warning("Sarvam disk cache read error: %s", disk_err)

		payload = {
			"inputs": [clean_text],
			"target_language_code": target_lang,
			"speaker": spk,
			"model": self.model,
			"pace": p,
		}

		headers = {
			"api-subscription-key": self.api_key,
			"Content-Type": "application/json",
		}

		try:
			client = self._get_client()
			resp = await client.post(SARVAM_TTS_API_URL, json=payload, headers=headers)
			if resp.status_code != 200:
				log.error("Sarvam TTS error %s: %s", resp.status_code, resp.text)
				raise RuntimeError(f"Sarvam API returned HTTP {resp.status_code}: {resp.text}")

			data = resp.json()
			audios = data.get("audios")
			if not audios or not isinstance(audios, list) or len(audios) == 0:
				raise RuntimeError("Sarvam API returned no audio data.")

			audio_b64 = audios[0]
			audio_bytes = base64.b64decode(audio_b64)

			# Store in cache (evicting oldest entry if full)
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
				wav_file.write_bytes(audio_bytes)
				b64_file.write_text(audio_b64, "utf-8")
			except Exception as disk_err:
				log.warning("Sarvam disk cache write error: %s", disk_err)

			return audio_bytes
		except httpx.TimeoutException as exc:
			log.error("Sarvam TTS request timed out: %s", exc)
			raise RuntimeError("Sarvam TTS request timed out.") from exc
		except Exception as exc:
			log.error("Failed to synthesize speech via Sarvam: %s", exc)
			raise

	async def synthesize_b64(
		self,
		text: str,
		speaker: Optional[str] = None,
		target_language_code: Optional[str] = None,
		pace: Optional[float] = None,
	) -> str:
		"""Convert text to speech and return base64 encoded audio string directly.
		
		Saves client-side base64 encoding overhead and leverages memory cache.
		"""
		if not self.is_configured:
			raise ValueError("Sarvam API key is not configured. Set SARVAM_API_KEY in .env.")

		clean_text = self.clean_text_for_speech(text)
		if not clean_text:
			raise ValueError("Text payload is empty after sanitization.")

		clean_text = clean_text[:2500]
		target_lang = target_language_code or self.default_language
		spk = self._validate_speaker(speaker)
		p = pace if pace is not None else self.default_pace

		cache_key = f"{clean_text}|{spk}|{target_lang}|{p}"
		if cache_key in self._cache_b64:
			log.info("Sarvam TTS b64 cache hit for '%s...' (0ms latency)", clean_text[:35])
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
					log.info("Sarvam TTS b64 disk cache hit for '%s...' (0ms API cost)", clean_text[:35])
					return audio_b64
			except Exception as disk_err:
				log.warning("Sarvam disk cache read error: %s", disk_err)

		# Calling synthesize populates both _cache, _cache_b64, and disk cache
		await self.synthesize(clean_text, speaker=spk, target_language_code=target_lang, pace=p)
		return self._cache_b64.get(cache_key, "")

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

	def get_filler_audio(self, index: int = 0) -> Optional[dict]:
		"""Return pre-warmed filler audio dictionary with text and audio_b64 for 0ms latency."""
		if not self.is_configured:
			return None
		phrase = self.DEFAULT_FILLERS[index % len(self.DEFAULT_FILLERS)]
		clean = self.clean_text_for_speech(phrase)
		spk = self.default_speaker
		lang = self.default_language
		p = self.default_pace
		cache_key = f"{clean}|{spk}|{lang}|{p}"
		b64 = self._cache_b64.get(cache_key)
		if b64:
			return {"text": phrase, "audio_b64": b64, "tts_provider": "sarvam"}
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

		spk = self.default_speaker
		lang = self.default_language
		p = self.default_pace

		for phrase in options:
			clean = self.clean_text_for_speech(phrase)
			cache_key = f"{clean}|{spk}|{lang}|{p}"
			b64 = self._cache_b64.get(cache_key)
			if b64:
				return {"text": phrase, "audio_b64": b64, "tts_provider": "sarvam"}

		# Look for any available cached filler in DEFAULT_FILLERS that doesn't match last_phrase
		for fallback_phrase in self.DEFAULT_FILLERS:
			if fallback_phrase == last_phrase:
				continue
			clean = self.clean_text_for_speech(fallback_phrase)
			cache_key = f"{clean}|{spk}|{lang}|{p}"
			b64 = self._cache_b64.get(cache_key)
			if b64:
				return {"text": fallback_phrase, "audio_b64": b64, "tts_provider": "sarvam"}

		# Look for ANY cached filler audio that doesn't match last_phrase
		for k, b64 in self._cache_b64.items():
			candidate_text = k.split("|")[0]
			if candidate_text and candidate_text != last_phrase:
				return {"text": candidate_text, "audio_b64": b64, "tts_provider": "sarvam"}

		return self.get_filler_audio(0)

	async def warmup(self, phrases: list[str]) -> None:
		"""Pre-warm Sarvam TTS cache with frequent phrases to ensure 0ms first-turn latency.
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

		spk = self.default_speaker
		lang = self.default_language
		p = self.default_pace

		for phrase in all_phrases:
			clean = self.clean_text_for_speech(phrase)
			if not clean:
				continue
			cache_key = f"{clean}|{spk}|{lang}|{p}"
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
						log.debug("Sarvam warmup loaded from disk (0 API calls): '%s...'", clean[:35])
						continue
				except Exception:
					pass

			try:
				await self.synthesize_b64(clean)
				log.info("Pre-warmed Sarvam TTS cache for: '%s...'", clean[:40])
			except Exception as exc:
				log.warning("Sarvam TTS warmup skipped for '%s...': %s", clean[:40], exc)
				if "402" in str(exc) or "insufficient_quota" in str(exc):
					log.info("Sarvam account has no available credits; skipping remaining warmup.")
					break

