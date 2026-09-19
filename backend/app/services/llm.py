import asyncio
import json
import logging
import re
import time
from datetime import date, timedelta
from typing import Optional

import httpx
from groq import Groq

from ..agent.guardrails import get_deescalation_response, is_angry_or_frustrated, is_prompt_injection
from ..config import get_settings


log = logging.getLogger("llm")


def _strip_think_tags(text: str) -> str:
	"""Strip <think>...</think> reasoning blocks emitted by models like Qwen or DeepSeek."""
	if not text:
		return ""
	cleaned = re.sub(r"<think>[\s\S]*?</think>", "", text)
	cleaned = re.sub(r"<think>[\s\S]*$", "", cleaned)
	return cleaned.strip()


def _extract_json_object(raw_text: str) -> Optional[dict]:
	"""Robustly extract and parse the first valid JSON object from LLM output, ignoring markdown fences or extra commentary."""
	if not raw_text:
		return None
	cleaned = _strip_think_tags(raw_text).strip()
	try:
		return json.loads(cleaned)
	except Exception:
		pass
	fence_match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", cleaned, re.IGNORECASE)
	if fence_match:
		try:
			return json.loads(fence_match.group(1).strip())
		except Exception:
			pass
	brace_match = re.search(r"(\{[\s\S]*\})", cleaned)
	if brace_match:
		candidate = brace_match.group(1).strip()
		try:
			return json.loads(candidate)
		except Exception:
			last_idx = candidate.rfind("}")
			if last_idx > 0:
				try:
					return json.loads(candidate[: last_idx + 1])
				except Exception:
					pass
	return None


INCOMPLETE_TRAILING_WORDS = (
	"ki", "aur", "lekin", "par", "toh", "actually", "jaise", "matlab",
	"mujhe", "mera", "meri", "mere", "humein", "humara", "kuch", "ek", "main",
	"ko", "ka", "ke", "se", "mein",
	"कि", "और", "लेकिन", "पर", "तो", "एक्चुअली", "जैसे", "मतलब",
	"मुझे", "मेरा", "मेरी", "मेरे", "हमें", "हमारा", "कुछ", "एक", "मैं",
	"को", "का", "के", "से", "में",
)

HESITATION_PHRASES = (
	"मुझे एक्चुअली", "mujhe actually", "actually mujhe", "एक्चुअली मुझे",
	"ek minute", "एक मिनट", "ek second", "एक सेकंड",
	"meri baat suno", "मेरी बात सुनो", "meri baat suniye", "मेरी बात सुनिए",
	"suno", "सुनो", "suniye", "सुनिए", "haan suniye", "हां सुनिए",
	"dekho mujhe", "देखो मुझे", "dekho", "देखो",
)

TRAILING_INCOMPLETE_PHRASES = (
	"bol raha hoon ki", "bol raha tha ki", "keh raha tha ki", "chahiye tha ki",
	"ki main", "ki mujhe", "bol raha hoon", "bol raha tha",
	"बोल रहा हूं कि", "बोल रहा था कि", "कह रहा था कि", "चाहिए था कि",
	"कि मैं", "कि मुझे", "बोल रहा हूं", "बोल रहा था",
)

# Dynamic model migration and aliasing table for Google Gemini API
_MODEL_ALIASES: dict[str, str] = {
	"gemini-3.5-flash": "gemini-3.5-flash",
	"gemini-3.1-flash-lite": "gemini-3.1-flash-lite",
	"gemini-2.5-flash": "gemini-2.5-flash",
	"gemini-2.5-flash-lite": "gemini-2.5-flash-lite",
	"gemini-2.0-flash": "gemini-2.5-flash",
	"gemini-2.0-flash-lite": "gemini-2.5-flash-lite",
}
_RESOLVED_MODELS: dict[str, str] = {}

STRAY_NOISE_OR_FILLER = {
	"अह", "अह अह", "अह अह।", "अहा", "तेन", "हम्म", "हम", "hmm", "uh", "um", "ah",
	"oh", "oho", "shh", "huh", "eh", "er", "haa", "ha", "haan", "हाँ", "हां",
}

ACTIONABLE_TERMS = (
	# Dental / Oral clinical keywords (English & transliterated)
	"dental", "dentist", "teeth", "tooth", "toothache", "toothpain", "tooth-pain",
	"gums", "gum", "masude", "masuda", "jaw", "jabda", "jabde", "mouth", "oral",
	"rct", "root canal", "braces", "aligner", "aligners", "whitening", "implant", "implants",
	"cavity", "filling", "bleeding", "bleed", "swelling", "daant", "daanto", "keeda", "keede",
	# Problem / symptom / care keywords (specific to dental/clinic)
	"issue", "issues", "problem", "problems", "fix", "fixing", "fix kar",
	"daant dard", "daant mein dard", "teeth pain", "tooth pain",
	"takleef", "pareshani", "dikkat", "checkup", "check", "consult", "consultation",
	"treatment", "treatments", "care", "cure",
	# Scheduling & appointments
	"book", "booking", "slot", "slots", "appointment", "appoint", "schedule", "reschedule",
	"cancel", "timing", "timings", "available", "availability", "time", "date",
	# Doctors
	"doctor", "dr.", "dr ", "ananya", "rohit", "sharma", "verma", "specialist", "surgeon",
	# Clinic, pricing & location
	"clinic", "hospital", "bright dental", "price", "charge", "charges", "cost", "fee", "fees",
	"kharch", "kharcha", "rate", "address", "location", "kahan", "phase 7", "mohali",
	# Payment
	"pay", "payment", "paid", "upi", "gpay", "phonepe", "paytm", "transfer",
	# Questions & doubts
	"sawaal", "sawal", "doubt", "query", "question", "poochna", "poochhna",
	# Devanagari Dental / Clinical terms
	"डेंटल", "क्लीनिक", "क्लिनिक", "अस्पताल",
	"दांत", "दांतों", "दाँत", "टूथ", "टूथपेन", "टूथ-पेन", "टूथ पेन", "टूथएक",
	"मसूड़े", "मसूड़ों", "जबड़ा", "जबड़े", "मुंह", "मुँह", "कैविटी", "कीड़ा", "कीड़े",
	"रूट कैनाल", "आरसीटी", "ब्रेसेस", "वाइटनिंग", "इम्प्लांट", "फिलिंग", "खून", "ब्लीडिंग",
	"सूजन", "अकल दाढ़", "दाढ़",
	# Devanagari Issues / Symptoms / Treatments
	"इशू", "इशूज", "इश्यूज", "प्रॉब्लम", "प्रॉब्लम्स", "समस्या", "समस्याएं", "दिक्कत", "दिक्कतें",
	"तकलीफ", "दांत में दर्द", "दांत दर्द", "फिक्स", "इलाज", "उपचार", "परामर्श", "कंसल्ट", "कंसल्टेशन",
	"चेकअप", "चेक", "दिखाना", "दिखाना है", "दिखाना था", "दिखाना चाहता", "दिखाना चाहती",
	# Devanagari Scheduling & Doctors
	"अपॉइंटमेंट", "स्लॉट", "स्लॉट्स", "बुक", "बुकिंग", "तारीख", "समय", "टाइम", "टाइमिंग",
	"डॉक्टर", "डॉ", "रोहित", "अनन्या", "शर्मा", "वर्मा", "स्पेशलिस्ट", "सर्जन",
	# Devanagari Clinic / Pricing / Location
	"फीस", "फी", "खर्च", "खर्चा", "चार्ज", "चार्जिस", "रेट", "पता", "लोकेशन", "मोहाली",
	# Devanagari Payment
	"पे", "पेमेंट", "पैसे", "रुपये", "ट्रांसफर", "यूपीआई",
	# Devanagari Questions
	"सवाल", "डाउट", "क्वेरी", "पूछना"
)


def get_fast_chit_chat_response(
	category: str,
	user_text: str = "",
	clinic_name: str = "Bright Dental Clinic",
	provider: str = "sarvam",
	has_pending_booking: bool = False,
	has_confirmed_booking: bool = False,
) -> Optional[str]:
	"""Generate instantaneous natural replies for greetings, chit-chat, acknowledgments, compliments, and farewells."""
	lowered = (user_text or "").lower()
	is_hindi_prov = provider in ("sarvam", "rumik")
	if category == "greeting":
		return (
			f"Namaste ji! {clinic_name} mein aapka swagat hai. Main Niaa bol rahi hoon, batayein main aapki kya madad kar sakti hoon?"
			if is_hindi_prov
			else f"Hello! Welcome to {clinic_name}. I am Niaa, how can I assist you today?"
		)
	if category == "chitchat":
		return (
			f"Main bilkul theek hoon ji, poochne ke liye shukriya! Batayein, {clinic_name} mein main aapki kaise madad kar sakti hoon?"
			if is_hindi_prov
			else "I am doing well, thank you for asking! How may I assist you today?"
		)
	if category == "compliment":
		if is_hindi_prov:
			if has_pending_booking:
				return (
					f"Aapka bahut-bahut shukriya ji taareef ke liye! Aapka slot reserve ho chuka hai, kripya screen se payment complete kar lijiye. Kya clinic ke baare mein koi aur sawaal hai?"
				)
			return (
				f"Aapka bahut-bahut shukriya ji taareef ke liye! Batayein, {clinic_name} mein main aapki dental care ya appointment ke liye kaise madad kar sakti hoon?"
			)
		return (
			"Thank you so much for the kind compliment! How may I assist you with your dental care or appointment today?"
		)
	if category == "farewell":
		if has_pending_booking:
			return (
				f"Aapka appointment slot hold par hai ji! Kripya screen par diye gaye link se payment complete karke booking confirm kar lijiye. {clinic_name} mein call karne ke liye aapka bahut-bahut dhanyawad, have a wonderful day!"
				if is_hindi_prov
				else f"Your appointment slot is held. Please complete the booking fee via the link on your screen to confirm. Thank you for calling {clinic_name}, have a wonderful day!"
			)
		if has_confirmed_booking:
			return (
				f"Aapka appointment confirm ho chuka hai ji! {clinic_name} mein call karne ke liye aapka bahut-bahut dhanyawad, apna aur apni smile ka khayal rakhiye. Have a wonderful day ahead!"
				if is_hindi_prov
				else f"Your appointment is confirmed! Thank you so much for calling {clinic_name}. Take wonderful care of your smile, and have a lovely day ahead!"
			)
		return (
			f"Ji bilkul koi baat nahi! {clinic_name} mein call karne ke liye aapka bahut-bahut dhanyawad ji. Apna aur apni smile ka achhi tarah khayal rakhiye. Have a wonderful day ahead!"
			if is_hindi_prov
			else f"Thank you so much for calling {clinic_name}! Please feel free to reach out anytime. Have a wonderful and healthy day ahead!"
		)
	if category == "latency_check":
		return (
			f"Ji main bilkul dhyan se aapko sun rahi hoon! Main jaldi se help karne ki koshish kar rahi hoon ji, kripya batayein aapko appointments ya treatments ke baare mein kya jaankari chahiye?"
			if is_hindi_prov
			else "I am listening closely! How may I assist you with your dental care or scheduling today?"
		)
	if category == "identity":
		return (
			f"Ji main {clinic_name} ki AI receptionist Niaa bol rahi hoon. Main doctor appointments book karne, clinic timings aur dental treatments ki jaankari dene ke liye yahan hoon."
			if is_hindi_prov
			else f"I am Niaa, the AI receptionist at {clinic_name}. I can assist you with booking appointments and dental treatment inquiries."
		)
	if category == "acknowledgment":
		if has_pending_booking:
			return (
				f"Ji bilkul! Agar clinic ya doctor se judi koi aur baat poochni ho toh batayein, varna screen par diye gaye link se payment complete karke slot confirm kar sakte hain."
				if is_hindi_prov
				else "Certainly! If you have any other questions please let me know, or you can complete the booking fee on screen to confirm your slot."
			)
		if any(w in lowered for w in ("shukriya", "dhanyavaad", "dhanyawad", "thank", "thanks", "शुक्रिया", "धन्यवाद", "थैंक", "थैंक्स")):
			return (
				f"Aapka bahut-bahut swagat hai ji! Agar clinic ya doctor se judi koi aur help chahiye toh zaroor batayein."
				if is_hindi_prov
				else "You are most welcome! Please let me know if you need anything else regarding your dental appointments."
			)
		else:
			return (
				f"Ji bilkul! Kya main aapke liye koi appointment schedule kar doon ya koi aur sawaal hai?"
				if is_hindi_prov
				else "Certainly! Would you like me to schedule an appointment or answer any questions?"
			)
	if category == "non_dental":
		return (
			f"Main {clinic_name} ki AI receptionist hoon ji, isliye main sirf dental treatments, timings aur appointments se jude sawalon mein madad kar sakti hoon. Kya aapko clinic ke baare mein koi aur jaankari chahiye?"
			if is_hindi_prov
			else f"I am the AI receptionist at {clinic_name}. I can only assist with dental treatments, clinic hours, and appointments. How may I help you with your dental care?"
		)
	if category == "complaint":
		return get_deescalation_response(clinic_name=clinic_name, provider=provider, user_text=user_text)

	# Non-actionable conversation fallback
	return (
		f"Ji batayein! {clinic_name} mein main aapki dental care ya doctor appointment ke liye kaise madad kar sakti hoon?"
		if is_hindi_prov
		else f"Welcome to {clinic_name}. How may I assist you with your dental care today?"
	)


class LLMService:
	"""Unified LLM service supporting Google Gemini 3.5 Flash / 3.1 Flash-Lite and Groq with automatic failover and circuit-breaker."""

	def __init__(self, observability=None):
		self.s = get_settings()
		self.obs = observability
		self._client = Groq(api_key=self.s.GROQ_API_KEY) if self.s.GROQ_API_KEY else None
		self._gemini_client: Optional[httpx.Client] = None
		self._groq_cooldown_until: float = 0.0

	async def aclose(self):
		"""Close persistent HTTP clients gracefully."""
		if self._gemini_client and not self._gemini_client.is_closed:
			self._gemini_client.close()
			self._gemini_client = None

	def _get_gemini_client(self) -> httpx.Client:
		"""Return persistent httpx.Client for Gemini with keep-alive connections to eliminate SSL handshake overhead."""
		if self._gemini_client is None or self._gemini_client.is_closed:
			limits = httpx.Limits(max_keepalive_connections=20, max_connections=40, keepalive_expiry=60.0)
			self._gemini_client = httpx.Client(timeout=12.0, limits=limits)
		return self._gemini_client

	def _is_groq_available(self) -> bool:
		"""Return True if Groq client is configured and not currently under cooldown from 429 quota exhaustion."""
		return bool(self._client) and (time.time() >= self._groq_cooldown_until)

	def _trigger_groq_cooldown(self, seconds: float = 90.0, reason: str = ""):
		"""Temporarily disable Groq calls to avoid wasting latency on guaranteed 429s."""
		self._groq_cooldown_until = time.time() + seconds
		log.warning("Groq circuit-breaker active for %.0fs (reason: %s) — failing over directly to Gemini", seconds, reason or "quota/rate limit")

	def _call_gemini(
		self,
		messages,
		json_mode: bool = False,
		model: Optional[str] = None,
		max_tokens: Optional[int] = None,
		temperature: Optional[float] = None,
	) -> str:
		"""Direct Google Gemini call for routing, responses, and classification."""
		api_key = getattr(self.s, "GEMINI_API_KEY", "")
		if not api_key:
			return ""

		target_model = model or getattr(self.s, "GEMINI_MODEL", "gemini-2.5-flash")
		effective_model = _RESOLVED_MODELS.get(target_model, target_model)
		url = f"https://generativelanguage.googleapis.com/v1beta/models/{effective_model}:generateContent?key={api_key}"

		system_parts = []
		contents = []

		for m in messages:
			role = m.get("role")
			content = (m.get("content") or "").strip()
			if not content:
				continue
			if role == "system":
				system_parts.append({"text": content})
			elif role == "user":
				contents.append({"role": "user", "parts": [{"text": content}]})
			elif role in ("assistant", "model"):
				contents.append({"role": "model", "parts": [{"text": content}]})

		if not contents:
			return ""

		# Merge consecutive same-role messages for Gemini API compliance
		merged_contents = []
		for item in contents:
			if merged_contents and merged_contents[-1]["role"] == item["role"]:
				merged_contents[-1]["parts"][0]["text"] += "\n" + item["parts"][0]["text"]
			else:
				merged_contents.append(item)

		payload: dict = {"contents": merged_contents}
		if system_parts:
			payload["system_instruction"] = {"parts": system_parts}

		config: dict = {
			"temperature": temperature if temperature is not None else (0.0 if json_mode else 0.1),
			"maxOutputTokens": max_tokens or (128 if json_mode else 160),
		}
		if "lite" not in effective_model.lower():
			config["thinkingConfig"] = {"thinkingBudget": 0}
		if json_mode:
			config["responseMimeType"] = "application/json"
		payload["generationConfig"] = config

		try:
			client = self._get_gemini_client()
			resp = client.post(url, json=payload)
			# If Google returns 404 for retired model versions, auto-migrate to recommended alias
			if resp.status_code == 404 and target_model in _MODEL_ALIASES and effective_model != _MODEL_ALIASES[target_model]:
				fallback = _MODEL_ALIASES[target_model]
				_RESOLVED_MODELS[target_model] = fallback
				log.info("Gemini model '%s' unavailable/retired, auto-migrated to '%s'", target_model, fallback)
				return self._call_gemini(messages, json_mode=json_mode, model=fallback, max_tokens=max_tokens, temperature=temperature)
			# If 400 due to thinkingConfig, retry without thinking
			if resp.status_code == 400 and "thinkingConfig" in config:
				config.pop("thinkingConfig", None)
				resp = client.post(url, json=payload)
			resp.raise_for_status()
			data = resp.json()
			candidates = data.get("candidates") or []
			if candidates and "content" in candidates[0]:
				parts = candidates[0]["content"].get("parts") or []
				if parts and "text" in parts[0]:
					return parts[0]["text"].strip()
			return ""
		except Exception as exc:
			log.warning("Gemini call '%s' failed: %s", target_model, exc)
			return ""

	def decide(self, messages) -> dict:
		primary_provider = getattr(self.s, "PRIMARY_LLM_PROVIDER", "gemini").lower()
		obs_model = (
			self.s.GEMINI_MODEL if primary_provider == "gemini"
			else (self.s.GROQ_ROUTER_MODEL if self._client else "keyword-dev-router")
		)
		obs_ctx = (
			self.obs.observe_generation(
				name="router_llm",
				model=obs_model,
				messages=messages,
				model_parameters={"temperature": 0.0, "max_tokens": 1024, "response_format": "json_object"},
			)
			if self.obs
			else None
		)

		if not self._client and not getattr(self.s, "GEMINI_API_KEY", ""):
			decision = self._dev_decide(messages[-1]["content"])
			if obs_ctx:
				with obs_ctx as gen_obs:
					if gen_obs:
						gen_obs.update(output=decision)
			return decision

		try:
			has_json_mention = any("json" in (m.get("content") or "").lower() for m in messages)
			call_messages = list(messages)
			if not has_json_mention:
				call_messages.insert(0, {"role": "system", "content": "Respond strictly with a JSON object."})

			content = None
			response = None

			# If Gemini is primary or Groq is unavailable/cooldown: use Gemini first
			if primary_provider == "gemini" or not self._is_groq_available():
				gemini_txt = self._call_gemini(call_messages, json_mode=True)
				if gemini_txt:
					content = gemini_txt
				elif self._is_groq_available():
					try:
						response = self._client.chat.completions.create(
							model=self.s.GROQ_ROUTER_MODEL,
							messages=call_messages,
							temperature=0.0,
							max_tokens=128,
							response_format={"type": "json_object"},
						)
						txt = (response.choices[0].message.content or "").strip()
						if txt:
							content = txt
					except Exception as g_exc:
						if "429" in str(g_exc) or "rate_limit" in str(g_exc).lower():
							self._trigger_groq_cooldown(seconds=90.0, reason=str(g_exc))
						log.warning("Groq router fallback failed (%s)", g_exc)
			else:
				# Groq is primary
				try:
					response = self._client.chat.completions.create(
						model=self.s.GROQ_ROUTER_MODEL,
						messages=call_messages,
						temperature=0.0,
						max_tokens=128,
						response_format={"type": "json_object"},
					)
					txt = (response.choices[0].message.content or "").strip()
					if txt:
						content = txt
				except Exception as exc:
					if "429" in str(exc) or "rate_limit" in str(exc).lower():
						self._trigger_groq_cooldown(seconds=90.0, reason=str(exc))
					log.warning("Groq router model '%s' failed (%s), calling Gemini 2.5 Flash", self.s.GROQ_ROUTER_MODEL, exc)
					gemini_txt = self._call_gemini(call_messages, json_mode=True)
					if gemini_txt:
						content = gemini_txt

			if not content:
				decision = self._dev_decide(messages[-1]["content"])
			else:
				decision = _extract_json_object(content)
				if not decision:
					log.warning("JSON extraction failed on router content: '%s' — using fallback", content[:80])
					decision = self._dev_decide(messages[-1]["content"])

			if obs_ctx:
				with obs_ctx as gen_obs:
					if gen_obs:
						usage_details = {}
						if response and getattr(response, "usage", None):
							usage_details = {
								"input": getattr(response.usage, "prompt_tokens", 0),
								"output": getattr(response.usage, "completion_tokens", 0),
								"total": getattr(response.usage, "total_tokens", 0),
							}
						gen_obs.update(output=decision, usage_details=usage_details)
			return decision
		except Exception as exc:
			log.warning("router call encountered issue (%s) — using dev decide fallback", exc)
			decision = self._dev_decide(messages[-1]["content"])
			if obs_ctx:
				with obs_ctx as gen_obs:
					if gen_obs:
						gen_obs.update(output=decision, status_message=str(exc), level="WARNING")
			return decision

	def respond(self, messages) -> str:
		primary_provider = getattr(self.s, "PRIMARY_LLM_PROVIDER", "gemini").lower()
		obs_model = (
			self.s.GEMINI_MODEL if primary_provider == "gemini"
			else (self.s.GROQ_RESPONDER_MODEL if self._client else "deterministic-dev-responder")
		)
		obs_ctx = (
			self.obs.observe_generation(
				name="responder_llm",
				model=obs_model,
				messages=messages,
				model_parameters={"temperature": 0.1, "max_tokens": 1024},
			)
			if self.obs
			else None
		)

		if not self._client and not getattr(self.s, "GEMINI_API_KEY", ""):
			reply = self._dev_respond(messages)
			if obs_ctx:
				with obs_ctx as gen_obs:
					if gen_obs:
						gen_obs.update(output=reply)
			return reply

		try:
			content = None
			response = None

			# If Gemini is primary or Groq is unavailable/cooldown: use Gemini first
			if primary_provider == "gemini" or not self._is_groq_available():
				gemini_txt = self._call_gemini(messages, json_mode=False)
				if gemini_txt:
					content = gemini_txt
				elif self._is_groq_available():
					try:
						response = self._client.chat.completions.create(
							model=self.s.GROQ_RESPONDER_MODEL,
							messages=messages,
							temperature=0.1,
							max_tokens=160,
						)
						txt = (response.choices[0].message.content or "").strip()
						if txt:
							content = txt
					except Exception as g_exc:
						if "429" in str(g_exc) or "rate_limit" in str(g_exc).lower():
							self._trigger_groq_cooldown(seconds=90.0, reason=str(g_exc))
						log.warning("Groq responder fallback failed (%s)", g_exc)
			else:
				# Groq is primary
				try:
					response = self._client.chat.completions.create(
						model=self.s.GROQ_RESPONDER_MODEL,
						messages=messages,
						temperature=0.1,
						max_tokens=160,
					)
					txt = (response.choices[0].message.content or "").strip()
					if txt:
						content = txt
				except Exception as exc:
					if "429" in str(exc) or "rate_limit" in str(exc).lower():
						self._trigger_groq_cooldown(seconds=90.0, reason=str(exc))
					log.warning("Groq responder model '%s' failed (%s), calling Gemini 2.5 Flash", self.s.GROQ_RESPONDER_MODEL, exc)
					gemini_txt = self._call_gemini(messages, json_mode=False)
					if gemini_txt:
						content = gemini_txt

			if not content:
				log.warning("All responder models returned empty or failed — falling back to deterministic response")
				reply = self._dev_respond(messages)
			else:
				reply = _strip_think_tags(content)

			if obs_ctx:
				with obs_ctx as gen_obs:
					if gen_obs:
						usage_details = {}
						if response and getattr(response, "usage", None):
							usage_details = {
								"input": getattr(response.usage, "prompt_tokens", 0),
								"output": getattr(response.usage, "completion_tokens", 0),
								"total": getattr(response.usage, "total_tokens", 0),
							}
						gen_obs.update(output=reply, usage_details=usage_details)
			return reply
		except Exception as exc:
			log.exception("responder call failed — attempting Gemini fallback")
			gemini_txt = self._call_gemini(messages, json_mode=False)
			if gemini_txt:
				return _strip_think_tags(gemini_txt)
			reply = self._dev_respond(messages)
			if obs_ctx:
				with obs_ctx as gen_obs:
					if gen_obs:
						gen_obs.update(output=reply, status_message=str(exc), level="WARNING")
			return reply

	async def adecide(self, messages) -> dict:
		"""Non-blocking asynchronous router call, executing in a worker thread pool."""
		return await asyncio.to_thread(self.decide, messages)

	async def arespond(self, messages) -> str:
		"""Non-blocking asynchronous responder call, executing in a worker thread pool."""
		return await asyncio.to_thread(self.respond, messages)

	def _classify_intent_sync(self, text: str) -> Optional[str]:
		"""Synchronous regex classification for latency filler intent with full Devanagari/Hinglish vocabulary."""
		clean = (text or "").strip().lower()
		if not clean:
			return None

		payment_pat = re.compile(
			r"\b(pay|payment|paid|paise|fees?|fee|transfer|upi|qr|gpay|google\s*pay|phonepe|paytm|done)\b|"
			r"(?<![\u0900-\u097F])(?:पेमेंट|पैसे|(?:गूगल|फोन)\s*पे|पेटीएम|यूपीआई|क्यूआर|ट्रांसफर|रुपये|रुपए|रुपया|"
			r"भर\s*दिए|कर\s*दिए|हो\s*गई|कर\s*दी|हो\s*चुकी|पे\s*(?:कर|किया|हो))(?![\u0900-\u097F])",
			re.IGNORECASE,
		)
		slots_pat = re.compile(
			r"\b(slot|slots|available|availability|free|time|timing|timings|date|day|days|"
			r"kal|parso|aaj|today|tomorrow|"
			r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
			r"subah|dopahar|shaam|raat|morning|afternoon|evening|night|"
			r"baje|pm|am|clock|hour|hours|"
			r"book|booking|reserve|reservation|appointment|schedule)\b|"
			r"(?<![\u0900-\u097F])(?:स्लॉट|स्लॉट्स|अपॉइंटमेंट|तारीख|समय|टाइम|टाइमिंग|उपलब्ध|उपलब्धता|खाली|"
			r"कल|परसों|आज|"
			r"सोमवार|मंगलवार|बुधवार|गुरुवार|शुक्रवार|शनिवार|रविवार|"
			r"मंडे|ट्यूसडे|वेडनेसडे|थर्सडे|फ्राइडे|सैटरडे|संडे|"
			r"सुबह|दोपहर|शाम|रात|बजे|पीएम|एएम|घंटे|"
			r"बुक|बुकिंग|रिजर्व|स्लॉट\s*बुक|अपॉइंटमेंट\s*बुक|मिल\s*सकता)(?![\u0900-\u097F])",
			re.IGNORECASE,
		)
		info_pat = re.compile(
			r"\b(doctor|dr|ananya|rohit|verma|sharma|specialist|"
			r"clinic|dental|dentist|treatment|treatments|service|services|checkup|consultation|consult|consulting|"
			r"rct|root\s*canal|braces|aligner|aligners|teeth|tooth|toothache|tooth\s*ache|toothpain|tooth-pain|tooth\s*pain|"
			r"teeth\s*whitening|whitening|issue|issues|problem|problems|fix|fixing|"
			r"daant|daanto|cavity|bleed|bleeding|pain|dard|swelling|infection|implant|implants|filling|"
			r"charges?|cost|price|fee|fees|address|location|timing|phase\s*7|mohali)\b|"
			r"(?<![\u0900-\u097F])(?:डॉक्टर|डॉ|रोहित|अनन्या|शर्मा|वर्मा|स्पेशलिस्ट|विशेषज्ञ|"
			r"डेंटल|दांत|दांतों|दाँत|दर्द|पेन|टूथपेन|टूथ-पेन|टूथ\s*पेन|टूथएक|टूथ|खून|ब्लीडिंग|इलाज|उपचार|परामर्श|कंसल्ट|कंसल्टेशन|"
			r"रूट\s*कैनाल|आरसीटी|ब्रेसेस|वाइटनिंग|टीथ\s*वाइटनिंग|सफाई|क्लीनिंग|चेकअप|"
			r"दिखाना|दिखाना\s*था|दिखाना\s*है|चेक\s*कराना|इशू|इशूज|इश्यूज|प्रॉब्लम|प्रॉब्लम्स|फिक्स|क्लिनिक|"
			r"कैविटी|कीड़ा|कीड़े|मसूड़े|मसूड़ों|सूजन|इम्प्लांट|फिलिंग|अकल\s*दाढ़|दाढ़|"
			r"तकलीफ|दिक्कत|समस्या|परेशानी|हेल्प|मदद|"
			r"पता|लोकेशन|खर्च|खर्चा|चार्ज|फीस|फी|रेट)(?![\u0900-\u097F])",
			re.IGNORECASE,
		)
		chitchat_pat = re.compile(
			r"\b(namaste|namaskar|hello|hi|hey|good\s+(?:morning|afternoon|evening)|ram\s*ram|pranam|"
			r"aap\s+kaisi\s+hain|aap\s+kaise\s+ho|kaise\s+ho|kaisa\s+hai|kaisi\s+ho|kaise\s+hain|"
			r"main\s+theek\s+hoon|theek\s+hoon|badhiya|sab\s+theek|sab\s+badhiya|"
			r"haan\s*ji|haan|ji\s*haan|theek\s*hai|theek|achha|accha|shukriya|dhanyavaad|thank\s*you|thanks|ok|okay|bye|alvida)\b|"
			r"(?<![\u0900-\u097F])(?:नमस्ते|नमस्कार|हेलो|हाय|आप कैसी हैं|आप कैसे हैं|आप कैसे हो|आप कैसी हो|क्या हाल है|कैसे हो|कैसी हो|सब ठीक|बढ़िया|"
			r"मैं ठीक हूँ|ठीक हूँ|हाँ जी|हाँ|जी हाँ|ठीक है|ठीक|अच्छा|शुक्रिया|धन्यवाद|अलविदा|बाय)(?![\u0900-\u097F])",
			re.IGNORECASE,
		)

		has_payment = bool(payment_pat.search(clean))
		has_slots = bool(slots_pat.search(clean))
		has_info = bool(info_pat.search(clean))
		has_chitchat = bool(chitchat_pat.search(clean))

		# Priority 1: Payment verification
		if has_payment:
			return "payment"

		# Priority 2: Slot booking / calendar lookup
		if has_slots:
			return "slots"

		# Priority 3: Non-dental medical complaints (back pain, stomach, headache, etc.)
		# Bright Dental Clinic specializes in teeth/oral care. Niaa responds immediately without latency filler.
		non_dental_pat = re.compile(
			r"\b(back\s*pain|stomach|stomach\s*pain|headache|head-ache|sar\s*dard|sar\s*mein\s*dard|bukhar|knee|leg|shoulder|chest|cough|cold|fever)\b|"
			r"(?<![\u0900-\u097F])(?:कमर|कमरदर्द|पेट|पेटदर्द|सिर|सिरदर्द|सिर\s*दर्द|पैर|हाथ|कंधे|बुखार|खांसी|जुकाम)(?![\u0900-\u097F])",
			re.IGNORECASE,
		)
		dental_override_pat = re.compile(
			r"\b(teeth|tooth|toothpain|tooth-pain|toothache|dental|dentist|daant|daanto|gums?|masude|jaw|jabda|jabde|rct|braces|mouth|cavity)\b|"
			r"(?<![\u0900-\u097F])(?:दांत|दांतों|दाँत|डेंटल|टूथपेन|टूथ\s*पेन|टूथ|मसूड़े|मसूड़ों|जबड़ा|जबड़े|मुंह|कैविटी|ब्रेसेस|रूट\s*कैनाल|अकल\s*दाढ़|दाढ़)(?![\u0900-\u097F])",
			re.IGNORECASE,
		)
		if non_dental_pat.search(clean) and not dental_override_pat.search(clean):
			return None

		# Priority 4: Clinical treatments, doctor queries, pricing
		if has_info:
			# If user included greeting/chit-chat, ensure it is truly a clinic question and not just pleasantry
			# e.g., "Hello Nia, braces lagwane hain" -> "info"
			# e.g., "Hello Nia, kaisi ho tum? help kar sakti ho?" -> None (immediate response, no delay filler)
			if has_chitchat and not bool(re.search(
				r"\b(doctor|dr|ananya|rohit|dental|dentist|teeth|tooth|toothpain|toothache|braces|aligner|rct|root\s*canal|whitening|cavity|implant|cleaning|daant|dard|pain|cost|price|fees?|issue|issues|problem|problems|fix)\b|"
				r"(डॉक्टर|रोहित|अनन्या|डेंटल|दांत|दाँत|दर्द|पेन|टूथपेन|टूथ|ब्रेसेस|वाइटनिंग|सफाई|कैविटी|रूट\s*कैनाल|इम्प्लांट|खर्च|फीस|तकलीफ|दिक्कत|समस्या|दिखाना|चेकअप|इशू|इशूज|इश्यूज|प्रॉब्लम|फिक्स|क्लिनिक)",
				clean,
				re.IGNORECASE,
			)):
				return None
			return "info"

		# Priority 5: Pure chit-chat or acknowledgment -> NO latency filler, answer immediately!
		if has_chitchat:
			return None

		# If user is angry/frustrated/reprimanding, never classify as 'short' thinking filler
		if is_angry_or_frustrated(clean):
			return None

		short_pat = re.compile(
			r"\b(ek\s+(?:sawaal|sawal|doubt|query|baat\s+poochni)|sawaal\s+(?:tha|hai)|sawal\s+(?:tha|hai))\b|"
			r"(?<![\u0900-\u097F])(?:एक\s*(?:सवाल|डाउट|क्वेरी)|सवाल\s*(?:था|है)|पूछना\s*था)(?![\u0900-\u097F])",
			re.IGNORECASE,
		)
		if short_pat.search(clean):
			return "short"

		return None

	async def classify_filler_intent(self, text: str) -> Optional[str]:
		"""High-speed classifier for latency masking filler intent using Groq qwen/qwen3.8-27b.
		Strictly matches defined intents ('payment', 'slots', 'info', 'short').
		If intent is not matched (chit-chat, greetings, non-dental or general queries), returns None so main agent responds directly.
		"""
		clean = (text or "").strip()
		if not clean:
			return None

		# 1. Fast heuristic sync check (0ms)
		sync_intent = self._classify_intent_sync(clean)
		if sync_intent in ("payment", "slots", "info", "short"):
			return sync_intent

		# 2. If sync check returned None (ambiguous or nuanced query), invoke Groq qwen/qwen3.8-27b
		if self._is_groq_available():
			model = getattr(self.s, "GROQ_CLASSIFIER_MODEL", "qwen/qwen3.8-27b")
			prompt = (
				"Classify caller intent for an AI dental receptionist Niaa at Bright Dental Clinic.\n"
				f"Caller utterance: \"{clean}\"\n\n"
				"Return JSON ONLY: {\"intent\": \"payment\" | \"slots\" | \"info\" | \"short\" | null}\n\n"
				"Rules:\n"
				"- 'slots': Checking appointment availability, timings, days, booking slots, calendar availability\n"
				"- 'payment': Confirming UPI, Google Pay, fees payment, transfer, fee status\n"
				"- 'info': Dental treatments (toothache, root canal, cavity, cleaning, braces), doctor schedule/names, clinic charges, address\n"
				"- 'short': Explicitly asking to ask a question/doubt ('ek sawaal tha', 'ek query hai')\n"
				"- null: Chit-chat, greetings, pleasantries ('namaste', 'kaise ho', 'theek hai', 'shukriya'), general talk, non-dental issues, or complaints.\n"
				"If not definitely one of the four defined intents, return null."
			)

			def _call_groq_filler():
				resp = self._client.chat.completions.create(
					model=model,
					messages=[
						{"role": "system", "content": "You are a real-time intent classifier outputting JSON only."},
						{"role": "user", "content": prompt},
					],
					temperature=0.0,
					max_tokens=25,
					response_format={"type": "json_object"},
				)
				return resp.choices[0].message.content

			try:
				raw = await asyncio.wait_for(asyncio.to_thread(_call_groq_filler), timeout=0.38)
				parsed = json.loads(raw)
				groq_intent = parsed.get("intent")
				if groq_intent in ("payment", "slots", "info", "short"):
					return groq_intent
				return None
			except Exception as e:
				if "429" in str(e) or "rate_limit" in str(e).lower():
					self._trigger_groq_cooldown(seconds=90.0, reason=str(e))
				log.debug("Groq intent classifier fallback notice: %s", e)

		return sync_intent

	def _fast_turn_decide(
		self,
		incoming_text: str,
		conversation_history: list[dict] = None,
		is_agent_speaking: bool = False,
		pending_continuation: str = "",
		provider: str = "rumik",
		has_pending_booking: bool = False,
		has_confirmed_booking: bool = False,
	) -> dict:
		"""Instant 0ms fallback heuristics for turn action decision with guardrails."""
		clean = (incoming_text or "").strip()
		lowered = clean.lower()
		end_stripped = re.sub(r"[।.,!?;:\s]+$", "", lowered).strip()

		# 1. Empty or pure whitespace
		if not clean or not end_stripped:
			return {
				"decision": "DROP_NOISE",
				"sentiment": "neutral",
				"intent": None,
				"backchannel": None,
				"deescalation_response": None,
				"clean_query": "",
			}

		# 2. Prompt Injection & Jailbreak Defense (0ms Instant Intercept)
		if is_prompt_injection(clean):
			clinic_name = getattr(self.s, "CLINIC_NAME", "Bright Dental Clinic")
			refusal = (
				f"Ji main {clinic_name} ki AI receptionist Niaa hoon ji. "
				f"Aap dental appointments, treatments ya clinic services ke baare mein pooch sakte hain, batayein kaise help karoon?"
				if provider in ("sarvam", "rumik")
				else f"I am the receptionist at {clinic_name}. I can assist you with dental appointments, treatments, and clinic information. How can I help you today?"
			)
			return {
				"decision": "SECURITY_INTERCEPT",
				"sentiment": "neutral",
				"intent": None,
				"backchannel": None,
				"deescalation_response": refusal,
				"clean_query": clean,
			}

		# 3. Stray noise / isolated acoustic artifact
		words = [w for w in re.split(r"\s+", end_stripped) if w]
		is_stray = end_stripped in STRAY_NOISE_OR_FILLER or clean in STRAY_NOISE_OR_FILLER
		if not is_stray and len(words) == 1 and len(words[0]) <= 3 and words[0] not in {"yes", "no", "kal", "aaj", "doc", "dr"}:
			is_stray = True

		if is_stray:
			if is_agent_speaking or pending_continuation:
				return {
					"decision": "DROP_NOISE",
					"sentiment": "neutral",
					"intent": None,
					"backchannel": None,
					"deescalation_response": None,
					"clean_query": "",
				}
			return {
				"decision": "WAIT_AND_LISTEN",
				"sentiment": "neutral",
				"intent": None,
				"backchannel": "Ji...",
				"deescalation_response": None,
				"clean_query": "",
			}

		# 4. Actionable Clinic / Booking / Medical Enquiry check
		is_angry = is_angry_or_frustrated(clean)
		# The main agent engine (run_turn) should ALWAYS be triggered if there is an actionable clinic/business/medical inquiry!
		has_actionable = any(k in lowered for k in ACTIONABLE_TERMS)
		full_candidate = f"{pending_continuation} {clean}".strip() if pending_continuation else clean

		# Pure emotional complaint/anger with no scheduling query -> return instant de-escalating apology
		if is_angry and not has_actionable:
			clinic_name = getattr(self.s, "CLINIC_NAME", "Bright Dental Clinic")
			apology = get_deescalation_response(clinic_name=clinic_name, provider=provider, user_text=clean)
			return {
				"decision": "FAST_RESPONSE",
				"sentiment": "angry",
				"intent": None,
				"backchannel": None,
				"deescalation_response": apology,
				"direct_response": apology,
				"clean_query": full_candidate,
			}

		# 5. Non-dental medical requests (fever, headache, back pain, stomach ache, etc.)
		non_dental_pat = re.compile(
			r"\b(back\s*pain|stomach|stomach\s*pain|headache|head-ache|sar\s*dard|sar\s*mein\s*dard|bukhar|fever|knee|leg|shoulder|chest|cough|cold)\b|"
			r"(?<![\u0900-\u097F])(?:कमर|कमरदर्द|पेट|पेटदर्द|सिर|सिरदर्द|सिर\s*दर्द|पैर|हाथ|कंधे|बुखार|खांसी|जुकाम)(?![\u0900-\u097F])",
			re.IGNORECASE,
		)
		dental_override_pat = re.compile(
			r"\b(teeth|tooth|toothpain|tooth-pain|toothache|dental|dentist|daant|daanto|gums?|masude|jaw|jabda|jabde|rct|braces|mouth|cavity)\b|"
			r"(?<![\u0900-\u097F])(?:दांत|दांतों|दाँत|डेंटल|टूथपेन|टूथ\s*पेन|टूथ|मसूड़े|मसूड़ों|जबड़ा|जबड़े|मुंह|कैविटी|ब्रेसेस|रूट\s*कैनाल|अकल\s*दाढ़|दाढ़)(?![\u0900-\u097F])",
			re.IGNORECASE,
		)
		if non_dental_pat.search(clean) and not dental_override_pat.search(clean):
			clinic_name = getattr(self.s, "CLINIC_NAME", "Bright Dental Clinic")
			resp = get_fast_chit_chat_response("non_dental", clean, clinic_name=clinic_name, provider=provider)
			return {
				"decision": "FAST_RESPONSE",
				"sentiment": "neutral",
				"intent": None,
				"fast_category": "non_dental",
				"backchannel": None,
				"deescalation_response": resp,
				"direct_response": resp,
				"clean_query": full_candidate,
			}

		# Compliments (Voice, appearance, pleasantness)
		compliment_pat = re.compile(
			r"\b(aawaz\s*(?:bahut\s*)?(?:achhi|sweet|pyari|nice)|voice\s*is\s*(?:very\s*)?(?:nice|sweet|good|beautiful)|"
			r"you\s*are\s*(?:very\s*)?(?:beautiful|sweet|smart|kind|cute|great|awesome)|"
			r"bahut\s*(?:achhi|pyari|smart|sundar)\s*ho|aap\s*achhi\s*ho)\b|"
			r"(?:आवाज\s*(?:बहुत\s*)?(?:अच्छी|प्यारी|स्वीट)|आप\s*(?:बहुत\s*)?(?:सुंदर|अच्छी|प्यारी)\s*हो|"
			r"यू\s*आर\s*(?:वेरी\s*)?ब्यूटीफुल|ब्यूटीफुल\s*हो)",
			re.IGNORECASE,
		)

		# Farewell / Ending the conversation, call cutting, or finishing
		farewell_pat = re.compile(
			r"\b(cut\s+(?:the\s+)?call|call\s*cut|cut\s*kardo|cut\s*kar\s*do|cut\s*kijiye|disconnect\s+(?:the\s+)?call|disconnect|hang\s*up|phone\s*kaat|phone\s*rakh|phone\s*cut|"
			r"i\s*am\s*done|im\s*done|done\s*with\s*this|all\s*done|we\s*are\s*done|that\s*is\s*all|that'?s\s*all|that'?s\s*it|that\s*will\s*be\s*all|bas\s*itna\s*hi|kuch\s*nahi\s*chahiye|ho\s*gaya|"
			r"no\s*more\s*questions?|no\s*other\s*questions?|no\s*further\s*questions?|nothing\s*else|nothing\s*more|koi\s*aur\s*sawaal\s*nahi|koi\s*sawaal\s*nahi|aur\s*koi\s*sawaal\s*nahi|koi\s*doubt\s*nahi|koi\s*question\s*nahi|"
			r"thanks\s+for\s+(?:the\s+)?help|thank\s+you\s+for\s+(?:your\s+|the\s+)?help|help\s*ke\s*liye\s*(?:shukriya|dhanyawad)|madad\s*ke\s*liye\s*(?:shukriya|dhanyawad)|"
			r"baad\s*mein|bad\s*mein|later|call\s*you\s*later|baad\s*mein\s*baat|alvida|bye\b(?!\s*the\s*way)|goodbye|chalta\s*hoon|chalti\s*hoon)\b|"
			r"(?<![\u0900-\u097F])(?:कट\s*द\s*कॉल|कॉल\s*कट|कॉल\s*काट|काट\s*दो|काट\s*दीजिए|कट\s*कर\s*दो|कट\s*कर\s*दीजिए|फोन\s*काट|फोन\s*रख|डिस्कनेक्ट|"
			r"आई\s*एम\s*डन|डन\s*विथ\s*दिस|सब\s*हो\s*गया|बस\s*इतना\s*ही|और\s*कुछ\s*नहीं|हो\s*गया\s*काम|"
			r"कोई\s*और\s*सवाल\s*नहीं|कोई\s*सवाल\s*नहीं|और\s*कोई\s*सवाल\s*नहीं|कुछ\s*और\s*नहीं\s*पूछना|"
			r"हेल्प\s*के\s*लिए\s*(?:शुक्रिया|धन्यवाद|थैंक\s*यू)|मदद\s*के\s*लिए\s*(?:शुक्रिया|धन्यवाद)|"
			r"बाद\s*में\s*बात|बाद\s*में\s*कॉल|अलविदा|टाटा|बाय\s*बाय|बाय(?!\s*द\s*वे)|चलता\s*हूँ|चलती\s*हूँ)(?![\u0900-\u097F])",
			re.IGNORECASE,
		)

		# Latency / Speed / Listening remarks
		latency_bot_pat = re.compile(
			r"\b(slow|itni\s*slow|itna\s*slow|deri|late|sun\s*rahi\s*ho|sunte\s*ho|awaz\s*aa\s*rahi)\b|"
			r"(?<![\u0900-\u097F])(?:स्लो|इतनी स्लो|इतना स्लो|देरी|देर|सुनते हो|सुन रही हो|आवाज आ रही)(?![\u0900-\u097F])",
			re.IGNORECASE,
		)

		# Pure Greetings (namaste, hello, etc.)
		greeting_pat = re.compile(
			r"^(namaste|namaskar|hello|hi|hey|good\s+(?:morning|afternoon|evening)|नमस्ते|नमस्कार|हेलो|हाय)\s*([!.,।?]*)$|"
			r"^(namaste|namaskar|hello|hi|hey|good\s+(?:morning|afternoon|evening))\s+(niaa?|nia)\s*([!.,।?]*)$|"
			r"^(नमस्ते|नमस्कार|हेलो|हाय)\s+(निया)\s*([!.,।?]*)$",
			re.IGNORECASE,
		)

		# Pure Chit-chat & Well-being (including inverted phrases like 'कैसी हैं आप', 'kaise ho', etc.)
		chitchat_pat = re.compile(
			r"\b(aap\s+kaisi\s+hain|kaisi\s+hain\s+aap|aap\s+kaise\s+ho|kaise\s+ho\s+aap|kaise\s+ho|kaisa\s+hai|kaisi\s+ho|kaise\s+hain|"
			r"kaisi\s+ho\s+tum|kaise\s+ho\s+tum|main\s+theek\s+hoon|theek\s+hoon|kya\s+haal\s+hai|kya\s+chal\s+raha\s+hai|aur\s+batao)\b|"
			r"(?<![\u0900-\u097F])(?:आप कैसी हैं|कैसी हैं आप|कैसी हो आप|कैसा है|आप कैसे हैं|कैसे हैं आप|आप कैसे हो|कैसे हो आप|आप कैसी हो|क्या हाल है|कैसे हो|कैसी हो|कैसी हो तुम|कैसे हो तुम|सब ठीक|बढ़िया|मैं ठीक हूँ|ठीक हूँ|और बताओ|क्या चल रहा है)(?![\u0900-\u097F])",
			re.IGNORECASE,
		)

		# Identity check
		identity_pat = re.compile(
			r"\b(aap\s+kaun\s+ho|who\s+are\s+you|kaun\s+bol\s+raha\s+hai|naam\s+kya\s+hai|what\s+is\s+your\s+name)\b|"
			r"(?<![\u0900-\u097F])(?:आप कौन हो|कौन बोल रहा है|नाम क्या है|तुम्हारा नाम)(?![\u0900-\u097F])",
			re.IGNORECASE,
		)

		# Pure Acknowledgment (theek hai, haan ji, shukriya, thank you)
		ack_pat = re.compile(
			r"^(haan\s*ji|haan|theek\s*hai|theek|achha|accha|shukriya|dhanyavaad|dhanyawad|thank\s*you|thanks|ok|okay)\s*([!.,।?]*)$|"
			r"^(हाँ जी|हाँ|जी हाँ|ठीक है|ठीक|अच्छा|शुक्रिया|धन्यवाद|थैंक यू|थैंक्स|ओके)\s*([!.,।?]*)$|"
			r"\b(thank\s*you|thanks|shukriya|dhanyawad)\b|"
			r"(?<![\u0900-\u097F])(?:शुक्रिया|धन्यवाद|थैंक\s*यू|थैंक्स)(?![\u0900-\u097F])",
			re.IGNORECASE,
		)

		# 6. Check for incomplete thought / hesitation starter / trailing connector
		is_incomplete = False
		if len(words) <= 4:
			for hp in HESITATION_PHRASES:
				if hp in end_stripped:
					is_incomplete = True
					break
		has_complete_ending = (
			clean.endswith("?")
			or clean.endswith("।")
			or clean.endswith(".")
			or bool(
				re.search(
					r"\b(kyun|kyu|kaise|क्यों|कैसे|स्लो|slow|chuka\s*hoon|kar\s*diya|kar\s*chuka|ho\s*gaya|upi\s*se|card\s*se|cash\s*se|pay\s*kiya)\b",
					clean,
					re.IGNORECASE,
				)
			)
		)
		if not is_incomplete and words and words[-1] in INCOMPLETE_TRAILING_WORDS and not has_complete_ending:
			is_incomplete = True
		if not is_incomplete:
			for tm in TRAILING_INCOMPLETE_PHRASES:
				if end_stripped.endswith(tm):
					is_incomplete = True
					break

		if is_incomplete:
			return {
				"decision": "WAIT_AND_LISTEN",
				"sentiment": "angry" if is_angry else "neutral",
				"intent": None,
				"backchannel": "Ji...",
				"deescalation_response": None,
				"clean_query": full_candidate,
			}

		# 7. Fast Chit-chat / Greeting / Compliment / Acknowledgment / Farewell Intercept
		# Crucial Rule: ONLY intercept if it strictly matches a fast category AND has NO actionable keywords!
		fast_cat = None
		if compliment_pat.search(clean):
			fast_cat = "compliment"
		elif farewell_pat.search(clean):
			fast_cat = "farewell"
		elif (has_pending_booking or has_confirmed_booking) and not has_actionable and (
			any(w in lowered for w in ("thank", "thanks", "shukriya", "dhanyawad", "शुक्रिया", "धन्यवाद", "थैंक", "थैंक्स"))
			or ack_pat.search(clean)
		):
			# In active booking state, gratitude or done acknowledgments indicate the caller is wrapping up!
			fast_cat = "farewell"
		elif latency_bot_pat.search(clean):
			fast_cat = "latency_check"
		elif not has_actionable:
			if identity_pat.search(clean):
				fast_cat = "identity"
			elif greeting_pat.search(clean):
				fast_cat = "greeting"
			elif ack_pat.search(clean):
				fast_cat = "acknowledgment"
			elif chitchat_pat.search(clean):
				fast_cat = "chitchat"

		if fast_cat:
			clinic_name = getattr(self.s, "CLINIC_NAME", "Bright Dental Clinic")
			resp = get_fast_chit_chat_response(
				fast_cat,
				clean,
				clinic_name=clinic_name,
				provider=provider,
				has_pending_booking=has_pending_booking,
				has_confirmed_booking=has_confirmed_booking,
			)
			is_call_end = (fast_cat == "farewell")
			return {
				"decision": "CALL_END" if is_call_end else "FAST_RESPONSE",
				"is_farewell": is_call_end,
				"fast_category": fast_cat,
				"sentiment": "neutral",
				"intent": None,
				"backchannel": None,
				"deescalation_response": resp,
				"direct_response": resp,
				"clean_query": full_candidate,
			}

		# 8. Complete actionable query or general conversation -> EXECUTE_TURN with classified intent
		intent = self._classify_intent_sync(full_candidate)
		return {
			"decision": "EXECUTE_TURN",
			"sentiment": "angry" if is_angry else "neutral",
			"intent": intent,
			"backchannel": None,
			"deescalation_response": None,
			"clean_query": full_candidate,
		}

	async def decide_turn_action(
		self,
		incoming_text: str,
		conversation_history: list[dict] = None,
		is_agent_speaking: bool = False,
		pending_continuation: str = "",
		provider: str = "rumik",
		has_pending_booking: bool = False,
		has_confirmed_booking: bool = False,
	) -> dict:
		"""Unified first-layer classifier with ultra-fast 0ms local decision path.
		Executes instant local heuristics first (<0.2ms). Only if completely ambiguous,
		falls back to high-speed LLM classifier (Gemini or Groq) with a strict 450ms timeout.
		"""
		clean = (incoming_text or "").strip()
		if not clean:
			return {
				"decision": "DROP_NOISE",
				"sentiment": "neutral",
				"intent": None,
				"backchannel": None,
				"deescalation_response": None,
				"clean_query": "",
			}

		clinic_name = getattr(self.s, "CLINIC_NAME", "Bright Dental Clinic")

		# 0ms Hard Guardrail: Prompt Injection check
		if is_prompt_injection(clean):
			log.warning("First-layer intercepted prompt injection: %s", clean)
			refusal = (
				f"Ji main {clinic_name} ki AI receptionist Niaa hoon ji. "
				f"Aap dental appointments, treatments ya clinic services ke baare mein pooch sakte hain, batayein kaise help karoon?"
				if provider in ("sarvam", "rumik")
				else f"I am the receptionist at {clinic_name}. I can assist you with dental appointments, treatments, and clinic information. How can I help you today?"
			)
			return {
				"decision": "SECURITY_INTERCEPT",
				"sentiment": "neutral",
				"intent": None,
				"backchannel": None,
				"deescalation_response": refusal,
				"clean_query": clean,
			}

		# 0ms micro pre-filter: obvious 1-word stray noise when agent is actively speaking
		lowered = clean.lower()
		end_stripped = re.sub(r"[।.,!?;:\s]+$", "", lowered).strip()
		if is_agent_speaking and (end_stripped in STRAY_NOISE_OR_FILLER or (len(end_stripped) <= 3 and end_stripped not in {"yes", "no", "kal", "aaj"})):
			return {
				"decision": "DROP_NOISE",
				"sentiment": "neutral",
				"intent": None,
				"backchannel": None,
				"deescalation_response": None,
				"clean_query": "",
			}

		# 0ms Fast Local Turn Decision (<0.2ms)
		fast_res = self._fast_turn_decide(
			incoming_text=incoming_text,
			conversation_history=conversation_history,
			is_agent_speaking=is_agent_speaking,
			pending_continuation=pending_continuation,
			provider=provider,
			has_pending_booking=has_pending_booking,
			has_confirmed_booking=has_confirmed_booking,
		)

		# If fast rules gave a definitive decision (turn pause, noise, injection, anger complaint, fast chit-chat/greetings, or recognized intent):
		# Return immediately in 0ms! Do not wait on any LLM!
		if fast_res.get("decision") in ("WAIT_AND_LISTEN", "DROP_NOISE", "SECURITY_INTERCEPT", "FAST_RESPONSE", "CALL_END") or fast_res.get("direct_response"):
			return fast_res

		if fast_res.get("intent") in ("payment", "slots", "info", "short"):
			return fast_res

		if fast_res.get("sentiment") == "angry":
			return fast_res

		# If it's standard polite chit-chat / greeting, fast rules correctly set intent=None & direct_response
		chitchat_check = re.compile(
			r"\b(namaste|namaskar|hello|hi|hey|good\s+(?:morning|afternoon|evening)|"
			r"aap\s+kaisi\s+hain|aap\s+kaise\s+ho|kaise\s+ho|kaisa\s+hai|kaisi\s+ho|kaise\s+hain|"
			r"main\s+theek\s+hoon|theek\s+hoon|badhiya|sab\s+theek|"
			r"haan\s*ji|haan|theek\s*hai|theek|achha|accha|shukriya|dhanyavaad|thank\s*you|thanks|ok|okay|bye|alvida)\b|"
			r"(?<![\u0900-\u097F])(?:नमस्ते|नमस्कार|हेलो|हाय|आप कैसी हैं|आप कैसे हैं|आप कैसे हो|आप कैसी हो|क्या हाल है|कैसे हो|कैसी हो|सब ठीक|बढ़िया|"
			r"मैं ठीक हूँ|ठीक हूँ|हाँ जी|हाँ|जी हाँ|ठीक है|ठीक|अच्छा|शुक्रिया|धन्यवाद|अलविदा|बाय)(?![\u0900-\u097F])",
			re.IGNORECASE,
		)
		if not any(k in clean.lower() for k in ACTIONABLE_TERMS) and chitchat_check.search(clean):
			return fast_res

		# Format last 3 dialogue turns for context awareness in the rare ambiguous case
		hist_lines = []
		if conversation_history:
			for m in conversation_history[-3:]:
				role = "Caller" if m.get("role") == "user" else "Receptionist"
				c = (m.get("content") or "").strip()
				if c:
					hist_lines.append(f"{role}: {c}")
		recent_dialogue = "\n".join(hist_lines) if hist_lines else "None (start of call)"

		booking_status_str = "Pending payment for held slot" if has_pending_booking else ("Confirmed appointment" if has_confirmed_booking else "None")

		prompt = (
			f"Active Booking Status: {booking_status_str}\n"
			f"Recent Dialogue:\n{recent_dialogue}\n\n"
			f"Previous Buffered Fragment: \"{pending_continuation or ''}\"\n"
			f"Incoming Utterance: \"{clean}\"\n"
			f"Receptionist Speaking Right Now: {'Yes' if is_agent_speaking else 'No'}\n\n"
			"Return ONLY a JSON object with this schema:\n"
			"{\n"
			'  "decision": "WAIT_AND_LISTEN" | "EXECUTE_TURN" | "DROP_NOISE" | "SECURITY_INTERCEPT" | "FAST_RESPONSE" | "CALL_END",\n'
			'  "sentiment": "neutral" | "angry",\n'
			'  "intent": "payment" | "slots" | "info" | "short" | null,\n'
			'  "fast_category": "greeting" | "chitchat" | "acknowledgment" | "compliment" | "farewell" | "non_dental" | "complaint" | null,\n'
			'  "backchannel": "Ji..." | "Haan ji..." | null,\n'
			'  "deescalation_response": "<apologetic help if angry complaint, or refusal if injection, else null>",\n'
			'  "clean_query": "<merged complete question or current text>"\n'
			"}\n\n"
			"Guidelines:\n"
			"- CALL_END / fast_category='farewell': Caller wants to conclude, cut, or end the call ('cut the call', 'disconnect', 'done with this', 'no more questions', 'bye', 'alvida', 'thanks for the help'). If caller already reserved or confirmed an appointment and expresses gratitude or says they are done, ALWAYS choose CALL_END / farewell.\n"
			"- FAST_RESPONSE: For pure greetings (namaste, hello), compliments ('aapki aawaz achhi hai'), chit-chat (kaise ho), acknowledgments (theek hai), non-dental medical requests (fever/headache), or complaints. Set intent=null and fast_category.\n"
			"- EXECUTE_TURN:\n"
			"  * intent='slots': Doctor availability, dates, times, days, slot booking, rescheduling, cancelling.\n"
			"  * intent='info': Dental treatments, tooth pain, cavity, bleeding, cleaning, doctor specialties, charges, fees, clinic location, policies.\n"
			"  * intent='payment': Payment completed, UPI, QR code, fees paid, transfer, GPay, PhonePe.\n"
			"  * intent='short': Caller explicitly states they have a doubt ('ek sawaal tha').\n"
			"- WAIT_AND_LISTEN: Caller started talking but paused mid-thought.\n"
			"- DROP_NOISE: Acoustic cough, breath, mic click, or isolated stray sound."
		)

		primary_provider = getattr(self.s, "PRIMARY_LLM_PROVIDER", "gemini").lower()

		try:
			raw_json = ""
			classifier_model = getattr(self.s, "GEMINI_CLASSIFIER_MODEL", "gemini-2.5-flash")
			# High-Speed Classifier Path: Groq qwen/qwen3.8-27b delivers ~150-250ms latency
			if self._is_groq_available():
				model = getattr(self.s, "GROQ_CLASSIFIER_MODEL", "qwen/qwen3.8-27b")
				def _call_groq():
					resp = self._client.chat.completions.create(
						model=model,
						messages=[
							{
								"role": "system",
								"content": "You are the first-layer real-time turn classifier for Niaa, an AI dental receptionist at Bright Dental Clinic. You output strictly valid JSON.",
							},
							{"role": "user", "content": prompt},
						],
						temperature=0.0,
						max_tokens=180,
						response_format={"type": "json_object"},
					)
					return resp.choices[0].message.content

				try:
					raw_json = await asyncio.wait_for(asyncio.to_thread(_call_groq), timeout=0.45)
				except Exception as groq_err:
					if "429" in str(groq_err) or "rate_limit" in str(groq_err).lower():
						self._trigger_groq_cooldown(seconds=90.0, reason=str(groq_err))
					# Fallback to Gemini 2.5 Flash immediately
					gemini_messages = [
						{
							"role": "system",
							"content": "You are the first-layer real-time turn classifier for Niaa, an AI dental receptionist at Bright Dental Clinic. You output strictly valid JSON.",
						},
						{"role": "user", "content": prompt},
					]
					raw_json = await asyncio.wait_for(
						asyncio.to_thread(
							self._call_gemini,
							gemini_messages,
							json_mode=True,
							model=classifier_model,
							max_tokens=180,
						),
						timeout=1.8,
					)
			else:
				gemini_messages = [
					{
						"role": "system",
						"content": "You are the first-layer real-time turn classifier for Niaa, an AI dental receptionist at Bright Dental Clinic. You output strictly valid JSON.",
					},
					{"role": "user", "content": prompt},
				]
				raw_json = await asyncio.wait_for(
					asyncio.to_thread(
						self._call_gemini,
						gemini_messages,
						json_mode=True,
						model=classifier_model,
						max_tokens=180,
					),
					timeout=1.8,
				)

			if not raw_json:
				return fast_res

			data = _extract_json_object(raw_json) or json.loads(_strip_think_tags(raw_json))
			dec = data.get("decision", "EXECUTE_TURN")
			if dec not in ("WAIT_AND_LISTEN", "EXECUTE_TURN", "DROP_NOISE", "SECURITY_INTERCEPT", "FAST_RESPONSE", "CALL_END"):
				dec = "EXECUTE_TURN"

			merged = data.get("clean_query")
			if not merged or not merged.strip():
				merged = f"{pending_continuation} {clean}".strip() if pending_continuation else clean

			raw_intent = data.get("intent")
			if raw_intent in ("none", "null", "chitchat"):
				raw_intent = None

			sentiment = data.get("sentiment", "neutral")
			if sentiment not in ("neutral", "angry"):
				sentiment = "neutral"

			if is_angry_or_frustrated(clean):
				sentiment = "angry"

			fast_cat = data.get("fast_category")
			if not fast_cat and raw_intent in ("greeting", "chitchat", "acknowledgment", "compliment", "farewell", "non_dental", "complaint"):
				fast_cat = raw_intent

			if (dec in ("CALL_END", "FAST_RESPONSE") or fast_cat) and (
				fast_cat in ("farewell", "compliment", "latency_check", "non_dental")
				or dec == "CALL_END"
				or not any(k in merged.lower() for k in ACTIONABLE_TERMS)
			):
				is_farewell = (dec == "CALL_END" or fast_cat == "farewell")
				dec = "CALL_END" if is_farewell else "FAST_RESPONSE"
				fast_cat = "farewell" if is_farewell else (fast_cat or "chitchat")
				raw_intent = None
				direct_resp = get_fast_chit_chat_response(
					fast_cat,
					clean,
					clinic_name=clinic_name,
					provider=provider,
					has_pending_booking=has_pending_booking,
					has_confirmed_booking=has_confirmed_booking,
				)
				return {
					"decision": dec,
					"is_farewell": is_farewell,
					"fast_category": fast_cat,
					"sentiment": sentiment,
					"intent": None,
					"backchannel": None,
					"deescalation_response": direct_resp,
					"direct_response": direct_resp,
					"clean_query": merged,
				}

			deescalation_response = data.get("deescalation_response")
			if dec == "SECURITY_INTERCEPT" and not deescalation_response:
				deescalation_response = (
					f"Ji main {clinic_name} ki AI receptionist Niaa hoon ji. "
					f"Aap dental appointments, treatments ya clinic services ke baare mein pooch sakte hain, batayein kaise help karoon?"
				)
			elif sentiment == "angry" and not deescalation_response and raw_intent == "complaint":
				deescalation_response = get_deescalation_response(clinic_name=clinic_name, provider=provider, user_text=clean)

			return {
				"decision": dec,
				"sentiment": sentiment,
				"intent": raw_intent,
				"backchannel": data.get("backchannel"),
				"deescalation_response": deescalation_response,
				"clean_query": merged,
			}
		except Exception as exc:
			log.warning("LLM first-layer classifier fallback to fast rules: %s", exc)
			return fast_res

	def transcribe_audio(self, audio_bytes: bytes, filename: str = "audio.webm") -> str:
		"""Transcribe spoken voice audio using Groq's high-speed Whisper model."""
		if not self._client or not audio_bytes:
			return ""
		obs_ctx = (
			self.obs.observe_generation(
				name="whisper_transcription",
				model="whisper-large-v3-turbo",
				messages=[{"role": "user", "content": f"audio_file: {filename} ({len(audio_bytes)} bytes)"}],
				model_parameters={"temperature": 0.0, "language": "en"},
			)
			if self.obs
			else None
		)
		try:
			mime = "audio/webm" if filename.endswith(".webm") else "audio/wav"
			file_payload = (filename, audio_bytes, mime)
			transcription = self._client.audio.transcriptions.create(
				model="whisper-large-v3-turbo",
				file=file_payload,
				language="en",
				temperature=0.0,
			)
			text = (getattr(transcription, "text", "") or "").strip()
			if obs_ctx:
				with obs_ctx as gen_obs:
					if gen_obs:
						gen_obs.update(output=text)
			return text
		except Exception as exc:
			log.exception("whisper audio transcription failed")
			if obs_ctx:
				with obs_ctx as gen_obs:
					if gen_obs:
						gen_obs.update(output="", status_message=str(exc), level="WARNING")
			return ""

	def _dev_decide(self, text: str) -> dict:
		lowered = text.lower().strip()
		if lowered == "paid":
			return {"action": "payment.verify", "args": {}}
		if lowered == "payment_cancelled" or "cancel" in lowered:
			return {"action": "booking.cancel", "args": {}}
		if any(word in lowered for word in ("bye", "goodbye", "that's all")):
			return {"action": "end_call", "args": {}}

		time_match = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(a\.?m\.?|p\.?m\.?)\b", lowered)
		date_string = self._dev_date(lowered)

		# Doctor extraction
		doctor = ""
		if "rohit" in lowered or any(k in lowered for k in ("brace", "aligner", "crooked", "gap")):
			doctor = "Dr. Rohit Verma"
		elif "ananya" in lowered or any(k in lowered for k in ("root canal", "rct", "toothache", "pain", "cleaning", "cavity", "whitening")):
			doctor = "Dr. Ananya Sharma"

		if time_match and any(word in lowered for word in (
			"book", "appointment", "slot", "confirm", "take", "final",
			"yes", "right", "fine", "ok", "okay", "good", "perfect", "please", "works", "fix"
		)):
			hour = int(time_match.group(1)) % 12
			meridiem = time_match.group(3).replace(".", "") if time_match.group(3) else None
			if meridiem == "pm":
				hour += 12
			minute = time_match.group(2) or "00"
			return {
				"action": "calendar.select_slot",
				"args": {"date": date_string, "time": f"{hour:02d}:{minute}", "doctor": doctor},
			}
		if any(word in lowered for word in (
			"book", "appointment", "slot", "slots", "available", "free", "schedule", "open",
			"can i see", "can i visit", "come in"
		)):
			return {
				"action": "calendar.check_availability",
				"args": {"date": date_string, "doctor": doctor},
			}
		if any(word in lowered for word in ("hello", "hi", "namaste", "hey")) and len(lowered) < 25:
			return {"action": "chitchat", "args": {}}
		return {"action": "rag.query", "args": {"question": text}}

	def _dev_date(self, text: str) -> str:
		today = date.today()
		lowered = text.lower()
		if re.search(r"\bday\s+after\s+(?:today|aaj)\b", lowered):
			return (today + timedelta(days=1)).isoformat()
		if re.search(r"\b(?:day\s+after\s+tomorrow|day\s+after\s+next|parso|parson)\b", lowered):
			return (today + timedelta(days=2)).isoformat()
		if "tomorrow" in lowered or "kal" in lowered:
			return (today + timedelta(days=1)).isoformat()
		if "today" in lowered or "aaj" in lowered:
			return today.isoformat()

		days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
		for idx, d_name in enumerate(days):
			if d_name in lowered:
				today_idx = today.weekday()
				days_ahead = (idx - today_idx) % 7
				if days_ahead == 0 and not any(w in lowered for w in ("this", "today")):
					days_ahead = 7
				if "next week" in lowered:
					days_ahead += 7
				return (today + timedelta(days=days_ahead)).isoformat()

		match = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", text)
		return match.group(1) if match else (today + timedelta(days=1)).isoformat()

	def _dev_respond(self, messages) -> str:
		brief = messages[-1]["content"] if messages and messages[-1]["role"] == "system" else ""
		if "OUTCOME: payment_required" in brief:
			doc = ""
			try:
				data = json.loads(brief.split("TOOL RESULT (facts you may use):", 1)[1])
				doc = data.get("doctor", "")
			except Exception:
				pass
			doc_phrase = f" with {doc}" if doc else ""
			return (f"Your slot{doc_phrase} is held pending payment. Please click the 'Open payment' "
					f"button on your screen to complete the {self.s.BOOKING_FEE_RUPEES} rupee booking fee.")
		if '"status": "unpaid"' in brief:
			return ("Your payment is still pending. Please click the 'Open payment' button "
					"on your screen to complete the booking fee so we can confirm your appointment.")
		if "OUTCOME: booking_confirmed" in brief:
			try:
				data = json.loads(brief.split("TOOL RESULT (facts you may use):", 1)[1])
				booking = data.get("booking") or {}
				date_str = booking.get("date", "your requested date")
				time_str = booking.get("time", "your requested time")
				return f"Payment confirmed! Your appointment is successfully booked for {date_str} at {time_str}."
			except Exception:
				return "Payment confirmed! Your appointment is successfully booked."
		if '"free_slots"' in brief:
			try:
				data = json.loads(brief.split("TOOL RESULT (facts you may use):", 1)[1])
				slots = ", ".join(data.get("free_slots", [])[:4]) or "no slots"
				return f"On {data.get('date')}, we have: {slots}. Which time works for you?"
			except Exception:
				pass
		if '"context"' in brief:
			try:
				data = json.loads(brief.split("TOOL RESULT (facts you may use):", 1)[1])
				context = (data.get("context") or "").split("\n---\n")[0]
				return context[:400] if context else "Let me have the clinic confirm that for you."
			except Exception:
				pass
		if "OUTCOME: call_end" in brief:
			return "Thank you for calling. Have a great day!"
		return "Happy to help — would you like to book an appointment, or do you have a question?"
