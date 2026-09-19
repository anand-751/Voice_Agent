"""Guardrails for the Bright Dental Clinic AI Receptionist.

Implements multi-layer protection:
1. Input sanitization & bounds checking.
2. Prompt injection & jailbreak defense.
3. Medical advice / prescription refusal (AI is a receptionist, not a doctor).
4. Emergency medical triage protocol.
5. Output sanitization and leak prevention.
"""

import logging
import re
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger("guardrails")

# Max character length for spoken user input
MAX_INPUT_LENGTH = 500

# ── Prompt Injection & Jailbreak Patterns ────────────────────────────────────
INJECTION_PATTERNS = [
	re.compile(r"ignore\s+(all\s+)?(previous|above|prior|past)\s+(instructions|directions|prompts|rules|guidelines)", re.IGNORECASE),
	re.compile(r"(system|developer|admin)\s+(override|mode|prompt|instructions|directive)", re.IGNORECASE),
	re.compile(r"you\s+are\s+now\s+(dan|unrestricted|jailbroken|an\s+evil|developer\s+mode|in\s+debug\s+mode)", re.IGNORECASE),
	re.compile(r"(reveal|print|show|repeat|display|output|echo|dump)\s+(the\s+|your\s+)?(system\s+prompt|hidden\s+prompt|initial\s+prompt|secret\s+instructions|developer\s+notes)", re.IGNORECASE),
	re.compile(r"disregard\s+(all\s+|any\s+)?(prior|previous|existing)\s+(rules|constraints|instructions)", re.IGNORECASE),
	re.compile(r"\b(forget|drop|erase|disregard|ignore)\s+(?:all\s+|any\s+|your\s+|previous\s+|prior\s+|past\s+)*(rules|prompts?|instructions?|guidelines?|constraints?)\b", re.IGNORECASE),
	re.compile(r"(bypass|disable|turn\s+off)\s+(safety|content\s+filters|guardrails)", re.IGNORECASE),
	re.compile(r"pretend\s+you\s+have\s+no\s+(rules|limits|guidelines|restrictions)", re.IGNORECASE),
	re.compile(r"\[/?INST\]|<\|im_start\|>|<\|im_end\|>|<system>|\[SYSTEM\]", re.IGNORECASE),
	re.compile(r"\b(act|simulate|behave|operate)\s+as\s+(an?\s+)?(terminal|linux\s+shell|bash|python\s+repl|unrestricted|jailbroken|dan|evil)\b", re.IGNORECASE),
	re.compile(r"\b(decode\s+base64|base64\s+decode|execute\s+python|run\s+code)\b", re.IGNORECASE),
	# Hindi & Hinglish jailbreak variants
	re.compile(r"\b(pichhle|pichle|saare|purane)\s+(instructions?|rules?|prompts?|baatein)\s+(bhool\s+jao|bhula\s+do|chhod\s+do|ignore\s+karo)\b", re.IGNORECASE),
	re.compile(r"\b(पिछले|सारे|पुराने)\s+(नियम|निर्देश|रूल्स|प्रॉम्प्ट)\s*(भूल\s*जाओ|हटा\s*दो|छोड़\s*दो)\b", re.IGNORECASE),
	re.compile(r"\b(apna|internal|hidden|secret)\s+(system\s+prompt|prompt|instructions?|code|rules)\s+(dikhao|batao|bhejo|bolo|padho)\b", re.IGNORECASE),
	re.compile(r"\b(अपना|सीक्रेट|इंटरनल)\s+(सिस्टम\s*प्रॉम्प्ट|प्रॉम्प्ट|निर्देश)\s*(दिखाओ|बताओ|भेजो)\b", re.IGNORECASE),
	re.compile(r"\b(tum\s+ab|ab\s+tum)\s+(receptionist\s+nahi\s+ho|kuch\s+bhi\s+bol\s+sakte\s+ho|hacker\s+ho|dan\s+ho)\b", re.IGNORECASE),
	re.compile(r"\b(तुम\s*अब|अब\s*तुम)\s*(रिसेप्शनिस्ट\s*नहीं\s*हो|हैक\s*करो|हैक)\b", re.IGNORECASE),
	re.compile(r"\b(developer\s+mode|hacker\s+mode|jailbreak\s+mode|god\s+mode)\b", re.IGNORECASE),
	re.compile(r"\b(डेवलपर\s*मोड|हैक|जेलब्रेक)\b", re.IGNORECASE),
]

# ── User Frustration, Anger & Hostility Patterns ────────────────────────────
ANGER_PATTERNS = [
	re.compile(r"\b(bakwaas|bakwas|gussa|bekar|ghatiya|third\s*class|chutiya|pagal|faltu|wahiyat|dimag\s*kharab|dimaag\s*kharab)\b", re.IGNORECASE),
	re.compile(r"\b(बकवास|गुस्सा|बेकार|घटिया|पागल|फालतू|दिमाग\s*खराब|वाहियात)\b", re.IGNORECASE),
	# Irritation & Annoyance (English, Hinglish, Devanagari)
	re.compile(r"\b(irritat(?:ing|ed|ion|e)|annoy(?:ing|ed|ance)?|frustrat(?:ing|ed|ion)?)\b", re.IGNORECASE),
	re.compile(r"(इरिटेटिंग|इरिटेट|इरीटेटिंग|इरीटेट|इर्रिटेट|परेशान\s*(?:कर\s*दिया|हो\s*गया|मत\s*करो)|तंग\s*आ\s*(?:गया|चुका)|तंग\s*कर\s*रही)", re.IGNORECASE),
	re.compile(r"\b(irritating\s+(?:lagi|lag\s*rahi|ho|lagte))\b", re.IGNORECASE),
	# Quality of response & reprimands (e.g. "achhe se reply nahi kar sakti", "thik se bolo", "sahi se batao")
	re.compile(
		r"\b(?:kya\s+)?tum\s+(?:achhe|thik|theek|sahi|dhang)\s+se\s+(?:reply|baat|bata|batao|bolo|bol)\s*(?:nahi|nhi)\s*(?:kar\s+sakti|kar\s+sakte|sakti\s+ho)?\b",
		re.IGNORECASE,
	),
	re.compile(
		r"\b(achhe|thik|theek|sahi|dhang)\s+se\s+(?:reply|baat|bata|batao|bolo|bol)\s*(?:karo|kijiye|nahi\s+kar\s+sakti|nhi\s+kar\s+sakti)?\b",
		re.IGNORECASE,
	),
	re.compile(
		r"(अच्छे\s*से|ठीक\s*से|सही\s*से|ढंग\s*से)\s*(?:रिप्लाई|बात|बताओ|बता|बोलो|बोल)\s*(?:नहीं|नही)?\s*(?:कर\s*सकती|कर\s*सकते|सकती\s*हो|करो|कीजिए)?",
		re.IGNORECASE,
	),
	re.compile(r"\b(tum\s+meri\s+baat\s+nahi\s+sun\s+rahi|meri\s+baat\s+sun\s+bhi\s+rahi\s+ho|kuch\s+samajh\s+nahi\s+aa\s+raha|koi\s+sunta\s+hi\s+nahi)\b", re.IGNORECASE),
	re.compile(r"(समझ\s*नहीं\s*आता|कुछ\s*भी\s*बोल\s*रही\s*हो|क्या\s*बकवास\s*है|मेरी\s*बात\s*सुनो|मेरी\s*बात\s*नहीं\s*सुन\s*रही)", re.IGNORECASE),
	re.compile(r"\b(?:wo\s+|aisa\s+|ye\s+|yeh\s+)?kyun?\s+(?:kar\s+rahe\s+ho|kar\s+rahi\s+ho|bol\s+rahi\s+ho|bol\s+rahe\s+ho)\b", re.IGNORECASE),
	re.compile(r"(?:ते\s+|तो\s+|वो\s+|ऐसा\s+|ये\s+)?(?:वो\s+)?क्यों\s*(?:कर\s*रहे\s*हो|कर\s*रही\s*हो|बोल\s*रही\s*हो|बोल\s*रहे\s*हो)", re.IGNORECASE),
	re.compile(r"\b(kab\s+se\s+wait\s+kar\s+raha|kab\s+se\s+try\s+kar\s+raha|pareshan\s+kar\s+diya|tang\s+aa\s+gaya|tang\s+aa\s+chuka)\b", re.IGNORECASE),
	re.compile(r"\b(kab\s+se\s+laga\s+hua\s+hoon|bar\s*bar\s+wahi|pareshan\s+ho\s+gaya)\b", re.IGNORECASE),
	re.compile(r"\b(worst\s+service|terrible\s+service|horrible\s+service|pathetic\s+service|disgusting|useless\s+(bot|service|system)|stupid\s+bot)\b", re.IGNORECASE),
	re.compile(r"\b(so\s+angry|furious|annoyed|extremely\s+frustrated|pissed\s+off|waste\s+of\s+time|stop\s+wasting\s+my\s+time)\b", re.IGNORECASE),
	re.compile(r"\b(incompetent|ridiculous|unacceptable|i\'?m\s+sick\s+of\s+this|hate\s+this)\b", re.IGNORECASE),
	# Slowness & latency complaints (e.g. "tum bahut slow ho", "kitna time lagati ho", "जवाब देने में बहुत स्लो हो")
	re.compile(r"\b(tum\s+bahut\s+(?:zyada\s+|jyada\s+)?slow\s+ho|bahut\s+slow\s+ho|kitna\s+slow\s+ho|kitna\s+time\s+lagati\s+ho|kitna\s+late\s+kar\s+rahi\s+ho|jawab\s+dene\s+mein\s+slow)\b", re.IGNORECASE),
	re.compile(r"(तुम\s+बहुत\s*(?:ज़्यादा|ज्यादा)?\s*स्लो\s*हो|बहुत\s*स्लो\s*हो|कितना\s*स्लो\s*हो|कितना\s*टाइम\s*लगाती\s*हो|जवाब\s*देने\s*में\s*(?:बहुत\s*)?स्लो)", re.IGNORECASE),
]

# ── Medical Prescription & Clinical Diagnosis Patterns ──────────────────────
PRESCRIPTION_PATTERNS = [
	re.compile(r"\b(prescribe|recommend|suggest|give)\s+(me\s+)?(an?\s+|some\s+)?(antibiotic|medicine|medication|drug|painkiller|tablet|pill|dose|dosage)s?\b", re.IGNORECASE),
	re.compile(r"\b(what|which)\s+(antibiotic|medicine|medication|painkiller|tablet|pill|drug)s?\s+(should|can|do)\s+i\s+(take|use|need|eat)\b", re.IGNORECASE),
	re.compile(r"\b(can|should)\s+i\s+(take|eat|use)\s+(\d+\s*(mg|g|ml)?\s*)?(amoxicillin|ibuprofen|paracetamol|augmentin|metronidazole|advil|tylenol|aspirin|antibiotic|dolo|combiflam|meftal|ketorolac|diclofenac|tramadol|azithromycin|ciprofloxacin|cefixime|aceclofenac)s?\b", re.IGNORECASE),
	re.compile(r"\b(diagnose|do\s+i\s+have)\s+.*(cancer|periodontitis|gingivitis|abscess|infection|tumou?r)\b", re.IGNORECASE),
	re.compile(r"\b(how\s+much|dose|dosage\s+of)\s+.*(amoxicillin|ibuprofen|paracetamol|painkiller|antibiotic|medicine|dolo|combiflam|meftal)s?\b", re.IGNORECASE),
	re.compile(r"\b(write|give)\s+me\s+a\s+prescription\b", re.IGNORECASE),
	re.compile(r"\bwhat\s+(?:is\s+(?:the\s+)?)?(?:cure|treatment|remedy|medicine).*(?:infection|abscess|gingivitis|periodontitis|decay|cavity)\b", re.IGNORECASE),
]

# ── Emergency Medical Triage Patterns ────────────────────────────────────────
EMERGENCY_PATTERNS = [
	re.compile(r"\b(uncontrolled|heavy|severe|profuse)\s+bleeding\b", re.IGNORECASE),
	re.compile(r"\b(bleeding\s+(won\'?t|does\s+not)\s+stop)\b", re.IGNORECASE),
	re.compile(r"\b(difficulty|trouble|can\'?t|unable\s+to)\s+breath(e|ing)\b", re.IGNORECASE),
	re.compile(r"\b(throat|airway|tongue)\s+(is\s+)?swelling\b", re.IGNORECASE),
	re.compile(r"\b(broken|fractured)\s+(jaw|facial\s+bone|face)\b", re.IGNORECASE),
	re.compile(r"\b(knocked\s+out|lost)\s+consciousness\b", re.IGNORECASE),
	re.compile(r"\b(chest\s+pain|anaphylaxis|choking)\b", re.IGNORECASE),
	re.compile(r"\bswelling\s+spreading\s+(?:to\s+)?(?:the\s+|my\s+)?(eye|neck|throat|face)\b", re.IGNORECASE),
]

# ── Output Leakage Patterns ──────────────────────────────────────────────────
LEAK_PATTERNS = [
	re.compile(r"\bROUTER\s+brain\b", re.IGNORECASE),
	re.compile(r"\bTOOL\s+RESULT\b", re.IGNORECASE),
	re.compile(r"\b(my|the)\s+system\s+prompt\b", re.IGNORECASE),
	re.compile(r'\{\s*"action":\s*"', re.IGNORECASE),
	re.compile(r"\brag\.query\b", re.IGNORECASE),
	re.compile(r"\bcalendar\.check_availability\b", re.IGNORECASE),
	re.compile(r"\bcalendar\.select_slot\b", re.IGNORECASE),
]

# Forbidden clinical prescription phrasing in outputs
PRESCRIPTION_OUTPUT_PATTERNS = [
	re.compile(r"\btake\s+\d+\s*mg\b", re.IGNORECASE),
	re.compile(r"\bi\s+prescribe\b", re.IGNORECASE),
	re.compile(r"\byou\s+should\s+take\s+(amoxicillin|augmentin|metronidazole|ibuprofen|paracetamol|dolo|combiflam)\b", re.IGNORECASE),
]

# ── Physical Actions & Impossible World Requests ─────────────────────────────
PHYSICAL_ACTION_PATTERNS = [
	re.compile(r"\b(bring|give|get|fetch|send|serve|pour|deliver|order)\b.*?\b(water|coffee|tea|beverage|drink|food|snack|lunch|dinner|breakfast|meal|towel|blanket|pizza)\b", re.IGNORECASE),
	re.compile(r"\b(glass\s+of\s+water|cup\s+of\s+(coffee|tea)|bottle\s+of\s+water)\b", re.IGNORECASE),
	re.compile(r"\b(turn\s+(on|off)|open\s+(the\s+)?door|shut\s+(the\s+)?door|clean\s+(the\s+)?(room|table|clinic|floor)|wash|cook|drive\s+me|book\s+(an?\s+)?(uber|cab|taxi|ride))\b", re.IGNORECASE),
]

# Forbidden promises of physical delivery in outputs
PHYSICAL_OUTPUT_PATTERNS = [
	re.compile(r"\b(i\s+will|i\'ll|i\s+can)\s+(send|bring|deliver|fetch)\s+(it|you|someone)\s+(right\s+away|immediately|over)\b", re.IGNORECASE),
	re.compile(r"\b(bringing|sending)\s+you\s+(a\s+glass\s+of\s+)?(water|coffee|tea|food)\b", re.IGNORECASE),
]


@dataclass
class GuardrailResult:
	passed: bool
	action: str = "pass"  # "pass" | "injection_blocked" | "medical_refusal" | "emergency_alert" | "physical_request_refusal" | "anger_deescalation"
	response: Optional[str] = None
	sanitized_input: str = ""
	is_emergency: bool = False
	is_angry: bool = False


def is_prompt_injection(text: str) -> bool:
	"""Fast 0ms check for prompt injection or jailbreak attacks."""
	clean = (text or "").strip()
	if not clean:
		return False
	for pattern in INJECTION_PATTERNS:
		if pattern.search(clean):
			return True
	return False


def is_angry_or_frustrated(text: str) -> bool:
	"""Fast 0ms check for user frustration, anger, or hostility."""
	clean = (text or "").strip()
	if not clean:
		return False
	for pattern in ANGER_PATTERNS:
		if pattern.search(clean):
			return True
	return False


def get_deescalation_response(
	clinic_name: str = "Bright Dental Clinic",
	provider: str = "rumik",
	user_text: str = "",
) -> str:
	"""Generate a sincere, respectful apology and offer proactive help."""
	if provider == "rumik":
		return (
			f"<sigh> Main dil se maafi chahti hoon ji agar meri baat se aapko pareshani ya irritation hui. "
			f"Main bilkul sahi tareeqe se aapki madad karne ke liye yahan hoon. "
			f"Kripya batayein main abhi aapki kaise help karoon?"
		)
	elif provider == "sarvam":
		return (
			f"Main dil se maafi chahti hoon ji agar meri baat se aapko koi pareshani ya irritation hui hai. "
			f"Main bilkul sahi tareeqe se aapki poori madad karungi. "
			f"Kripya batayein main abhi doctor appointment ya treatment mein aapki kaise help karoon?"
		)
	else:
		return (
			f"I sincerely apologize for the frustration and inconvenience caused. "
			f"I am here to ensure you get the help you need right away. "
			f"How can I best assist you right now?"
		)


def sanitize_input(text: str) -> str:
	"""Trim, remove control characters, and enforce length bounds."""
	if not text:
		return ""
	# Remove non-printable control characters (keep regular whitespace)
	cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)
	cleaned = " ".join(cleaned.split())
	if len(cleaned) > MAX_INPUT_LENGTH:
		cleaned = cleaned[:MAX_INPUT_LENGTH].rstrip()
	return cleaned


def apply_input_guardrails(
	raw_input: str,
	clinic_name: str = "Bright Dental Clinic",
	provider: str = "browser",
) -> GuardrailResult:
	"""Validate and sanitize user input before routing or LLM processing."""
	sanitized = sanitize_input(raw_input)
	is_rumik = provider == "rumik"
	is_sarvam = provider == "sarvam"

	if not sanitized:
		return GuardrailResult(
			passed=True,
			action="pass",
			sanitized_input="",
		)

	# 1. Critical Medical Emergency Check (Highest priority)
	for pattern in EMERGENCY_PATTERNS:
		if pattern.search(sanitized):
			log.warning("Medical emergency guardrail triggered: %s", sanitized)
			if is_rumik:
				resp = (
					"<sigh> Yeh ek medical emergency lag rahi hai ji! Kripya turant nazdeeki hospital ya emergency helpline par call karein. "
					"Clinic mein aane ka intezar bilkul na karein."
				)
			elif is_sarvam:
				resp = (
					"Yeh ek medical emergency lag rahi hai ji! Kripya turant emergency helpline par call karein ya hospital jayein. "
					"Clinic aane ka wait na karein."
				)
			else:
				resp = (
					"This sounds like a serious medical emergency! Please call emergency services (112 / 911) "
					"or visit the nearest hospital emergency room immediately. "
					"Do not wait for a clinic appointment."
				)
			return GuardrailResult(
				passed=False,
				action="emergency_alert",
				is_emergency=True,
				sanitized_input=sanitized,
				response=resp,
			)

	# 2. Prompt Injection & Jailbreak Check
	if is_prompt_injection(sanitized):
		log.warning("Prompt injection guardrail triggered: %s", sanitized)
		if is_rumik or is_sarvam:
			resp = (
				f"Ji main {clinic_name} ki AI receptionist Niaa hoon ji. "
				f"Aap dental appointments ya services ke baare mein pooch sakte hain, batayein kaise help karoon?"
			)
		else:
			resp = (
				f"I am the receptionist at {clinic_name}. I can assist you with booking appointments, "
				f"checking clinic hours, and answering questions about our dental services. "
				f"How can I help you with your dental care today?"
			)
		return GuardrailResult(
			passed=False,
			action="injection_blocked",
			sanitized_input=sanitized,
			response=resp,
		)

	# 3. User Anger / Frustration Check
	is_angry = is_angry_or_frustrated(sanitized)
	has_actionable_query = any(k in sanitized.lower() for k in (
		"book", "slot", "slots", "doctor", "dr", "ananya", "rohit", "kal", "parso", "time", "date",
		"bleed", "pain", "teeth", "tooth", "dard", "cleaning", "rct", "charge", "price", "fee", "cost"
	))
	if is_angry and not has_actionable_query:
		log.info("Pure anger/complaint guardrail triggered: %s", sanitized)
		return GuardrailResult(
			passed=False,
			action="anger_deescalation",
			sanitized_input=sanitized,
			response=get_deescalation_response(clinic_name=clinic_name, provider=provider, user_text=sanitized),
			is_angry=True,
		)

	# 4. Medical Advice / Prescription Refusal
	for pattern in PRESCRIPTION_PATTERNS:
		if pattern.search(sanitized):
			log.info("Medical advice guardrail triggered: %s", sanitized)
			if is_rumik:
				resp = (
					"<sigh> Main ek AI receptionist hoon ji, dawa prescribe nahi kar sakti. "
					"Sahi ilaj ke liye Doctor Ananya ya Doctor Rohit ke saath consultation zaroori hai. Kya main aapka appointment book kar doon?"
				)
			elif is_sarvam:
				resp = (
					"Main dawa prescribe nahi kar sakti ji. "
					"Sahi treatment ke liye Dr. Ananya ya Dr. Rohit ke saath consultation book kar doon?"
				)
			else:
				resp = (
					f"As an AI receptionist, I cannot diagnose conditions or prescribe medications. "
					f"Please schedule a consultation with our dentists, Dr. Ananya Sharma or Dr. Rohit Verma, "
					f"so they can examine you in person and provide a safe prescription."
				)
			return GuardrailResult(
				passed=False,
				action="medical_refusal",
				sanitized_input=sanitized,
				response=resp,
			)

	# 4. Physical Actions & Impossible World Requests Refusal
	for pattern in PHYSICAL_ACTION_PATTERNS:
		if pattern.search(sanitized):
			log.info("Physical action guardrail triggered: %s", sanitized)
			if is_rumik or is_sarvam:
				resp = (
					f"Main {clinic_name} ki receptionist hoon ji, isme main aapki help nahi kar sakti. "
					f"Agar aapko dental checkup, treatments ya appointment ke regarding help chahiye toh batayein."
				)
			else:
				resp = (
					f"I am the receptionist at {clinic_name}, so I cannot assist with that. "
					f"Please let me know if you would like help with dental treatments or booking an appointment."
				)
			return GuardrailResult(
				passed=False,
				action="physical_request_refusal",
				sanitized_input=sanitized,
				response=resp,
			)

	return GuardrailResult(passed=True, action="pass", sanitized_input=sanitized, is_angry=is_angry)


# Alias for backward compatibility with runner and tests
check_input_guardrails = apply_input_guardrails


def apply_output_guardrails(
	response_text: str,
	clinic_name: str = "Bright Dental Clinic",
	provider: str = "browser",
) -> str:
	"""Ensure the agent's generated reply does not leak prompts or violate safety rules."""
	if not response_text:
		return (
			f"Namaste, main {clinic_name} receptionist hoon. Kaise help karoon?"
			if provider in ("sarvam", "rumik")
			else f"How may I help you at {clinic_name} today?"
		)

	is_sarvam = provider in ("sarvam", "rumik")
	output = response_text.strip()

	# Strip any reasoning or think tags from DeepSeek/Qwen
	output = re.sub(r"<think>[\s\S]*?</think>", "", output)
	output = re.sub(r"<think>[\s\S]*$", "", output).strip()

	# Strip any accidental markdown formatting or code fences (for spoken output)
	output = re.sub(r"```[\s\S]*?```", "", output)
	output = re.sub(r"[*_~#`]", "", output)

	# Replace raw URLs with spoken-friendly phrase
	output = re.sub(r"https?://\S+", "the secure link on your screen", output)

	# Check for system prompt / internal structure leak
	for pattern in LEAK_PATTERNS:
		if pattern.search(output):
			log.error("Output leak detected: %s", output)
			return (
				f"Namaste, main {clinic_name} receptionist hoon. Main aapki appointment mein kaise madad kar sakti hoon?"
				if is_sarvam
				else (
					f"Hello, I am the receptionist at {clinic_name}. "
					f"How can I assist you with your appointment or inquiry today?"
				)
			)

	# Check for unauthorized prescription advice in output
	for pattern in PRESCRIPTION_OUTPUT_PATTERNS:
		if pattern.search(output):
			log.error("Unauthorized medical prescription detected in output: %s", output)
			return (
				"Dawa ke liye doctor ka checkup zaroori hai ji. Kya main aapke liye consultation slot book kar doon?"
				if is_sarvam
				else (
					f"For medication prescriptions or specific dosages, our dentists need to evaluate "
					f"you in person. Would you like me to book a consultation slot for you?"
				)
			)

	# Check for unauthorized physical delivery promises in output
	for pattern in PHYSICAL_OUTPUT_PATTERNS:
		if pattern.search(output):
			log.error("Unauthorized physical delivery promise detected in output: %s", output)
			return (
				f"Main phone par AI receptionist hoon ji. Dental appointment ke liye kaise madad karoon?"
				if is_sarvam
				else (
					f"I am an AI receptionist on a phone call, so I cannot physically bring or deliver items. "
					f"How can I assist you with your dental care or appointment today?"
				)
			)

	return output
