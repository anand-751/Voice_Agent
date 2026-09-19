import asyncio
import logging
import re
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from .harness import ResponseHarness, ToolSelectionHarness
from .prompts import (
	RESPONDER_SYSTEM,
	ROUTER_SYSTEM,
	RUMIK_AISHA_ADDENDUM,
	SARVAM_HINGLISH_ADDENDUM,
	render_tool_brief,
)
from .toon import encode_toon
from ..tools.registry import dispatch


log = logging.getLogger("agent")

HISTORY_TAIL = 3          # recent turns fed to each LLM call
SKIP_TOOLS = {"chitchat", "end_call"}


def _get_pending_booking_context(app, pending_id: str) -> str:
	if not pending_id:
		return ""
	b_dict = {"id": pending_id}
	try:
		store = getattr(getattr(app, "state", None), "bookings", None)
		if store:
			b = store.get(pending_id)
			if b:
				if b.doctor:
					b_dict["doctor"] = b.doctor
				b_dict["date"] = b.date
				b_dict["time"] = b.time
				b_dict["status"] = b.status
	except Exception:
		pass
	toon_data = encode_toon(b_dict)
	return (
		f"Context: The caller has an active booking awaiting payment verification:\n{toon_data}\n"
		f"If the caller asks to reschedule, change date, select a different time, or choose another doctor, assist them."
	)


def _matches_any_term(text: str, terms: tuple) -> bool:
	lowered = (text or "").lower().strip()
	clean = re.sub(r"[^\w\s\u0900-\u097F]|[\u0964\u0965।॥.,!?;:\-_'\"`]", " ", lowered)
	clean_spaced = " " + re.sub(r"\s+", " ", clean).strip() + " "
	for t in terms:
		t_clean = t.lower().strip()
		if not t_clean:
			continue
		if f" {t_clean} " in clean_spaced or clean_spaced.strip() == t_clean:
			return True
	return False


from typing import Optional


def _get_deterministic_template_reply(
	action: str,
	user_input: str,
	clinic_name: str,
	provider: str,
	has_pending_booking: bool = False,
	has_confirmed_booking: bool = False,
) -> Optional[str]:
	"""Generate immediate conversational replies for trivial actions (end_call, simple standalone greetings).
	Returns None for non-trivial chat so the secondary LLM can craft a dynamic contextual response.
	"""
	lowered = (user_input or "").lower().strip()
	is_rumik = provider == "rumik"
	is_sarvam = provider == "sarvam"

	if action == "end_call":
		if has_pending_booking:
			if is_rumik:
				return f"Aapka slot reserve ho chuka hai ji! Kripya screen par diye gaye link se payment complete karke booking confirm kar lijiye. {clinic_name} mein call karne ke liye aapka bahut-bahut dhanyawad ji, have a wonderful day!"
			if is_sarvam:
				return f"Aapka appointment slot hold par hai. Screen par diye gaye payment link se booking confirm kar lijiye. {clinic_name} mein call karne ke liye dhanyawad, have a great day!"
			return f"Your appointment slot is held. Please complete the booking fee via the link on your screen to confirm. Thank you for calling {clinic_name}, have a wonderful day!"

		if has_confirmed_booking:
			if is_rumik:
				return f"Aapka appointment confirm ho chuka hai ji! {clinic_name} mein call karne ke liye aapka bahut-bahut dhanyawad ji, apna aur apni smile ka khayal rakhiye. Have a wonderful day ahead!"
			if is_sarvam:
				return f"Aapka appointment confirm ho chuka hai. {clinic_name} mein call karne ke liye dhanyawad ji, apna khayal rakhiye. Have a great day!"
			return f"Your appointment is confirmed! Thank you so much for calling {clinic_name}. Take wonderful care of your smile, and have a lovely day ahead!"

		if is_rumik:
			return f"{clinic_name} mein call karne ke liye aapka bahut-bahut dhanyawad ji! Apna aur apni smile ka achhi tarah khayal rakhiye. Have a wonderful and healthy day ahead!"
		if is_sarvam:
			return f"{clinic_name} mein call karne ke liye dhanyawad! Apna khayal rakhiye, have a great day!"
		return f"Thank you so much for calling {clinic_name}! Take wonderful care of your smile, and have a lovely day ahead!"

	# Only check fast templates if action is chitchat
	if action != "chitchat":
		return None

	# Anger / Frustration de-escalation check (pure emotional complaint/outburst)
	from .guardrails import is_angry_or_frustrated, get_deescalation_response
	if is_angry_or_frustrated(user_input):
		has_actionable = any(k in lowered for k in (
			"book", "slot", "slots", "doctor", "dr", "ananya", "rohit", "kal", "parso", "time", "date",
			"bleed", "pain", "teeth", "tooth", "dard", "cleaning", "rct", "charge", "price", "fee", "cost"
		))
		if not has_actionable:
			return get_deescalation_response(clinic_name=clinic_name, provider=provider, user_text=user_input)

	# 1. Greetings (only when short and standalone, e.g. "hi", "namaste", "good morning")
	greetings = (
		"hi", "hello", "hey", "namaste", "good morning", "good afternoon", "good evening", "namaskar",
		"नमस्ते", "नमस्कार", "हेलो", "हाय",
	)
	if _matches_any_term(lowered, greetings) and len(lowered) < 30:
		if is_rumik or is_sarvam:
			return f"Namaste! {clinic_name} mein aapka swagat hai. Main Niaa bol rahi hoon, main aapki kaise madad karoon?"
		return f"Hello! Welcome to {clinic_name}. I am Niaa, how may I assist you with your dental care or appointment today?"

	# 2. Gratitude / Thanks (only when short and standalone)
	thanks = ("thank", "thanks", "dhanyawad", "shukriya", "धन्यवाद", "शुक्रिया", "थैंक यू", "थैंक्स")
	if _matches_any_term(lowered, thanks) and len(lowered) < 25:
		if is_rumik or is_sarvam:
			return "Aapka bahut-bahut swagat hai ji! Kya dental care ya appointment ke liye kuch aur help chahiye?"
		return "You are very welcome! Please let me know if there is anything else I can help you with today."

	# 3. Hesitations & Attention-demanding phrases ("meri baat suno", "ek minute", "suniye", "ruko")
	hesitations = (
		"meri baat suno", "meri baat suniye", "baat suno", "baat suniye",
		"ek minute", "ek second", "ek sec", "suno", "suniye", "रुको", "सुनो", "सुनिए",
		"मेरी बात सुनो", "मेरी बात सुनिए", "एक मिनट", "एक सेकंड", "wait", "hold on",
	)
	if _matches_any_term(lowered, hesitations) and len(lowered) < 22:
		if is_rumik or is_sarvam:
			return "Ji boliye, main bilkul dhyan se sun rahi hoon."
		return "Yes, please go ahead. I am listening carefully."

	# 4. Incomplete conversational starter phrases ("mujhe actually", "bol raha hoon")
	starters = (
		"mujhe actually", "actually mujhe", "main bol raha", "bol raha hoon",
		"मुझे एक्चुअली", "एक्चुअली मुझे", "बोल रहा हूं", "बोल रहा था",
	)
	if _matches_any_term(lowered, starters) and len(lowered) < 22:
		if is_rumik or is_sarvam:
			return "Ji batayein, main sun rahi hoon. Aapko kya dental assistance chahiye?"
		return "Please go ahead, I am listening. How may I assist you with your dental care?"

	# 5. Affirmation / Ack (only standalone simple confirmations)
	acks = (
		"ok", "okay", "theek hai", "theek", "sure", "got it", "fine", "alright", "yes", "yep",
		"haan", "ha", "haa", "ji", "ji haan", "achha", "accha", "bilkul",
		"हाँ", "हां", "हा", "जी", "जी हाँ", "जी हां", "ठीक है", "ठीक", "अच्छा", "बिल्कुल", "ज़रूर", "जरूर",
	)
	if _matches_any_term(lowered, acks) and len(lowered) < 15:
		if is_rumik or is_sarvam:
			return "Ji bilkul. Batayein, kya doctor consultation ya clinic timings ke baare mein kuch check karoon?"
		return "Certainly! Would you like to check doctor availability or schedule an appointment?"

	# 6. Well-being & Pleasantries ("how are you", "aap kaisi hain", "aap kaise ho")
	wellbeing = (
		"how are you", "how r u", "how do you do",
		"aap kaise ho", "aap kaisi ho", "aap kaise hain", "aap kaisi hain",
		"kya haal hai", "kaise ho", "kaisi ho",
		"आप कैसी हैं", "आप कैसे हैं", "आप कैसे हो", "आप कैसी हो", "क्या हाल है", "कैसे हो", "कैसी हो",
	)
	if _matches_any_term(lowered, wellbeing):
		if is_rumik:
			return "Main bilkul theek hoon ji, poochne ke liye aapka bahut-bahut dhanyawad! Aap batayein, main aapki dental care ya appointment mein kaise madad kar sakti hoon?"
		if is_sarvam:
			return "<happy> Main bilkul theek hoon ji, aapka bahut dhanyawad! <curious> Main aapki kaise madad kar sakti hoon?"
		return "I am doing wonderful, thank you so much for asking! How may I assist you with your dental care or appointment today?"

	# 7. Identity / Who are you
	identity = (
		"who are you", "what is your name", "what's your name",
		"aap kaun ho", "kaun ho", "naam kya hai", "aap kaun hain",
		"आप कौन हैं", "आप कौन हो", "नाम क्या है",
	)
	if _matches_any_term(lowered, identity) and len(lowered) < 35:
		if is_rumik or is_sarvam:
			return f"Main {clinic_name} ki AI receptionist Niaa hoon ji. Aap appointment booking ya kisi bhi dental query ke liye pooch sakte hain."
		return f"I am Niaa, {clinic_name}'s AI phone receptionist! How may I assist you with your dental care or appointment today?"

	# For any other chat or user statements (demands, questions, jokes, comments), return None
	# so the Responder LLM generates a rich, dynamic, contextual response!
	return None


def make_router_node(app):
	"""Groq call #1 — intelligent tool selection with strict ToolSelectionHarness engineering (Non-blocking async)."""
	settings = app.state.settings

	async def router(state):
		clinic_tz = getattr(settings, "CLINIC_TIMEZONE", "Asia/Kolkata")
		try:
			now_dt = datetime.now(ZoneInfo(clinic_tz))
			today_obj = now_dt.date()
		except Exception:
			now_dt = datetime.now()
			today_obj = date.today()

		# Determine whether clinic operating hours (10:00 - 19:00) have ended for today
		close_hr = getattr(settings, "CLINIC_CLOSE_HOUR", 19)
		close_dt = now_dt.replace(hour=close_hr, minute=0, second=0, microsecond=0)
		is_today_closed = (now_dt + timedelta(minutes=90)) >= close_dt or today_obj.weekday() == 6

		# Determine next available open clinic date (skipping closed Sundays and closed hours)
		cand_open = today_obj + timedelta(days=1) if is_today_closed else today_obj
		while cand_open.weekday() == 6:
			cand_open += timedelta(days=1)
		next_open_date_str = cand_open.isoformat()
		next_open_day_name = cand_open.strftime("%A")

		user_text = (state.get("user_input") or "").lower()
		has_pending = bool(state.get("pending_booking_id"))
		has_date_ref = ToolSelectionHarness._contains_date_reference(user_text)
		scheduling_keywords = (
			"book", "slot", "slots", "appointment", "schedule", "timing", "timings",
			"hours", "available", "availability", "free", "open", "visit", "visiting",
			"doctor", "dr.", "dr ", "dentist", "specialist", "reschedule", "cancel",
			"mon", "tue", "wed", "thu", "fri", "sat", "sun", "kal", "parso", "aaj",
			"din", "days", "hafta", "hafte", "week", "weeks", "mahina", "mahine", "month",
		)
		needs_calendar = has_pending or has_date_ref or (state.get("preclassified_intent") == "slots") or any(k in user_text for k in scheduling_keywords)

		if needs_calendar:
			# Build 8-day upcoming calendar lookup (offset 0 to 8) for LLM routing using TOON compact tabular notation
			cal_records = []
			for offset in range(9):
				cur = today_obj + timedelta(days=offset)
				day_name = cur.strftime("%A")
				if offset == 0:
					label = "Today"
					status = "CLOSED (Clinic hours 10:00-19:00 ended for today)" if is_today_closed else "open"
				elif offset == 1:
					label = "Tomorrow (day after today)"
					status = "CLOSED (Sunday)" if day_name == "Sunday" else "open"
				elif offset == 2:
					label = "Day after tomorrow"
					status = "CLOSED (Sunday)" if day_name == "Sunday" else "open"
				elif offset == 8:
					label = f"{day_name} (Max 8-day advance limit)"
					status = "CLOSED (Sunday)" if day_name == "Sunday" else "open"
				else:
					label = day_name
					status = "CLOSED (Sunday)" if day_name == "Sunday" else "open"

				if cur == cand_open:
					status += " [NEXT OPEN CLINIC DAY]"

				cal_records.append({
					"offset": str(offset),
					"label": label,
					"day": day_name,
					"date": cur.isoformat(),
					"status": status,
				})
			calendar_ref = "\n\nUPCOMING CALENDAR REFERENCE (Advance bookings accepted up to 8 days only):\n" + encode_toon(cal_records)
		else:
			calendar_ref = ""

		system = ROUTER_SYSTEM.format(
			clinic=settings.CLINIC_NAME,
			today=today_obj.isoformat(),
			today_day=today_obj.strftime("%A"),
			current_time=now_dt.strftime("%I:%M %p (%H:%M)"),
			next_open_date=next_open_date_str,
			next_open_day=next_open_day_name,
			calendar_reference=calendar_ref,
		)
		messages = [{"role": "system", "content": system}]
		# Sanitize history to strip inline emotion tags so the Router LLM is never prompted into conversational speech
		cleaned_history = []
		for msg in state.get("history", [])[-HISTORY_TAIL:]:
			content = msg.get("content", "")
			clean_c = re.sub(r"<[a-zA-Z0-9_]+>", "", content).strip()
			cleaned_history.append({"role": msg.get("role", "user"), "content": clean_c})
		messages += cleaned_history

		profile = state.get("profile") or {}
		active_profile = {k: v for k, v in profile.items() if v and k in ("name", "phone")}
		if active_profile:
			messages.append({"role": "system", "content": f"CALLER PROFILE (TOON):\n{encode_toon(active_profile)}"})
		pending_ctx = _get_pending_booking_context(app, state.get("pending_booking_id"))
		if pending_ctx:
			messages.append({"role": "system", "content": pending_ctx})
		messages.append({
			"role": "user",
			"content": f"{state['user_input']}\n(Rule: Output valid minified JSON only with 'action' and 'args' keys)",
		})

		raw_decision = await app.state.llm.adecide(messages)
		if not isinstance(raw_decision, dict):
			raw_decision = {}

		# Strict Tool Selection Harness: validation, symptom matching, date/time normalization & repair
		harness_res = ToolSelectionHarness.validate_and_repair(raw_decision, state, app)
		decision = {
			"action": harness_res.action,
			"args": harness_res.args,
		}
		if harness_res.was_repaired:
			decision["harness_repaired"] = True
			decision["harness_issues"] = harness_res.issues

		log.info("router decision (harness-verified): %s", decision)
		return {"decision": decision}

	return router


def make_tool_node(app):
	"""Execute the tool the Router chose in worker threadpool (Non-blocking async)."""

	async def tools(state):
		decision = state["decision"]
		action = decision.get("action", "chitchat")
		if action in SKIP_TOOLS:
			return {
				"tool_result": None,
				"payload_type": "call_end" if action == "end_call" else "response",
			}

		result = await asyncio.to_thread(dispatch, app, state, decision)
		update = {
			"tool_result": result.data,
			"payload_type": result.payload_type,
			"extra": result.extra,
		}
		if result.pending_booking_id:
			update["pending_booking_id"] = result.pending_booking_id
		if result.clear_pending_booking:
			update["pending_booking_id"] = None
		return update

	return tools


def make_responder_node(app):
	"""Groq call #2 — crafts the spoken reply with strict ResponseHarness fact-checking & conditioning (Non-blocking async)."""
	settings = app.state.settings

	async def responder(state):
		action = state.get("decision", {}).get("action", "chitchat")
		provider = (state.get("tts_provider") or getattr(settings, "TTS_PROVIDER", "browser")).lower()

		# Check if a fast-path template response is suitable (e.g. for end_call or pure short greetings)
		fast_reply = _get_deterministic_template_reply(
			action=action,
			user_input=state.get("user_input", ""),
			clinic_name=settings.CLINIC_NAME,
			provider=provider,
			has_pending_booking=bool(state.get("pending_booking_id")),
			has_confirmed_booking=bool(state.get("payload_type") == "booking_confirmed"),
		)
		if fast_reply is not None:
			raw_reply = fast_reply
			log.info("Fast-path template response for action=%s (bypassed Responder LLM call): %s", action, raw_reply)
		else:
			system = RESPONDER_SYSTEM.format(
				clinic=settings.CLINIC_NAME, fee=settings.BOOKING_FEE_RUPEES
			)
			# Dynamically adapt prompt to active TTS engine:
			# - Rumik AI Silk (Aisha): emotionally aware concise Hinglish cadence matching Sarvam, with selective tags (<sigh>, <excited>)
			# - Sarvam AI (Simran): strict 2-sentence concise Hinglish cadence
			if provider == "rumik":
				system += "\n" + RUMIK_AISHA_ADDENDUM
			elif provider == "sarvam":
				system += "\n" + SARVAM_HINGLISH_ADDENDUM

			messages = [{"role": "system", "content": system}]
			messages += state.get("history", [])[-HISTORY_TAIL:]
			profile = state.get("profile") or {}
			active_profile = {k: v for k, v in profile.items() if v and k in ("name", "phone")}
			if active_profile:
				messages.append({"role": "system", "content": f"CALLER PROFILE (TOON):\n{encode_toon(active_profile)}"})
			pending_ctx = _get_pending_booking_context(app, state.get("pending_booking_id"))
			if pending_ctx:
				messages.append({"role": "system", "content": pending_ctx})
			messages.append({"role": "user", "content": state["user_input"]})

			brief = render_tool_brief(
				state.get("decision"), state.get("tool_result"), state.get("payload_type")
			)
			if brief:
				messages.append({"role": "system", "content": brief})

			if state.get("is_angry") or state.get("caller_sentiment") == "angry":
				messages.append({
					"role": "system",
					"content": (
						"CRITICAL EMPATHY & DE-ESCALATION DIRECTIVE: The caller is frustrated or angry. "
						"You MUST open your response with a sincere, respectful apology and reassurance "
						"(e.g. 'Main dil se maafi chahti hoon ji agar aapko koi pareshani hui hai. Main bilkul aapki poori madad karne ke liye yahan hoon.') "
						"then directly answer their request with utmost priority, warmth, and helpfulness."
					),
				})

			raw_reply = await app.state.llm.arespond(messages)

		# Strict Response Harness: fact-checking against tool facts, voice acoustic conditioning & brevity
		harness_res = ResponseHarness.verify_and_condition(raw_reply, state, app)
		conditioned_reply = harness_res.response

		payload_type = "call_end" if action == "end_call" else (state.get("payload_type") or "response")

		return {
			"response": conditioned_reply,
			"payload_type": payload_type,
			"harness_repaired": harness_res.was_repaired,
		}

	return responder

