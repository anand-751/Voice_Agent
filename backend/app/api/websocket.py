import asyncio
import json
import logging
import re
import time
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..agent.runner import run_turn
from ..services.call_queue import CallQueueManager
from ..services.sarvam_stt import SarvamStreamingSTT
from ..services.session import SessionStore

log = logging.getLogger("ws")
router = APIRouter()

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
)

TRAILING_INCOMPLETE_PHRASES = (
	"bol raha hoon ki", "bol raha tha ki", "keh raha tha ki", "chahiye tha ki",
	"ki main", "ki mujhe", "bol raha hoon", "bol raha tha",
	"बोल रहा हूं कि", "बोल रहा था कि", "कह रहा था कि", "चाहिए था कि",
	"कि मैं", "कि मुझे", "बोल रहा हूं", "बोल रहा था",
)


def is_incomplete_utterance(text: str) -> bool:
	"""Check if an utterance was paused mid-sentence or ends on an incomplete connector."""
	clean = (text or "").strip()
	if not clean:
		return False
	lowered = clean.lower()
	end_stripped = re.sub(r"[।.,!?;:\s]+$", "", lowered).strip()
	words = [w for w in re.split(r"\s+", end_stripped) if w]
	if not words:
		return False
	# Check if whole utterance is a short starter/hesitation phrase (< 4 words)
	if len(words) <= 4:
		for hp in HESITATION_PHRASES:
			if hp in end_stripped:
				return True
	# Check trailing word
	last_word = words[-1]
	if last_word in INCOMPLETE_TRAILING_WORDS:
		return True
	# Check trailing incomplete multi-word phrases
	for tm in TRAILING_INCOMPLETE_PHRASES:
		if end_stripped.endswith(tm):
			return True
	return False


# Isolated hesitation sounds, filler noises, and single-syllable acoustic artifacts
STRAY_NOISE_OR_FILLER = {
	"अह", "अह अह", "अह अह।", "अहा", "तेन", "हम्म", "हम", "hmm", "uh", "um", "ah",
	"oh", "oho", "shh", "huh", "eh", "er", "haa", "ha", "haan", "हाँ", "हां",
}


def is_stray_noise_or_filler(text: str) -> bool:
	"""Check if text is an isolated hesitation sound, non-committal filler, or single acoustic artifact."""
	clean = (text or "").strip()
	lowered = clean.lower()
	end_stripped = re.sub(r"[।.,!?;:\s]+$", "", lowered).strip()
	if not end_stripped:
		return True
	if end_stripped in STRAY_NOISE_OR_FILLER or clean in STRAY_NOISE_OR_FILLER:
		return True
	# Check if single token of <= 3 characters without actionable keywords
	words = [w for w in re.split(r"\s+", end_stripped) if w]
	if len(words) == 1 and len(words[0]) <= 3 and words[0] not in {"yes", "no", "kal", "aaj", "doc", "dr"}:
		return True
	return False

_sessions = SessionStore()
_default_queue: Optional[CallQueueManager] = None


def _get_queue(settings) -> CallQueueManager:
	global _default_queue
	if _default_queue is None:
		_default_queue = CallQueueManager(
			max_concurrent=settings.MAX_CONCURRENT_CALLS,
			max_duration_seconds=settings.CALL_MAX_DURATION_SECONDS,
		)
	return _default_queue


@router.websocket("/conversation/stream")
async def conversation_stream(ws: WebSocket):
	"""One WebSocket equals one phone call or queued caller.
	
	Supports:
	- Streaming Linear16 16kHz PCM audio frames (via binary chunks or JSON {"type": "audio"})
	- Real-time Sarvam STT transcription with sub-second VAD turn-taking
	- Instant barge-in interrupt signalling
	- Standard text queries for backwards compatibility & typing
	"""
	await ws.accept()
	app = ws.app
	session = _sessions.create()
	settings = app.state.settings
	call_queue = getattr(app.state, "call_queue", None) or _get_queue(settings)

	# Enter active slot or wait in FIFO queue
	admitted = await call_queue.enter_or_wait(ws, session.id)
	if not admitted:
		_sessions.drop(session.id)
		return

	# Populate caller profile from query params if available (e.g. ?name=...&phone=...&email=...)
	q_name = (ws.query_params.get("name") or "").strip()
	q_phone = (ws.query_params.get("phone") or "").strip()
	q_email = (ws.query_params.get("email") or "").strip()
	if q_name:
		session.profile["name"] = q_name
	if q_phone:
		session.profile["phone"] = q_phone
	if q_email:
		session.profile["email"] = q_email

	# Fallback: if caller profile name is still empty, populate from latest saved DB profile
	if not session.profile.get("name") and hasattr(app.state, "bookings") and hasattr(app.state.bookings, "get_latest_profile"):
		try:
			latest_prof = app.state.bookings.get_latest_profile()
			if latest_prof and latest_prof.get("name"):
				for k, v in latest_prof.items():
					if v and not session.profile.get(k):
						session.profile[k] = v
				log.info("Populated session %s profile from database: %s", session.id, session.profile)
		except Exception:
			pass

	log.info("call started: %s (caller: %s)", session.id, session.profile.get("name") or "anonymous")

	stt_client: Optional[SarvamStreamingSTT] = None
	turn_lock = asyncio.Lock()
	is_agent_speaking = False
	was_interrupted = False
	last_agent_speech_end = 0.0

	current_turn_task: Optional[asyncio.Task] = None
	in_flight_user_text: str = ""
	interrupted_user_query: str = ""

	pending_continuation: str = ""
	pending_continuation_timer: Optional[asyncio.Task] = None
	turn_index: int = 0
	last_filler_text: str = ""
	last_filler_turn: int = -99

	# Graceful timeout hooks: Call queue checks these before ending a call
	active_call = call_queue.active_calls.get(session.id)
	if active_call:
		def _check_speech_busy() -> bool:
			turn_busy = bool(current_turn_task and not current_turn_task.done())
			speaking_busy = bool(is_agent_speaking)
			text_busy = bool(in_flight_user_text)
			recent_speaking = (time.time() - last_agent_speech_end) < 2.0
			return turn_busy or speaking_busy or text_busy or recent_speaking

		async def _wait_for_speech_done():
			# 1. Wait for in-flight turn generation to finish
			if current_turn_task and not current_turn_task.done():
				try:
					await asyncio.wait_for(asyncio.shield(current_turn_task), timeout=25.0)
				except Exception:
					pass
			# 2. Wait for audio playback on frontend to finish
			poll_start = time.time()
			while is_agent_speaking and (time.time() - poll_start < 25.0):
				await asyncio.sleep(0.2)
			# 3. Brief breather after speech ends so caller hears the whole sentence
			await asyncio.sleep(1.0)

		active_call.is_speech_in_progress = _check_speech_busy
		active_call.wait_for_speech_complete = _wait_for_speech_done

	async def handle_turn(user_text: str, preclassified_intent: Optional[str] = None, caller_sentiment: str = "neutral"):
		nonlocal is_agent_speaking, last_agent_speech_end, was_interrupted, turn_index, last_filler_text, last_filler_turn
		clean_text = (user_text or "").strip()
		if not clean_text:
			return

		async with turn_lock:
			is_agent_speaking = True
			was_interrupted = False
			turn_index += 1
			log.info("Processing turn #%d for session %s (sentiment=%s): %s", turn_index, session.id, caller_sentiment, clean_text)

			# Context-Aware Latency Masking:
			# Differentiates 'payment', 'slots', 'info', 'short', and None (chit-chat).
			# Anti-repetition: Never repeats the exact same filler phrase, and never spams fillers on consecutive turns unless intent is 'payment'.
			current_prov = (getattr(session, "tts_provider", None) or getattr(settings, "TTS_PROVIDER", "sarvam")).lower()
			intent = preclassified_intent
			if intent is None:
				llm_svc = getattr(app.state, "llm", None)
				if llm_svc and hasattr(llm_svc, "classify_filler_intent"):
					intent = await llm_svc.classify_filler_intent(clean_text)

			# Strict Gap Filler Gating:
			# Only fill the latency gap for explicitly defined intents: 'payment', 'slots', 'info', 'short', 'heavy'.
			# If intent is not matched (None, general talk, chit-chat, pleasantries, non-dental complaints),
			# DO NOT trigger any gap filler. Let the main agent give its answer with its normal latency.
			DEFINED_FILLER_INTENTS = {"payment", "slots", "info", "short", "heavy"}
			should_send_filler = bool(caller_sentiment != "angry" and intent in DEFINED_FILLER_INTENTS)

			filler = None
			if should_send_filler and intent:
				if current_prov == "sarvam":
					sarvam_svc = getattr(app.state, "sarvam", None)
					if sarvam_svc and sarvam_svc.is_configured:
						filler = sarvam_svc.get_contextual_filler_audio(intent, last_phrase=last_filler_text)
					if not filler:
						rumik_svc = getattr(app.state, "rumik", None)
						if rumik_svc and rumik_svc.is_configured:
							filler = rumik_svc.get_contextual_filler_audio(intent, last_phrase=last_filler_text)
				else:
					rumik_svc = getattr(app.state, "rumik", None)
					if rumik_svc and rumik_svc.is_configured:
						filler = rumik_svc.get_contextual_filler_audio(intent, last_phrase=last_filler_text)
					if not filler:
						sarvam_svc = getattr(app.state, "sarvam", None)
						if sarvam_svc and sarvam_svc.is_configured:
							filler = sarvam_svc.get_contextual_filler_audio(intent, last_phrase=last_filler_text)

				if filler:
					last_filler_text = filler["text"]
					last_filler_turn = turn_index
					try:
						await ws.send_json({
							"type": "filler_audio",
							"text": filler["text"],
							"audio_b64": filler["audio_b64"],
							"turn_index": turn_index,
							"tts_provider": filler.get("tts_provider", current_prov),
						})
						log.info("Sent contextual '%s' filler audio ('%s') to mask latency", intent, filler["text"])
					except Exception:
						pass

			try:
				await ws.send_json({"type": "agent_thinking"})
			except Exception:
				pass

			try:
				payload = await run_turn(app, session, clean_text, caller_sentiment=caller_sentiment, preclassified_intent=intent)

				# Real Latency Optimization: Direct Backend TTS Synthesis
				# Pre-synthesizes audio directly on the backend and attaches audio_b64 to the WS payload.
				# Eliminates the client -> backend HTTP roundtrip via ngrok, saving ~900ms - 1.2s!
				resp_text = payload.get("response") or ""
				current_prov = (getattr(session, "tts_provider", None) or getattr(settings, "TTS_PROVIDER", "sarvam")).lower()
				rumik_svc = getattr(app.state, "rumik", None)
				if current_prov == "rumik" and (not rumik_svc or not getattr(rumik_svc, "is_configured", True)):
					current_prov = "sarvam"
				if resp_text and current_prov in ("sarvam", "rumik"):
					audio_b64 = None
					prov_used = current_prov
					if current_prov == "rumik":
						rumik_svc = getattr(app.state, "rumik", None)
						if rumik_svc and rumik_svc.is_configured:
							try:
								t0 = time.time()
								audio_b64 = await rumik_svc.synthesize_b64(resp_text)
								if audio_b64:
									prov_used = "rumik"
									log.info("Direct Rumik TTS pre-synthesized in %.2fs (attached to WS payload)", time.time() - t0)
							except Exception as r_err:
								log.warning("Direct Rumik TTS synthesis error (falling back to Sarvam): %s", r_err)

					if not audio_b64:
						sarvam_svc = getattr(app.state, "sarvam", None)
						if sarvam_svc and sarvam_svc.is_configured:
							try:
								t0 = time.time()
								audio_b64 = await sarvam_svc.synthesize_b64(resp_text)
								if audio_b64:
									prov_used = "sarvam"
									log.info("Direct Sarvam TTS pre-synthesized in %.2fs (attached to WS payload)", time.time() - t0)
							except Exception as tts_err:
								log.warning("Direct Sarvam TTS synthesis error (falling back to client fetch): %s", tts_err)

					if audio_b64:
						payload["audio_b64"] = audio_b64
						payload["tts_provider"] = prov_used

				await ws.send_json(payload)
				last_agent_speech_end = time.time()
				if payload.get("type") == "payment_required":
					await call_queue.pause_timeout(session.id)
				elif payload.get("type") == "booking_confirmed":
					await call_queue.resume_timeout(session.id)
			except asyncio.CancelledError:
				log.info("handle_turn cancelled for session %s: %s", session.id, clean_text)
				raise
			except Exception as turn_err:
				log.exception("Error executing turn: %s", turn_err)
				await ws.send_json({
					"type": "response",
					"response": "Kripya ek baar aur dohraiye ji, main samajh nahi paayi.",
				})

	async def handle_turn_wrapper(text: str, intent: Optional[str] = None, caller_sentiment: str = "neutral"):
		nonlocal in_flight_user_text
		in_flight_user_text = text
		try:
			await handle_turn(text, preclassified_intent=intent, caller_sentiment=caller_sentiment)
		finally:
			in_flight_user_text = ""

	async def _continuation_timeout_handler(buffered_text: str):
		nonlocal pending_continuation, current_turn_task
		try:
			# Debounce duration: 0.75s waiting for caller to finish after pause
			await asyncio.sleep(0.75)
		except asyncio.CancelledError:
			return
		if pending_continuation == buffered_text:
			pending_continuation = ""
			current_turn_task = asyncio.create_task(handle_turn_wrapper(buffered_text))

	# Callbacks for Sarvam Streaming STT
	async def on_stt_speech_start(utterance_idx: int):
		nonlocal pending_continuation_timer
		log.debug("STT speech_start event received for utterance %s", utterance_idx)

		# Cancel pending continuation timer if caller resumed speaking
		if pending_continuation_timer and not pending_continuation_timer.done():
			pending_continuation_timer.cancel()
			pending_continuation_timer = None

		try:
			await ws.send_json({"type": "speech_start", "utterance_idx": utterance_idx})
		except Exception:
			pass

	async def on_stt_partial(text: str, utterance_idx: int):
		nonlocal pending_continuation_timer, is_agent_speaking
		# Cancel continuation timer as soon as user starts speaking more words
		if pending_continuation_timer and not pending_continuation_timer.done():
			pending_continuation_timer.cancel()
			pending_continuation_timer = None

		clean_partial = (text or "").strip()
		# Fast Barge-In: If user speaks substantive words while agent is speaking, trigger interrupt
		if is_agent_speaking and len(clean_partial) >= 4 and not is_stray_noise_or_filler(clean_partial):
			try:
				await ws.send_json({"type": "interrupt"})
				is_agent_speaking = False
			except Exception:
				pass

		try:
			await ws.send_json({
				"type": "interim_transcript",
				"text": text,
				"utterance_idx": utterance_idx,
			})
		except Exception:
			pass

	async def on_stt_final(text: str, utterance_idx: int):
		nonlocal is_agent_speaking, was_interrupted, current_turn_task, pending_continuation_timer, pending_continuation, interrupted_user_query, last_agent_speech_end, turn_index
		incoming_text = (text or "").strip()
		if not incoming_text:
			return

		# Cancel any active continuation timer as a new final utterance arrived
		if pending_continuation_timer and not pending_continuation_timer.done():
			pending_continuation_timer.cancel()
			pending_continuation_timer = None

		prefix = pending_continuation or interrupted_user_query or ""

		# Call the single unified first-layer classifier (Groq + Context)
		llm_svc = getattr(app.state, "llm", None)
		current_prov = (getattr(session, "tts_provider", None) or getattr(settings, "TTS_PROVIDER", "sarvam")).lower()
		tts_svc = getattr(app.state, current_prov, None)
		if current_prov == "rumik" and (not tts_svc or not getattr(tts_svc, "is_configured", True)):
			current_prov = "sarvam"
			tts_svc = getattr(app.state, "sarvam", None)
		if not tts_svc:
			tts_svc = getattr(app.state, "sarvam", None) or getattr(app.state, "rumik", None)
		hist_snapshot = list(session.history)[-4:] if hasattr(session, "history") else []

		has_pending = bool(session.pending_booking_id)
		has_confirmed = bool(getattr(session, "confirmed_booking", None))
		decision = None
		if llm_svc and hasattr(llm_svc, "decide_turn_action"):
			decision = await llm_svc.decide_turn_action(
				incoming_text=incoming_text,
				conversation_history=hist_snapshot,
				is_agent_speaking=is_agent_speaking,
				pending_continuation=prefix,
				provider=current_prov,
				has_pending_booking=has_pending,
				has_confirmed_booking=has_confirmed,
			)
		else:
			is_incomplete = is_incomplete_utterance(incoming_text)
			decision = {
				"decision": "WAIT_AND_LISTEN" if is_incomplete else "EXECUTE_TURN",
				"sentiment": "neutral",
				"intent": None,
				"backchannel": "Ji...",
				"deescalation_response": None,
				"clean_query": f"{prefix} {incoming_text}".strip() if prefix else incoming_text,
			}

		action = decision.get("decision", "EXECUTE_TURN")
		clean_query = decision.get("clean_query") or (f"{prefix} {incoming_text}".strip() if prefix else incoming_text)
		intent = decision.get("intent")
		sentiment = decision.get("sentiment", "neutral")
		backchannel = decision.get("backchannel") or "Ji..."
		deescalation = decision.get("deescalation_response")

		log.info(
			"First-layer turn decision for session %s: action=%s, sentiment=%s, intent=%s, incoming='%s', clean_query='%s'",
			session.id, action, sentiment, intent, incoming_text, clean_query
		)

		# 1. SECURITY_INTERCEPT: Prompt injection / jailbreak attack intercepted before touching tools
		if action == "SECURITY_INTERCEPT":
			log.warning("First-layer blocked prompt injection for session %s: '%s'", session.id, incoming_text)
			clinic_name = getattr(settings, "CLINIC_NAME", "Bright Dental Clinic")
			refusal_text = deescalation or (
				f"Ji main {clinic_name} ki AI receptionist Niaa hoon ji. "
				f"Aap dental appointments, treatments ya clinic services ke baare mein pooch sakte hain, batayein kaise help karoon?"
			)
			session.add("user", clean_query)
			session.add("assistant", refusal_text)

			try:
				await ws.send_json({
					"type": "final_transcript",
					"text": clean_query,
					"utterance_idx": utterance_idx,
				})
			except Exception:
				pass

			audio_b64 = None
			if tts_svc and hasattr(tts_svc, "synthesize_b64"):
				try:
					audio_b64 = await tts_svc.synthesize_b64(refusal_text)
				except Exception:
					pass

			payload = {
				"type": "response",
				"response": refusal_text,
				"guardrail_triggered": True,
				"guardrail_action": "injection_blocked",
			}
			if audio_b64:
				payload["audio_b64"] = audio_b64
				payload["tts_provider"] = current_prov

			await ws.send_json(payload)
			last_agent_speech_end = time.time()
			return

		# 1b. FAST_RESPONSE or CALL_END: Greetings, Chit-chat, Acknowledgments, Compliments,
		# or Farewell / Call Ending.
		# Gap filler audio is NOT triggered. The main agent engine (run_turn) DOES NOT RUN.
		# Classifier produces ultra-fast direct response (<250ms), pre-synthesizes audio via TTS, and returns immediately.
		is_call_end = (
			action == "CALL_END"
			or decision.get("is_farewell")
			or decision.get("fast_category") == "farewell"
		)
		if action in ("FAST_RESPONSE", "CALL_END") or decision.get("direct_response") or is_call_end:
			resp_text = decision.get("direct_response") or deescalation or "Ji batayein, main aapki kaise madad kar sakti hoon?"
			log.info(
				"Fast response intercept for session %s (action=%s, is_call_end=%s, main agent bypassed): '%s'",
				session.id, action, is_call_end, resp_text
			)

			# Cancel any in-flight background turn
			if current_turn_task and not current_turn_task.done():
				current_turn_task.cancel()
				try:
					await current_turn_task
				except (asyncio.CancelledError, Exception):
					pass

			pending_continuation = ""
			interrupted_user_query = ""

			session.add("user", clean_query)
			session.add("assistant", resp_text)

			# Send final transcript for user utterance
			try:
				await ws.send_json({
					"type": "final_transcript",
					"text": clean_query,
					"utterance_idx": utterance_idx,
				})
			except Exception:
				pass

			# Pre-synthesize audio directly on backend
			audio_b64 = None
			prov_used = current_prov
			if current_prov == "rumik":
				rumik_svc = getattr(app.state, "rumik", None)
				if rumik_svc and rumik_svc.is_configured:
					try:
						audio_b64 = await rumik_svc.synthesize_b64(resp_text)
					except Exception as r_err:
						log.warning("Fast response Rumik TTS error (fallback to Sarvam): %s", r_err)

			if not audio_b64:
				sarvam_svc = getattr(app.state, "sarvam", None)
				if sarvam_svc and sarvam_svc.is_configured:
					try:
						audio_b64 = await sarvam_svc.synthesize_b64(resp_text)
						if audio_b64:
							prov_used = "sarvam"
					except Exception as s_err:
						log.warning("Fast response Sarvam TTS error: %s", s_err)

			payload = {
				"type": "call_end" if is_call_end else "response",
				"response": resp_text,
				"direct_fast_response": True,
			}
			if is_call_end:
				payload["reason"] = "caller_goodbye"

			if audio_b64:
				payload["audio_b64"] = audio_b64
				payload["tts_provider"] = prov_used

			is_agent_speaking = True
			await ws.send_json(payload)
			last_agent_speech_end = time.time()

			if is_call_end:
				async def _farewell_close_watchdog():
					try:
						await asyncio.sleep(20.0)
						if ws.client_state.name == "CONNECTED":
							log.info("Closing session %s WebSocket after farewell speech timeout", session.id)
							await ws.close(code=1000)
					except Exception:
						pass
				asyncio.create_task(_farewell_close_watchdog())

			return

		# 2. DROP_NOISE: Acoustic cough, breath, or background mic bump -> drop completely
		if action == "DROP_NOISE":
			log.info("Dropping noise utterance '%s' (is_agent_speaking=%s)", incoming_text, is_agent_speaking)
			return

		# 3. WAIT_AND_LISTEN: Caller paused mid-thought or started with hesitation -> play soft backchannel & buffer
		if action == "WAIT_AND_LISTEN":
			pending_continuation = clean_query
			interrupted_user_query = ""

			# Play soft listening backchannel ("Ji...", "Haan ji...") from pre-warmed cache
			if tts_svc and hasattr(tts_svc, "get_contextual_filler_audio"):
				backchannel_clip = tts_svc.get_contextual_filler_audio("listening")
				if not backchannel_clip:
					alt_svc = getattr(app.state, "rumik" if current_prov == "sarvam" else "sarvam", None)
					if alt_svc and hasattr(alt_svc, "get_contextual_filler_audio"):
						backchannel_clip = alt_svc.get_contextual_filler_audio("listening")

				if backchannel_clip:
					try:
						await ws.send_json({
							"type": "filler_audio",
							"text": backchannel_clip["text"],
							"audio_b64": backchannel_clip["audio_b64"],
							"turn_index": turn_index,
							"tts_provider": backchannel_clip.get("tts_provider", current_prov),
						})
						log.info("Sent soft listening backchannel '%s' to caller", backchannel_clip["text"])
					except Exception:
						pass

			pending_continuation_timer = asyncio.create_task(_continuation_timeout_handler(clean_query))
			return

		# 4. EXECUTE_TURN: Caller finished their thought or instruction
		pending_continuation = ""
		interrupted_user_query = ""

		# If an earlier turn was in-flight (agent was computing or speaking mid-sentence),
		# cancel the unfinished turn task cleanly and pop the interrupted fragment
		if current_turn_task and not current_turn_task.done():
			log.info("Cancelling in-flight turn task for session %s to execute new clubbed query", session.id)
			current_turn_task.cancel()
			try:
				await current_turn_task
			except (asyncio.CancelledError, Exception):
				pass

		if prefix and session.history and len(session.history) >= 2:
			last_user_entry = session.history[-2]["content"] if session.history[-2]["role"] == "user" else ""
			if last_user_entry and is_incomplete_utterance(last_user_entry):
				popped = session.pop_last_turn()
				if popped:
					log.info("Superseded previous incomplete turn in session history: '%s'", popped.get("user"))

		was_interrupted = False

		# Send WebSocket UI update: turn_superseded if clubbed with prefix, otherwise final_transcript
		if prefix:
			try:
				await ws.send_json({
					"type": "turn_superseded",
					"combined_text": clean_query,
					"utterance_idx": utterance_idx,
				})
			except Exception:
				pass
		else:
			try:
				await ws.send_json({
					"type": "final_transcript",
					"text": clean_query,
					"utterance_idx": utterance_idx,
				})
			except Exception:
				pass

		current_turn_task = asyncio.create_task(handle_turn_wrapper(clean_query, intent, caller_sentiment=sentiment))

	async def on_stt_error(err_data: dict):
		log.warning("Sarvam STT reported error: %s", err_data)
		try:
			await ws.send_json({
				"type": "stt_error",
				"message": err_data.get("message", "Speech recognition error"),
			})
		except Exception:
			pass

	# Initialize Sarvam STT if enabled
	use_sarvam_stt = (
		getattr(settings, "STT_PROVIDER", "sarvam") == "sarvam"
		and bool(getattr(settings, "SARVAM_API_KEY", None))
	)

	if use_sarvam_stt:
		stt_client = SarvamStreamingSTT(
			api_key=settings.SARVAM_API_KEY,
			model=getattr(settings, "SARVAM_STT_MODEL", "saaras:v3-realtime"),
			language_code=getattr(settings, "SARVAM_STT_LANGUAGE", "hi-IN"),
			endpoint=getattr(settings, "SARVAM_STT_ENDPOINT", "wss://api.sarvam.ai/speech-to-text-realtime/ws"),
			silence_duration_ms=getattr(settings, "SARVAM_STT_SILENCE_MS", 700),
			sample_rate=getattr(settings, "SARVAM_STT_SAMPLE_RATE", 16000),
			on_speech_start=on_stt_speech_start,
			on_partial=on_stt_partial,
			on_final=on_stt_final,
			on_error=on_stt_error,
		)
		# Connect STT asynchronously
		asyncio.create_task(stt_client.connect())

	try:
		# Check if frontend passed active tts_provider in query params
		prov_param = (ws.query_params.get("tts_provider") or "").lower()
		if prov_param in ("rumik", "sarvam", "browser"):
			session.tts_provider = prov_param

		max_duration = getattr(settings, "CALL_MAX_DURATION_SECONDS", 180)
		clinic_name = getattr(settings, "CLINIC_NAME", "Bright Dental Clinic")
		current_prov = (getattr(session, "tts_provider", None) or getattr(settings, "TTS_PROVIDER", "sarvam")).lower()
		rumik_svc = getattr(app.state, "rumik", None)
		if current_prov == "rumik" and (not rumik_svc or not rumik_svc.is_configured):
			current_prov = "sarvam"
		if current_prov in ("rumik", "sarvam"):
			greeting_text = f"Namaste! {clinic_name} mein aapka swagat hai. Main Niaa bol rahi hoon, main aapki kaise madad kar sakti hoon?"
		else:
			greeting_text = f"Hello, welcome to {clinic_name}. I am Niaa, how may I help you today?"
		start_audio_b64 = None
		prov_used = current_prov

		if current_prov == "rumik":
			if rumik_svc and rumik_svc.is_configured:
				try:
					start_audio_b64 = await rumik_svc.synthesize_b64(greeting_text)
					prov_used = "rumik"
				except Exception as exc:
					log.warning("Could not pre-synthesize start greeting with Rumik: %s", exc)

		if not start_audio_b64 and current_prov in ("sarvam", "rumik"):
			sarvam_svc = getattr(app.state, "sarvam", None)
			if sarvam_svc and sarvam_svc.is_configured:
				try:
					start_audio_b64 = await sarvam_svc.synthesize_b64(greeting_text)
					prov_used = "sarvam"
				except Exception as exc:
					log.warning("Could not pre-synthesize start greeting: %s", exc)

		start_payload = {
			"type": "start",
			"session_id": session.id,
			"max_duration_seconds": max_duration,
			"stt_provider": "sarvam" if use_sarvam_stt else "browser",
			"sample_rate": getattr(settings, "SARVAM_STT_SAMPLE_RATE", 16000),
			"greeting": greeting_text,
		}
		if start_audio_b64:
			start_payload["audio_b64"] = start_audio_b64
			start_payload["tts_provider"] = prov_used

		await ws.send_json(start_payload)

		while True:
			message = await ws.receive()
			msg_type = message.get("type")

			if msg_type == "websocket.disconnect":
				raise WebSocketDisconnect

			# 1. Binary Audio Frame (Linear16 16kHz PCM)
			if "bytes" in message and message["bytes"]:
				audio_bytes = message["bytes"]
				if stt_client and stt_client.is_connected:
					await stt_client.send_audio(audio_bytes)
				continue

			# 2. Text Message (JSON payload)
			if "text" in message and message["text"]:
				try:
					data = json.loads(message["text"])
				except json.JSONDecodeError:
					continue

				# Audio chunk sent as JSON base64
				if data.get("type") == "audio":
					b64_audio = data.get("data") or ""
					if b64_audio and stt_client and stt_client.is_connected:
						await stt_client.send_audio(b64_audio)
					continue

				# Heartbeat ping/pong keepalive
				if data.get("type") == "ping":
					await ws.send_json({"type": "pong"})
					continue

				# State signal: Agent audio started/stopped playing on frontend
				if data.get("type") == "agent_speaking":
					is_agent_speaking = bool(data.get("speaking"))
					if not is_agent_speaking:
						last_agent_speech_end = time.time()
					continue

				# Dynamic TTS engine switch signal from frontend
				if data.get("type") == "set_tts_provider":
					new_prov = (data.get("provider") or "").lower()
					if new_prov in ("rumik", "sarvam", "browser"):
						session.tts_provider = new_prov
						log.info("Session %s switched TTS provider to: %s", session.id, new_prov)
					continue

				if data.get("tts_provider"):
					session.tts_provider = data.get("tts_provider").lower()

				# Profile handshake or update
				if data.get("type") == "init_profile" or data.get("user_profile"):
					profile = data.get("user_profile") or {}
					if profile:
						session.profile.update({k: v for k, v in profile.items() if v})
						log.info("Session %s caller profile updated: %s", session.id, session.profile)
						if hasattr(app.state, "bookings") and hasattr(app.state.bookings, "save_profile"):
							try:
								app.state.bookings.save_profile(
									name=session.profile.get("name", ""),
									phone=session.profile.get("phone", ""),
									email=session.profile.get("email", ""),
								)
							except Exception:
								pass
					if data.get("type") == "init_profile":
						continue

				# Direct text message (typing / browser speech fallback)
				user_input = (data.get("user_input") or "").strip()
				if user_input:
					if pending_continuation_timer and not pending_continuation_timer.done():
						pending_continuation_timer.cancel()
						pending_continuation_timer = None
					if current_turn_task and not current_turn_task.done():
						current_turn_task.cancel()
					current_turn_task = asyncio.create_task(handle_turn_wrapper(user_input))

	except WebSocketDisconnect:
		pass
	except Exception:
		log.exception("ws error in session %s", session.id)
	finally:
		if pending_continuation_timer and not pending_continuation_timer.done():
			pending_continuation_timer.cancel()
		if current_turn_task and not current_turn_task.done():
			current_turn_task.cancel()
		if stt_client:
			await stt_client.close()
		await call_queue.leave(session.id)
		_sessions.drop(session.id)
		log.info("call ended: %s", session.id)
