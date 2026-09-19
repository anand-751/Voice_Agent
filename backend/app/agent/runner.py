import logging

from contextlib import nullcontext as _nullcontext

from .guardrails import apply_output_guardrails, check_input_guardrails
from .state import AgentState

log = logging.getLogger("agent")


from typing import Optional


async def run_turn(app, session, user_input: str, caller_sentiment: str = "neutral", preclassified_intent: Optional[str] = None) -> dict:
	"""One full agentic turn with input/output guardrails: session state → LangGraph → WS payload."""
	clinic_name = getattr(app.state.settings, "CLINIC_NAME", "Bright Dental Clinic")
	provider = (getattr(session, "tts_provider", None) or getattr(app.state.settings, "TTS_PROVIDER", "browser")).lower()
	obs = getattr(getattr(app, "state", None), "observability", None)

	turn_ctx = (
		obs.observe_turn(
			session_id=session.id,
			user_input=user_input,
			profile=session.profile,
			metadata={
				"pending_booking_id": session.pending_booking_id,
				"turn_number": len(session.history) // 2 + 1,
			},
		)
		if obs
		else None
	)

	with turn_ctx if turn_ctx else _nullcontext() as turn_obs:
		# 1. Input Guardrails (sanitization, injection defense, medical refusal, emergency triage)
		gr_ctx = obs.observe_guardrail("input", user_input) if obs else None
		with gr_ctx if gr_ctx else _nullcontext() as gr_obs:
			guardrail_res = check_input_guardrails(user_input, clinic_name=clinic_name, provider=provider)
			if gr_obs:
				gr_obs.update(
					output={"passed": guardrail_res.passed, "action": guardrail_res.action},
					level="WARNING" if not guardrail_res.passed else "DEFAULT",
				)

		if obs:
			obs.score("guardrail_passed", 1.0 if guardrail_res.passed else 0.0)

		if not guardrail_res.passed:
			log.info("Turn blocked by guardrail: action=%s", guardrail_res.action)
			response = guardrail_res.response or "How may I assist you at Bright Dental Clinic today?"
			session.add("user", guardrail_res.sanitized_input or user_input)
			session.add("assistant", response)
			payload = {
				"type": "response",
				"response": response,
				"guardrail_triggered": True,
				"guardrail_action": guardrail_res.action,
			}
			if turn_obs:
				turn_obs.update(output=payload)
			return payload

		sanitized_text = guardrail_res.sanitized_input
		is_caller_angry = guardrail_res.is_angry or caller_sentiment == "angry"
		state: AgentState = {
			"session_id": session.id,
			"user_input": sanitized_text,
			"profile": session.profile,
			"history": list(session.history),
			"pending_booking_id": session.pending_booking_id,
			"tts_provider": provider,
			"is_angry": is_caller_angry,
			"caller_sentiment": "angry" if is_caller_angry else "neutral",
			"preclassified_intent": preclassified_intent,
		}

		try:
			result = await app.state.graph.ainvoke(state)
		except Exception as exc:
			log.exception("Unhandled exception during graph.ainvoke in run_turn: %s", exc)
			if obs:
				obs.score("turn_exception", 0.0, f"Error: {exc}")
			fallback_reply = (
				f"Ek chhota pause lene ke liye maafi chahta hoon ji. {clinic_name} mein kaise madad karoon?"
				if provider == "sarvam"
				else (
					f"I apologize for the brief pause. How can I assist you with your dental care "
					f"or appointment booking at {clinic_name} today?"
				)
			)
			session.add("user", sanitized_text)
			session.add("assistant", fallback_reply)
			payload = {"type": "response", "response": fallback_reply, "turn_recovered": True}
			if turn_obs:
				turn_obs.update(output=payload, status_message=str(exc), level="ERROR")
			return payload

		# 2. Output Guardrails (leakage prevention, prescription filter, spoken formatting)
		raw_response = result.get("response", "")
		out_gr_ctx = obs.observe_guardrail("output", raw_response) if obs else None
		with out_gr_ctx if out_gr_ctx else _nullcontext() as out_gr_obs:
			safe_response = apply_output_guardrails(raw_response, clinic_name=clinic_name, provider=provider)
			if out_gr_obs:
				out_gr_obs.update(output=safe_response)

		result["response"] = safe_response

		# Persist conversation + booking context back to the call session
		session.add("user", sanitized_text)
		session.add("assistant", safe_response)
		session.pending_booking_id = result.get("pending_booking_id")
		if result.get("payload_type") == "booking_confirmed":
			session.confirmed_booking = result.get("extra", {}).get("booking") or True

		payload = build_payload(result)

		# Milestone and conversion scoring
		if obs:
			if payload.get("type") == "booking_confirmed":
				obs.score("booking_milestone", 1.0, "Appointment confirmed")
			elif payload.get("type") == "payment_required":
				obs.score("booking_milestone", 0.7, "Payment link presented")
			elif result.get("decision", {}).get("action") == "calendar.check_availability":
				obs.score("booking_milestone", 0.3, "Checked slot availability")

		if turn_obs:
			turn_obs.update(output=payload)

		return payload


def build_payload(state: AgentState) -> dict:
	payload_type = state.get("payload_type") or "response"
	payload = {"type": payload_type, "response": state.get("response", "")}
	extra = state.get("extra") or {}

	if payload_type == "payment_required":
		payload["checkout_url"] = extra.get("checkout_url")
		payload["booking"] = (
			extra.get("booking")
			or (state.get("tool_result") or {}).get("booking")
			or ({"id": state.get("pending_booking_id")} if state.get("pending_booking_id") else None)
		)
	if payload_type == "booking_confirmed":
		payload["booking"] = (
			extra.get("booking")
			or (state.get("tool_result") or {}).get("booking")
		)
	return payload
