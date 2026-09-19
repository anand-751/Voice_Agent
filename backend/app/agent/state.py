from typing import Optional, TypedDict


class AgentState(TypedDict, total=False):
	# ── input ────────────────────────────────────────────
	session_id: str
	user_input: str
	profile: dict
	history: list
	pending_booking_id: Optional[str]
	preclassified_intent: Optional[str]

	# ── router output (Groq call #1) ─────────────────────
	decision: dict          # {"action": ..., "args": {...}, "reason": ...}

	# ── tool output ──────────────────────────────────────
	tool_result: Optional[dict]
	extra: dict             # checkout_url / booking for the WS payload

	# ── responder output (Groq call #2) ──────────────────
	response: str
	payload_type: str       # response | payment_required | booking_confirmed | call_end
