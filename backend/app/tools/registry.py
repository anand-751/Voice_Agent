import logging

from .base import ToolResult
from .booking_admin import cancel_booking
from .calendar_tool import check_availability, select_slot
from .payment_tool import verify_payment
from .rag import rag_answer


log = logging.getLogger("tools")

# action name (chosen by the Router LLM) → tool function
DISPATCH = {
	"rag.query": rag_answer,
	"calendar.check_availability": check_availability,
	"calendar.select_slot": select_slot,
	"payment.verify": verify_payment,
	"booking.cancel": cancel_booking,
}


def dispatch(app, state, decision) -> ToolResult:
	action = decision.get("action", "rag.query")
	fn = DISPATCH.get(action)
	if fn is None:
		log.warning("unknown action %r — falling back to RAG", action)
		fn = rag_answer
		decision = {"action": "rag.query",
					"args": {"question": state["user_input"]}}
		action = "rag.query"

	args = decision.get("args") or {}
	obs = getattr(getattr(app, "state", None), "observability", None)

	if obs:
		with obs.observe_tool(action, args) as tool_obs:
			try:
				res = fn(app, state, args)
				if tool_obs:
					tool_obs.update(output=res.data)
				return res
			except Exception as exc:
				log.exception("tool %s failed", action)
				if tool_obs:
					tool_obs.update(
						output={"error": True, "details": str(exc)},
						level="ERROR",
						status_message=f"Tool {action} failed: {exc}",
					)
				return ToolResult(
					data={"error": True,
						  "note": "The system had a hiccup. Apologise and ask the user to repeat."},
					payload_type="response",
				)

	try:
		return fn(app, state, args)
	except Exception:
		log.exception("tool %s failed", action)
		return ToolResult(
			data={"error": True,
				  "note": "The system had a hiccup. Apologise and ask the user to repeat."},
			payload_type="response",
		)
