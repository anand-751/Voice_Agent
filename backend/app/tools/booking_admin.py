import logging

from .base import ToolResult


log = logging.getLogger("tools.booking")


def cancel_booking(app, state, args) -> ToolResult:
	store = app.state.bookings
	booking = None
	if state.get("pending_booking_id"):
		booking = store.get(state["pending_booking_id"])
	if booking is None:
		phone = (state.get("profile") or {}).get("phone")
		booking = store.latest_pending(phone)

	if booking is None:
		return ToolResult(data={"cancelled": False,
								"note": "No pending booking to cancel."})

	store.cancel(booking.id)
	return ToolResult(
		data={"cancelled": True, "date": booking.date, "time": booking.time},
		clear_pending_booking=True,
	)
