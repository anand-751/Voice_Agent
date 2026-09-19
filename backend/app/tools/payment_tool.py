import logging

from .base import ToolResult
from ..services.booking_flow import finalize_paid_booking
from ..services.stripe_service import StripeService


log = logging.getLogger("tools.payment")


def verify_payment(app, state, args) -> ToolResult:
	"""Runs when the user returns from Stripe and says 'paid'.

	Source of truth = the booking store, which the Stripe webhook updates.
	Fallback = ask the Stripe API directly (covers local dev without webhook
	forwarding). The calendar event is created here if the webhook hasn't yet.
	"""
	settings = app.state.settings
	store = app.state.bookings

	booking = None
	if state.get("pending_booking_id"):
		booking = store.get(state["pending_booking_id"])
	if booking is None:
		phone = (state.get("profile") or {}).get("phone")
		booking = store.latest_pending(phone)

	if booking is None:
		return ToolResult(data={
			"status": "no_booking",
			"note": "No pending booking found for this user.",
		})

	if booking.status == "confirmed":
		return _confirmed(booking)

	# webhook or simulate-payment endpoint marked it paid
	if booking.status in ("paid", "confirmed"):
		booking = finalize_paid_booking(app, booking.id)
		return _confirmed(booking)

	# fallback: poll live Stripe directly
	if booking.stripe_session_id and not booking.stripe_session_id.startswith("cs_mock_"):
		try:
			if StripeService(settings).session_paid(booking.stripe_session_id):
				booking = finalize_paid_booking(app, booking.id)
				return _confirmed(booking)
		except Exception:
			log.exception("stripe session check failed")

	return ToolResult(
		data={
			"status": "unpaid",
			"date": booking.date,
			"time": booking.time,
			"fee_rupees": settings.BOOKING_FEE_RUPEES,
			"note": "Payment is not confirmed yet. Invite the user to complete it "
					"from the payment link.",
		},
		payload_type="response",
	)


def _confirmed(booking) -> ToolResult:
	data = {
		"status": "confirmed",
		"booking": {"id": booking.id, "date": booking.date,
					"time": booking.time, "name": booking.name},
	}
	return ToolResult(
		data=data,
		payload_type="booking_confirmed",
		extra={"booking": data["booking"]},
	)
