import logging

from fastapi import APIRouter, HTTPException, Request

from ..services.booking_flow import finalize_paid_booking
from ..services.stripe_service import StripeService


log = logging.getLogger("webhooks")
router = APIRouter()


@router.post("/webhooks/stripe")
async def stripe_webhook(request: Request):
	"""Stripe webhook with signature verification and idempotency."""
	app = request.app
	payload = await request.body()
	signature = request.headers.get("stripe-signature", "")

	try:
		event = StripeService(app.state.settings).construct_event(payload, signature)
	except Exception as exc:
		log.warning("webhook signature/payload rejected: %s", exc)
		raise HTTPException(status_code=400, detail=str(exc))

	event_id = event.get("id", "")
	event_type = event.get("type", "")

	if event_id and app.state.bookings.event_processed(event_id):
		return {"received": True, "duplicate": True}
	if event_id:
		app.state.bookings.record_event(event_id)

	if event_type == "checkout.session.completed":
		checkout_session = event["data"]["object"]
		booking_id = (checkout_session.get("metadata") or {}).get("booking_id")
		if booking_id:
			finalize_paid_booking(app, booking_id, checkout_session.get("id", ""))
			log.info("booking %s paid via webhook", booking_id)

	return {"received": True}


@router.post("/api/dev/simulate-payment/{booking_id}")
async def simulate_payment(request: Request, booking_id: str):
	"""DEV ONLY: simulate the Stripe webhook locally."""
	app = request.app
	if not app.state.settings.DEV_MOCK_EXTERNALS:
		raise HTTPException(status_code=404, detail="not found")
	booking = finalize_paid_booking(app, booking_id)
	if booking is None:
		raise HTTPException(status_code=404, detail="booking not found")
	return {"ok": True,
			"status": booking.status,
			"calendar_event_id": booking.calendar_event_id}
