import json
import logging
from dataclasses import dataclass


log = logging.getLogger("stripe")

try:
	import stripe
except ImportError:  # pragma: no cover
	stripe = None


@dataclass
class Checkout:
	url: str
	session_id: str


class StripeService:
	"""Stripe Checkout and webhook verification with mock mode."""

	def __init__(self, settings):
		self.s = settings
		if settings.stripe_configured and stripe is not None:
			stripe.api_key = settings.STRIPE_SECRET_KEY

	@property
	def live(self) -> bool:
		return bool(self.s.stripe_configured and stripe is not None)

	def create_checkout(self, booking) -> Checkout:
		if not self.live:
			if not self.s.DEV_MOCK_EXTERNALS:
				raise RuntimeError("Stripe is not configured")
			return Checkout(
				url=(f"{self.s.FRONTEND_URL}/payment-success"
					 f"?mock=1&booking={booking.id}"),
				session_id=f"cs_mock_{booking.id}",
			)
		session = stripe.checkout.Session.create(
			mode="payment",
			line_items=[{
				"price_data": {
					"currency": self.s.STRIPE_CURRENCY,
					"unit_amount": self.s.BOOKING_FEE_RUPEES * 100,
					"product_data": {
						"name": f"{self.s.CLINIC_NAME} — appointment",
						"description": f"{booking.date} at {booking.time}",
					},
				},
				"quantity": 1,
			}],
			metadata={"booking_id": booking.id},
			customer_email=booking.email or None,
			success_url=f"{self.s.FRONTEND_URL}/payment-success",
			cancel_url=f"{self.s.FRONTEND_URL}/payment-cancel",
		)
		return Checkout(url=session.url, session_id=session.id)

	def construct_event(self, payload: bytes, signature: str):
		if self.s.STRIPE_WEBHOOK_SECRET and self.live:
			return stripe.Webhook.construct_event(
				payload, signature, self.s.STRIPE_WEBHOOK_SECRET)
		return json.loads(payload.decode("utf-8"))

	def session_paid(self, session_id: str) -> bool:
		if session_id.startswith("cs_mock_"):
			return False
		if not self.live:
			return False
		session = stripe.checkout.Session.retrieve(session_id)
		return session.payment_status == "paid"
