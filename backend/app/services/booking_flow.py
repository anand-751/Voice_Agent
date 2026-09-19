import logging

from .calendar_service import CalendarService


log = logging.getLogger("booking_flow")


def finalize_paid_booking(app, booking_id: str, stripe_session_id: str = ""):
	"""Mark paid, create the calendar event, and confirm idempotently."""
	store = app.state.bookings
	booking = store.get(booking_id)
	if booking is None:
		log.warning("finalize: booking %s not found", booking_id)
		return None

	if booking.status == "confirmed" and booking.calendar_event_id:
		return booking

	if booking.status != "paid":
		store.mark_paid(booking_id, stripe_session_id)

	if not booking.calendar_event_id:
		doc_name = getattr(booking, "doctor", "") or "Dr. Ananya Sharma"
		summary = f"Dental appointment with {doc_name} — {booking.name}"
		event = CalendarService(app.state.settings).create_event(
			summary=summary,
			description=(f"Doctor: {doc_name}\n"
						 f"Patient: {booking.name}\n"
						 f"Phone: {booking.phone}\n"
						 f"Email: {booking.email}\n"
						 f"Booking ID: {booking.id}"),
			start_iso=booking.start_iso,
			end_iso=booking.end_iso,
			attendee_email=booking.email or None,
		)
		store.mark_confirmed(booking_id, event.get("id", ""))
		log.info("booking %s confirmed with calendar event %s (%s)",
				 booking_id, event.get("id"), doc_name)

	return store.get(booking_id)
