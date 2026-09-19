import logging

from datetime import date as dt_date, datetime as dt_datetime, timedelta as dt_timedelta
from zoneinfo import ZoneInfo

from .base import ToolResult
from ..services.booking_flow import finalize_paid_booking
from ..services.calendar_service import CalendarService
from ..services.doctor_service import (
	find_doctor_by_name,
	get_all_doctors,
	get_upcoming_available_dates,
	is_doctor_available_on_date,
	recommend_doctor_by_problem,
)
from ..services.stripe_service import StripeService


log = logging.getLogger("tools.calendar")


def _public_booking(booking) -> dict:
	cal_url = ""
	if booking.date:
		parts = booking.date.split("-")
		if len(parts) == 3:
			cal_url = f"https://calendar.google.com/calendar/r/day/{parts[0]}/{int(parts[1])}/{int(parts[2])}"
	return {
		"id": booking.id,
		"date": booking.date,
		"time": booking.time,
		"name": booking.name,
		"doctor": getattr(booking, "doctor", "") or "Dr. Ananya Sharma",
		"calendar_url": cal_url,
	}


def _resolve_doctor(state: dict, args: dict):
	"""Identify targeted or recommended doctor from args, profile, or user input."""
	raw_name = args.get("doctor") or (state.get("profile") or {}).get("doctor")
	if raw_name:
		doc = find_doctor_by_name(raw_name)
		if doc:
			return doc

	user_input = state.get("user_input", "")
	doc = find_doctor_by_name(user_input)
	if doc:
		return doc

	return recommend_doctor_by_problem(user_input)


def check_availability(app, state, args) -> ToolResult:
	settings = app.state.settings
	calendar = CalendarService(settings)
	date = args.get("date")
	if not date:
		return ToolResult(data={"error": "no date resolved"})

	try:
		date_obj = dt_date.fromisoformat(date)
		day_name = date_obj.strftime("%A")
	except Exception:
		date_obj = None
		day_name = ""

	tz = ZoneInfo(settings.CLINIC_TIMEZONE)
	today = dt_datetime.now(tz).date()
	max_date = today + dt_timedelta(days=8)
	is_sarvam = getattr(settings, "TTS_PROVIDER", "browser").lower() in ("sarvam", "rumik")

	# 1. Check 8-day advance booking window limit first
	if date_obj and date_obj > max_date:
		max_date_fmt = max_date.strftime("%A, %B %d")
		req_date_fmt = date_obj.strftime("%A, %B %d")
		return ToolResult(
			data={
				"available": False,
				"beyond_booking_window": True,
				"requested_date": date,
				"max_allowed_date": max_date.isoformat(),
				"note": (
					f"Hum advance bookings sirf aane wale ek hafte (agley 8 din, {max_date_fmt} tak) ke liye open rakhte hain ji. "
					f"Kripya 8 din ke andar ki date chunein, main turant slot check kar dungi!"
					if is_sarvam
					else (
						f"Bright Dental Clinic accepts advance bookings only within the coming week (up to 8 days in advance, through {max_date_fmt}). "
						f"We cannot check slots for {req_date_fmt} yet. Please select a date within the next 8 days."
					)
				),
			},
			payload_type="response",
		)

	# 2. Check Sunday clinic closure
	if day_name == "Sunday":
		next_monday = (date_obj + dt_timedelta(days=1)).isoformat() if date_obj else ""
		return ToolResult(
			data={
				"available": False,
				"clinic_closed": True,
				"requested_date": date,
				"requested_day": "Sunday",
				"next_open_date": next_monday,
				"note": (
					f"Bright Dental Clinic is closed on Sundays (emergencies are handled by phone only). "
					f"We are open Monday through Saturday from 10:00 AM to 7:00 PM. "
					f"Would you like to book an appointment for Monday, {next_monday} instead?"
				),
			},
			payload_type="response",
		)

	doc = _resolve_doctor(state, args)

	# 3. If a specific doctor is targeted, check if they practice on this day
	if doc:
		is_avail, _ = is_doctor_available_on_date(doc, date)
		if not is_avail:
			upcoming = get_upcoming_available_dates(doc, date, max_dates=3)
			return ToolResult(
				data={
					"available": False,
					"doctor_unavailable": True,
					"doctor": doc.name,
					"doctor_specialty": doc.specialty,
					"requested_date": date,
					"requested_day": day_name,
					"working_days": doc.working_days_str,
					"upcoming_available_dates": upcoming,
					"note": (
						f"{doc.name} ({doc.specialty}) only visits on {doc.working_days_str}. "
						f"They are NOT at the clinic on {day_name}s ({date}). "
						f"Please inform the user and suggest booking on their available dates: "
						f"{', '.join(d['formatted'] for d in upcoming)}."
					),
				},
				payload_type="response",
			)

	# 4. Check past date
	if date_obj and date_obj < today:
		return ToolResult(
			data={
				"available": False,
				"past_date": True,
				"requested_date": date,
				"note": (
					f"Yeh date ({date}) nikal chuki hai ji. Kripya aaj ya aane wale dino mein se date chunein."
					if is_sarvam
					else f"Cannot check: {date} is in the past. Please select today or an upcoming date within the next 8 days."
				),
			},
			payload_type="response",
		)

	slots = calendar.get_free_slots(date)
	is_sarvam = getattr(settings, "TTS_PROVIDER", "browser").lower() in ("sarvam", "rumik")
	max_slots = 4 if is_sarvam else 8
	data = {
		"date": date,
		"requested_day": day_name,
		"free_slots": slots[:max_slots],
		"all_free_slots": slots,
		"total_free": len(slots),
		"slot_minutes": settings.SLOT_MINUTES,
		"booking_fee_rupees": settings.BOOKING_FEE_RUPEES,
		"payment_required": settings.PAYMENT_REQUIRED_FOR_BOOKING,
	}

	# Check if checked date is today and clinic hours are over / no slots remaining
	if date_obj == today and len(slots) == 0:
		next_cand = today + dt_timedelta(days=1)
		while next_cand.weekday() == 6:  # Skip Sunday
			next_cand += dt_timedelta(days=1)
		if doc and next_cand.strftime("%A") not in doc.working_days:
			upcoming = get_upcoming_available_dates(doc, today.isoformat(), max_dates=2)
			if upcoming:
				next_cand = dt_date.fromisoformat(upcoming[0]["date"])
		next_open_str = next_cand.isoformat()
		next_open_slots = calendar.get_free_slots(next_open_str)
		data["today_fully_booked_or_closed"] = True
		data["next_open_date"] = next_open_str
		data["next_open_day"] = next_cand.strftime("%A")
		data["next_open_slots"] = next_open_slots[:max_slots]
		data["note"] = (
			f"Aaj clinic ka samay samapt ho chuka hai ji. Next open day {next_cand.strftime('%A')} ({next_open_str}) "
			f"mein available slots hain: {', '.join(next_open_slots[:3])}. Kya main {next_cand.strftime('%A')} ka slot check ya book kar doon?"
			if is_sarvam
			else (
				f"Clinic operating hours for today have concluded. Next open clinic day is {next_cand.strftime('%A')}, {next_open_str} "
				f"with open slots: {', '.join(next_open_slots[:3])}. Would you like to book for {next_cand.strftime('%A')}?"
			)
		)

	if doc:
		data["doctor"] = doc.name
		data["doctor_available"] = True
		data["doctor_specialty"] = doc.specialty
		data["doctor_days"] = doc.working_days_str
	else:
		available_docs = [d.name for d in get_all_doctors() if day_name in d.working_days]
		data["available_doctors"] = available_docs
		rec = recommend_doctor_by_problem(state.get("user_input", ""))
		if rec:
			data["recommended_doctor"] = rec.name
			data["doctor_specialty"] = rec.specialty
			data["doctor_days"] = rec.working_days_str

	return ToolResult(data=data)


def select_slot(app, state, args) -> ToolResult:
	"""Reserve the slot (pending) and open payment — the NESTED payment step.

	Validates doctor schedule before reserving.
	"""
	settings = app.state.settings
	calendar = CalendarService(settings)
	is_sarvam = getattr(settings, "TTS_PROVIDER", "browser").lower() in ("sarvam", "rumik")
	date, time_ = args.get("date"), args.get("time")
	if not date or not time_:
		return ToolResult(data={"error": "missing date or time"})

	try:
		date_obj = dt_date.fromisoformat(date)
		day_name = date_obj.strftime("%A")
	except Exception:
		date_obj = None
		day_name = ""

	tz = ZoneInfo(settings.CLINIC_TIMEZONE)
	today = dt_datetime.now(tz).date()
	max_date = today + dt_timedelta(days=8)

	# 1. Check 8-day advance booking window limit first
	if date_obj and date_obj > max_date:
		max_date_fmt = max_date.strftime("%A, %B %d")
		req_date_fmt = date_obj.strftime("%A, %B %d")
		return ToolResult(
			data={
				"available": False,
				"beyond_booking_window": True,
				"requested_date": date,
				"requested_time": time_,
				"max_allowed_date": max_date.isoformat(),
				"note": (
					f"Hum advance bookings sirf aane wale ek hafte (agley 8 din, {max_date_fmt} tak) ke liye hi karte hain ji. "
					f"Kripya aane wale 8 din ke andar ki date chunein, main turant aapka appointment book kar dungi!"
					if is_sarvam
					else (
						f"Cannot book: Bright Dental Clinic accepts bookings only within the coming week (up to 8 days in advance, through {max_date_fmt}). "
						f"{req_date_fmt} is beyond our advance booking window. Please select a date within the next 8 days."
					)
				),
			},
			payload_type="response",
		)

	# 2. Check Sunday closure
	if day_name == "Sunday":
		next_monday = (date_obj + dt_timedelta(days=1)).isoformat() if date_obj else ""
		return ToolResult(
			data={
				"available": False,
				"clinic_closed": True,
				"requested_date": date,
				"requested_day": "Sunday",
				"next_open_date": next_monday,
				"note": (
					f"Cannot book: Bright Dental Clinic is closed on Sundays (emergencies by phone only). "
					f"Our clinic hours are Monday through Saturday, 10:00 AM to 7:00 PM. "
					f"Would you like to book for Monday, {next_monday} instead?"
				),
			},
			payload_type="response",
		)

	doc = _resolve_doctor(state, args)
	if not doc:
		doc = find_doctor_by_name("Dr. Ananya Sharma")

	# 3. Check doctor schedule — DO NOT book if doctor is unavailable on this day
	if doc:
		is_avail, _ = is_doctor_available_on_date(doc, date)
		if not is_avail:
			upcoming = get_upcoming_available_dates(doc, date, max_dates=3)
			return ToolResult(
				data={
					"available": False,
					"doctor_unavailable": True,
					"doctor": doc.name,
					"doctor_specialty": doc.specialty,
					"requested_date": date,
					"requested_day": day_name,
					"working_days": doc.working_days_str,
					"upcoming_available_dates": upcoming,
					"note": (
						f"Cannot book: {doc.name} ({doc.specialty}) is only available on {doc.working_days_str}. "
						f"They do not practice on {day_name}s ({date}). "
						f"Please inform the user and recommend their available dates: "
						f"{', '.join(d['formatted'] for d in upcoming)}."
					),
				},
				payload_type="response",
			)

	# 4. Check past date
	if date_obj and date_obj < today:
		return ToolResult(
			data={
				"available": False,
				"past_date": True,
				"requested_date": date,
				"requested_time": time_,
				"note": (
					f"Yeh date ({date}) nikal chuki hai ji. Kripya aaj ya aane wale dino mein se slot chunein."
					if is_sarvam
					else f"Cannot book: {date} has already passed. Please select a date today or in the coming week."
				),
			},
			payload_type="response",
		)

	# Check clinic hours (CLINIC_OPEN_HOUR to CLINIC_CLOSE_HOUR)
	try:
		hr = int(time_.split(":")[0])
		if hr < settings.CLINIC_OPEN_HOUR or hr >= settings.CLINIC_CLOSE_HOUR:
			open_fmt = f"{settings.CLINIC_OPEN_HOUR}:00 AM"
			close_fmt = f"{settings.CLINIC_CLOSE_HOUR % 12 or 12}:00 PM"
			return ToolResult(
				data={
					"available": False,
					"out_of_hours": True,
					"requested_time": time_,
					"date": date,
					"clinic_hours": f"{open_fmt} to {close_fmt}",
					"note": (
						f"Cannot book: {time_} is outside Bright Dental Clinic operating hours ({open_fmt} to {close_fmt}). "
						f"Please choose a slot during our clinic hours."
					),
					"alternatives": calendar.get_free_slots(date)[:3 if is_sarvam else 5],
				},
				payload_type="response",
			)
	except Exception:
		pass

	# Check past time or less than 1.5 hours notice from current Indian time
	try:
		tz = ZoneInfo(settings.CLINIC_TIMEZONE)
		now_ist = dt_datetime.now(tz)
		min_slot_time = now_ist + dt_timedelta(minutes=90)
		req_dt = dt_datetime.fromisoformat(f"{date}T{time_}").replace(tzinfo=tz)
		if req_dt < min_slot_time:
			is_past = req_dt <= now_ist
			time_fmt = req_dt.strftime("%I:%M %p").lstrip("0")
			now_fmt = now_ist.strftime("%I:%M %p").lstrip("0")
			min_fmt = min_slot_time.strftime("%I:%M %p").lstrip("0")
			avail_slots = calendar.get_free_slots(date)[:3 if is_sarvam else 5]

			if is_past:
				reason = f"{time_fmt} on {date} has already passed (current Indian time is {now_fmt})."
			else:
				reason = f"appointments require at least 1.5 hours advance notice (current Indian time is {now_fmt})."

			return ToolResult(
				data={
					"available": False,
					"slot_in_past_or_too_soon": True,
					"is_past": is_past,
					"requested_date": date,
					"requested_time": time_,
					"current_time": now_fmt,
					"earliest_allowed_time": min_fmt,
					"alternatives": avail_slots,
					"note": (
						f"Cannot book: {reason} "
						f"The earliest available slot today is from {min_fmt} onwards. "
						f"Available slots: {', '.join(avail_slots) if avail_slots else 'none remaining today (please check tomorrow)'}."
					),
				},
				payload_type="response",
			)
	except Exception:
		pass

	free_slots_for_date = calendar.get_free_slots(date)
	start_iso, end_iso = calendar.slot_to_iso(date, time_)
	if time_ not in free_slots_for_date or not calendar.is_slot_free(start_iso, end_iso):
		return ToolResult(data={
			"available": False,
			"invalid_slot": time_ not in free_slots_for_date,
			"date": date,
			"time": time_,
			"doctor": doc.name if doc else "",
			"alternatives": free_slots_for_date[:3 if is_sarvam else 5],
			"note": (
				f"'{time_}' slot available nahi hai ji. Available slots hain: {', '.join(free_slots_for_date[:3])}."
				if is_sarvam
				else f"The requested slot '{time_}' is not available for {date}. Available slots are: {', '.join(free_slots_for_date[:3])}."
			),
		})

	# If the caller had an earlier unpaid pending booking, cancel it before creating the new one
	old_pending_id = state.get("pending_booking_id")
	if old_pending_id:
		try:
			app.state.bookings.cancel(old_pending_id)
		except Exception:
			pass

	profile = dict(state.get("profile") or {})
	doc_name = doc.name if doc else "Dr. Ananya Sharma"
	profile["doctor"] = doc_name

	caller_name = (profile.get("name") or "").strip()
	caller_phone = (profile.get("phone") or "").strip()
	caller_email = (profile.get("email") or "").strip()

	# Fallback: if caller profile name is missing or "Guest", lookup latest saved user profile in DB
	if (not caller_name or caller_name.lower() == "guest") and hasattr(app.state, "bookings") and hasattr(app.state.bookings, "get_latest_profile"):
		try:
			latest_db_prof = app.state.bookings.get_latest_profile()
			if latest_db_prof and latest_db_prof.get("name"):
				caller_name = latest_db_prof["name"].strip()
				if not caller_phone and latest_db_prof.get("phone"):
					caller_phone = latest_db_prof["phone"].strip()
				if not caller_email and latest_db_prof.get("email"):
					caller_email = latest_db_prof["email"].strip()
				profile["name"] = caller_name
				if caller_phone:
					profile["phone"] = caller_phone
				if caller_email:
					profile["email"] = caller_email
		except Exception:
			pass

	booking = app.state.bookings.create(
		name=caller_name or "Guest",
		phone=caller_phone or "",
		email=caller_email or "",
		date=date,
		time=time_,
		start_iso=start_iso,
		end_iso=end_iso,
		doctor=doc_name,
	)

	# ── nested payment step ────────────────────────────────────────────
	if not settings.PAYMENT_REQUIRED_FOR_BOOKING:
		booking = finalize_paid_booking(app, booking.id)
		return ToolResult(
			data={"reserved": True, "confirmed": True,
				  "doctor": doc_name,
				  "booking": _public_booking(booking)},
			payload_type="booking_confirmed",
			extra={"booking": _public_booking(booking)},
		)

	checkout = StripeService(settings).create_checkout(booking)
	app.state.bookings.attach_session(booking.id, checkout.session_id)

	return ToolResult(
		data={
			"reserved": True,
			"date": date,
			"time": time_,
			"doctor": doc_name,
			"fee_rupees": settings.BOOKING_FEE_RUPEES,
			"note": f"Slot with {doc_name} held as pending. Confirms only after payment is verified.",
		},
		payload_type="payment_required",
		extra={"checkout_url": checkout.url, "booking": _public_booking(booking)},
		pending_booking_id=booking.id,
	)
