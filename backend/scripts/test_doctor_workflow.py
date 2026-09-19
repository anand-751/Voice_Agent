"""Verification test for Doctor recommendations, day-of-week scheduling, and calendar attribution."""

import asyncio
import types
from datetime import date, timedelta
from app.config import get_settings
from app.services.doctor_service import (
	find_doctor_by_name,
	get_upcoming_available_dates,
	is_doctor_available_on_date,
	recommend_doctor_by_problem,
)
from app.services.booking_store import BookingStore
from app.services.booking_flow import finalize_paid_booking
from app.services.calendar_service import CalendarService
from app.tools.calendar_tool import check_availability, select_slot


def test_doctor_workflow():
	settings = get_settings()
	print(f"Clinic: {settings.CLINIC_NAME}")

	# 1. Test Doctor Specialty Problem Recommendation
	print("\n--- 1. Testing Doctor Recommendations by Problem ---")
	ortho_queries = [
		"I have crooked teeth and want to align them",
		"How much are clear aligners and braces?",
		"I have gaps between my front teeth",
	]
	for q in ortho_queries:
		doc = recommend_doctor_by_problem(q)
		assert doc is not None, f"Failed to recommend doctor for: {q}"
		assert doc.name == "Dr. Rohit Verma", f"Expected Dr. Rohit Verma, got {doc.name}"
		print(f"✓ '{q}' -> Recommended: {doc.name} ({doc.specialty})")

	endo_queries = [
		"I have severe tooth pain and swelling",
		"Do you do root canal treatment?",
		"I need a dental filling and teeth cleaning",
		"I want in-clinic teeth whitening",
	]
	for q in endo_queries:
		doc = recommend_doctor_by_problem(q)
		assert doc is not None, f"Failed to recommend doctor for: {q}"
		assert doc.name == "Dr. Ananya Sharma", f"Expected Dr. Ananya Sharma, got {doc.name}"
		print(f"✓ '{q}' -> Recommended: {doc.name} ({doc.specialty})")

	# 2. Test Doctor Schedule & Day-of-Week Validation
	print("\n--- 2. Testing Doctor Schedule Validation ---")
	rohit = find_doctor_by_name("Dr. Rohit Verma")
	assert rohit is not None

	# Monday Sept 7, 2026 -> Should NOT be available for Dr. Rohit
	avail_mon, day_mon = is_doctor_available_on_date(rohit, "2026-09-07")
	assert day_mon == "Monday"
	assert not avail_mon, "Dr. Rohit Verma should NOT be available on Monday"
	print(f"✓ Dr. Rohit Verma on {day_mon} (2026-09-07): Available = {avail_mon} (Correctly blocked)")

	# Tuesday Sept 8, 2026 -> Should be available for Dr. Rohit
	avail_tue, day_tue = is_doctor_available_on_date(rohit, "2026-09-08")
	assert day_tue == "Tuesday"
	assert avail_tue, "Dr. Rohit Verma SHOULD be available on Tuesday"
	print(f"✓ Dr. Rohit Verma on {day_tue} (2026-09-08): Available = {avail_tue} (Correctly allowed)")

	# Upcoming dates for Dr. Rohit from Monday Sept 7
	upcoming = get_upcoming_available_dates(rohit, "2026-09-07", max_dates=3)
	assert len(upcoming) == 3
	assert upcoming[0]["day"] == "Tuesday" and upcoming[0]["date"] == "2026-09-08"
	assert upcoming[1]["day"] == "Thursday" and upcoming[1]["date"] == "2026-09-10"
	assert upcoming[2]["day"] == "Saturday" and upcoming[2]["date"] == "2026-09-12"
	print(f"✓ Nearest available dates from Monday: {[d['formatted'] for d in upcoming]}")

	# 3. Test Tool-level Rejection on Unavailable Day
	print("\n--- 3. Testing Tool Rejection when booking Dr. Rohit on Monday ---")
	mock_store = BookingStore(":memory:")
	mock_app = types.SimpleNamespace(
		state=types.SimpleNamespace(settings=settings, bookings=mock_store)
	)

	# User tries checking availability for Dr. Rohit on Monday
	res_avail = check_availability(
		mock_app,
		{"user_input": "Is Dr. Rohit available on Monday?"},
		{"date": "2026-09-07", "doctor": "Dr. Rohit Verma"}
	)
	assert res_avail.data.get("doctor_unavailable") is True
	print(f"✓ check_availability correctly rejected Monday: {res_avail.data.get('note')}")

	# User tries to force slot selection for Dr. Rohit on Monday
	res_select = select_slot(
		mock_app,
		{"user_input": "Book me with Dr. Rohit Verma on Monday at 11:00", "profile": {"name": "Anand"}},
		{"date": "2026-09-07", "time": "11:00", "doctor": "Dr. Rohit Verma"}
	)
	assert res_select.data.get("doctor_unavailable") is True
	assert res_select.payload_type == "response"  # Payment NOT triggered!
	print(f"✓ select_slot correctly refused to reserve Monday slot: {res_select.data.get('note')}")

	# 4. Test Successful Booking with Doctor Attribution
	print("\n--- 4. Testing Successful Booking with Doctor Attribution on Tuesday ---")
	cal = CalendarService(settings)
	available_slots = cal.get_free_slots("2026-09-15")
	test_slot = available_slots[0] if available_slots else "11:30"
	res_valid = select_slot(
		mock_app,
		{"user_input": f"Book with Dr. Rohit Verma on Tuesday at {test_slot}", "profile": {"name": "Anand Choudhary", "phone": "9876543210"}},
		{"date": "2026-09-15", "time": test_slot, "doctor": "Dr. Rohit Verma"}
	)
	assert res_valid.data.get("reserved") is True
	assert res_valid.data.get("doctor") == "Dr. Rohit Verma"
	booking_id = res_valid.pending_booking_id
	assert booking_id is not None
	print(f"✓ Slot reserved with {res_valid.data.get('doctor')} on Tuesday 2026-09-15 at {test_slot}. Booking ID: {booking_id}")

	# Verify booking in database has doctor field
	b = mock_store.get(booking_id)
	assert b.doctor == "Dr. Rohit Verma"
	print(f"✓ Booking saved in store with doctor: '{b.doctor}'")

	# 5. Test Sunday Clinic Closure
	print("\n--- 5. Testing Sunday Clinic Closure ---")
	res_sunday_avail = check_availability(
		mock_app,
		{"user_input": "Do you have slots this coming Sunday?"},
		{"date": "2026-09-13"}
	)
	assert res_sunday_avail.data.get("clinic_closed") is True
	assert res_sunday_avail.data.get("available") is False
	assert "closed on Sundays" in res_sunday_avail.data.get("note", "")
	assert res_sunday_avail.data.get("next_open_date") == "2026-09-14"
	print(f"✓ Sunday availability check correctly rejected with closure notice: {res_sunday_avail.data.get('note')}")

	res_sunday_select = select_slot(
		mock_app,
		{"user_input": "Book me on Sunday at 11:00", "profile": {"name": "Anand"}},
		{"date": "2026-09-13", "time": "11:00"}
	)
	assert res_sunday_select.data.get("clinic_closed") is True
	assert res_sunday_select.payload_type == "response"
	print(f"✓ Sunday slot reservation correctly refused: {res_sunday_select.data.get('note')}")

	print("\n All Doctor recommendation, scheduling, and attribution tests passed successfully!")


if __name__ == "__main__":
	test_doctor_workflow()
