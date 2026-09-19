"""Verification test script for Google Calendar integration."""

from datetime import date, timedelta
from app.config import get_settings
from app.services.calendar_service import CalendarService
from app.tools.calendar_tool import check_availability, select_slot


def test_calendar():
	settings = get_settings()
	print(f"Clinic: {settings.CLINIC_NAME}")
	print(f"Timezone: {settings.CLINIC_TIMEZONE}")
	print(f"Google Configured: {settings.google_configured}")

	calendar = CalendarService(settings)
	tomorrow = (date.today() + timedelta(days=1)).isoformat()

	# 1. Test Free Slots Query
	print(f"\n--- 1. Testing Free Slots for Tomorrow ({tomorrow}) ---")
	free_slots = calendar.get_free_slots(tomorrow)
	print(f"✓ Found {len(free_slots)} available slots: {free_slots[:6]}")
	assert len(free_slots) > 0, "Expected at least 1 free slot"

	# 2. Test Slot To ISO Conversion
	print("\n--- 2. Testing Slot ISO Conversion ---")
	start_iso, end_iso = calendar.slot_to_iso(tomorrow, "16:00")
	print(f"✓ 16:00 ISO: start={start_iso}, end={end_iso}")
	assert "16:00:00" in start_iso
	assert "16:30:00" in end_iso

	# 3. Test Slot Availability Check
	print("\n--- 3. Testing Slot Availability Check ---")
	is_free = calendar.is_slot_free(start_iso, end_iso)
	print(f"✓ Slot 16:00 is free: {is_free}")

	# 4. Test Event Creation
	print("\n--- 4. Testing Event Creation ---")
	event = calendar.create_event(
		summary=f"Test Dental Appointment — Unit Test",
		description="Patient: Unit Test\nPhone: 9876543210\nReason: Routine Checkup",
		start_iso=start_iso,
		end_iso=end_iso,
		attendee_email="patient@example.com",
	)
	print(f"✓ Event result: ID={event.get('id')}, link={event.get('htmlLink')}")
	assert "id" in event

	# 5. Test LLM Tools: check_availability & select_slot
	print("\n--- 5. Testing LLM Tools Integration ---")
	import types
	mock_app = types.SimpleNamespace(state=types.SimpleNamespace(settings=settings))

	avail_res = check_availability(mock_app, {}, {"date": tomorrow})
	print(f"✓ LLM check_availability tool output: total_free={avail_res.data.get('total_free')}, slots={avail_res.data.get('free_slots')[:4]}")
	assert avail_res.data.get("total_free", 0) > 0

	print("\n All Google Calendar service & LLM tool tests passed successfully!")


if __name__ == "__main__":
	test_calendar()
