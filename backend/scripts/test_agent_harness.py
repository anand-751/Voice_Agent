"""Comprehensive verification test for Runtime Response and Tool Selection Agent Harness.

Tests:
1. ToolSelectionHarness:
   - Date & Time parsing and normalization
   - Doctor specialty auto-correlation (Orthodontics vs Endodontics)
   - State-enforced payment and cancellation routing
   - Automatic fallback when time is missing from slot reservation
2. ResponseHarness:
   - Anti-hallucination on free slots (detects fabricated times and injects verified slots)
   - Booking fee price verification & payment prompt enforcement
   - Acoustic voice conditioning (markdown stripping, acronym expansion, currency normalization)
   - Brevity control (max 3 spoken sentences)
3. End-to-end multi-turn integration test with Langfuse tracking
"""

import asyncio
import types
from datetime import date, timedelta
from app.config import get_settings
from app.agent.harness import ToolSelectionHarness, ResponseHarness
from app.agent.runner import run_turn
from app.agent.graph import build_graph
from app.tools.calendar_tool import select_slot, check_availability
from app.services.booking_store import BookingStore
from app.services.calendar_service import CalendarService
from app.services.llm import LLMService
from app.services.observability import ObservabilityService
from app.services.session import SessionStore
from app.services.vectorstore import VectorStore


def test_tool_selection_harness():
	print("\n--- 1. Testing Tool Selection Harness (Strict & Reliable Tool Routing) ---")
	settings = get_settings()
	app = types.SimpleNamespace(state=types.SimpleNamespace(settings=settings, observability=None))
	today = date.today()

	# A. Relative date & time normalization
	raw_decision = {"action": "calendar.select_slot", "args": {"date": "tomorrow", "time": "2pm"}}
	state = {"user_input": "Please book me tomorrow at 2pm"}
	harness_res = ToolSelectionHarness.validate_and_repair(raw_decision, state, app)

	expected_date = (today + timedelta(days=1)).isoformat()
	assert harness_res.action == "calendar.select_slot"
	assert harness_res.args["date"] == expected_date
	assert harness_res.args["time"] == "14:00"
	assert harness_res.was_repaired is True
	print(f"✓ Normalized 'tomorrow at 2pm' -> date={harness_res.args['date']}, time={harness_res.args['time']}")

	# A1. Dynamic Temporal Comprehension: 'day after today' vs 'day after tomorrow'
	date_day_after_today, _ = ToolSelectionHarness._normalize_date_string("", "Can I see the doctor day after today?", today)
	assert date_day_after_today == (today + timedelta(days=1)).isoformat(), f"Expected {(today + timedelta(days=1)).isoformat()}, got {date_day_after_today}"
	print(f"✓ Correctly resolved 'day after today' -> {date_day_after_today} (Tomorrow, +1 day)")

	date_day_after_tomorrow, _ = ToolSelectionHarness._normalize_date_string("", "Can I book day after tomorrow?", today)
	assert date_day_after_tomorrow == (today + timedelta(days=2)).isoformat()
	print(f"✓ Correctly resolved 'day after tomorrow' -> {date_day_after_tomorrow} (+2 days)")

	# A2. Dynamic Temporal Comprehension: 'coming saturday' and 'coming thursday'
	# Test with fixed anchor: Monday Sept 7, 2026
	anchor_monday = date(2026, 9, 7)
	sat_date, _ = ToolSelectionHarness._normalize_date_string("", "Is doctor available coming saturday?", anchor_monday)
	assert sat_date == "2026-09-12", f"Expected 2026-09-12, got {sat_date}"
	assert date.fromisoformat(sat_date).strftime("%A") == "Saturday"
	print(f"✓ Correctly resolved 'coming saturday' from Monday -> {sat_date} (Saturday)")

	thu_date, _ = ToolSelectionHarness._normalize_date_string("", "Is Dr. Rohit available coming thursday?", anchor_monday)
	assert thu_date == "2026-09-10", f"Expected 2026-09-10, got {thu_date}"
	assert date.fromisoformat(thu_date).strftime("%A") == "Thursday"
	print(f"✓ Correctly resolved 'coming thursday' from Monday -> {thu_date} (Thursday)")

	# A3. Override LLM Hallucinated Dates with Caller's Spoken Day
	# LLM mistakenly returned 2026-09-09 (Wednesday) when user asked for coming Saturday
	hallucinated_llm_decision = {"action": "calendar.check_availability", "args": {"date": "2026-09-09", "doctor": ""}}
	repaired_decision = ToolSelectionHarness.validate_and_repair(
		hallucinated_llm_decision, {"user_input": "Is doctor available coming Saturday?"}, app
	)
	# Check against current real today
	expected_sat, _ = ToolSelectionHarness._normalize_date_string("", "coming saturday", today)
	assert repaired_decision.args["date"] == expected_sat
	print(f"✓ Overrode LLM hallucinated Wednesday date with verified coming Saturday date: {repaired_decision.args['date']}")

	# A4. Auto-route doctor availability inquiry mistakenly routed to rag.query
	misrouted_decision = {"action": "rag.query", "args": {"question": "Are you open coming Saturday?"}}
	auto_routed = ToolSelectionHarness.validate_and_repair(
		misrouted_decision, {"user_input": "Are you open coming Saturday?"}, app
	)
	assert auto_routed.action == "calendar.check_availability"
	print(f"✓ Auto-routed misrouted availability question to '{auto_routed.action}' with date={auto_routed.args['date']}")

	# B. Doctor specialty auto-alignment for Orthodontics (braces/aligners)
	raw_decision = {"action": "calendar.select_slot", "args": {"date": "tomorrow", "time": "11:00", "doctor": ""}}
	state = {"user_input": "I want braces and clear aligners to fix my crooked teeth"}
	harness_res = ToolSelectionHarness.validate_and_repair(raw_decision, state, app)
	assert harness_res.args["doctor"] == "Dr. Rohit Verma"
	print(f"✓ Aligned caller symptom 'braces/crooked teeth' -> doctor='{harness_res.args['doctor']}'")

	# C. Doctor specialty auto-alignment for Endodontics (RCT/toothache)
	raw_decision = {"action": "calendar.check_availability", "args": {"date": "tomorrow", "doctor": ""}}
	state = {"user_input": "I have terrible tooth pain and need a root canal"}
	harness_res = ToolSelectionHarness.validate_and_repair(raw_decision, state, app)
	assert harness_res.args["doctor"] == "Dr. Ananya Sharma"
	print(f"✓ Aligned caller symptom 'root canal/tooth pain' -> doctor='{harness_res.args['doctor']}'")

	# D. State-enforced payment verification when pending booking exists
	state_with_pending = {
		"user_input": "I have completed the payment",
		"pending_booking_id": "bk_test_123",
	}
	raw_decision = {"action": "chitchat", "args": {}}  # Router mistakenly chose chitchat
	harness_res = ToolSelectionHarness.validate_and_repair(raw_decision, state_with_pending, app)
	assert harness_res.action == "payment.verify"
	print(f"✓ Enforced pending booking state -> redirected 'chitchat' to '{harness_res.action}'")

	# E. State-enforced booking cancellation
	state_cancel = {
		"user_input": "Please cancel my pending appointment",
		"pending_booking_id": "bk_test_123",
	}
	raw_decision = {"action": "rag.query", "args": {"question": "cancel"}}
	harness_res = ToolSelectionHarness.validate_and_repair(raw_decision, state_cancel, app)
	assert harness_res.action == "booking.cancel"
	print(f"✓ Enforced pending cancellation -> redirected to '{harness_res.action}'")

	# F. Degrade select_slot to check_availability when user has not specified a time
	raw_decision = {"action": "calendar.select_slot", "args": {"date": "tomorrow"}}
	state_no_time = {"user_input": "I want to book an appointment tomorrow"}
	harness_res = ToolSelectionHarness.validate_and_repair(raw_decision, state_no_time, app)
	assert harness_res.action == "calendar.check_availability"
	print(f"✓ Degraded missing-time select_slot to '{harness_res.action}'")

	# G. Degrade "book appointment for September 12 at Saturday" when LLM hallucinated 09:00 but user spoke no time
	raw_decision_hallucinated_time = {
		"action": "calendar.select_slot",
		"args": {"date": "2026-09-12", "time": "09:00", "doctor": "Dr. Ananya Sharma"},
	}
	state_no_spoken_time = {"user_input": "book appointment for September 12 at Saturday"}
	harness_res = ToolSelectionHarness.validate_and_repair(raw_decision_hallucinated_time, state_no_spoken_time, app)
	assert harness_res.action == "calendar.check_availability"
	print(f"✓ Correctly degraded hallucinated 9:00 AM slot to '{harness_res.action}' because caller spoke no time")

	# H. Degrade out-of-hours time (e.g. 9:00 AM when clinic opens at 10:00 AM)
	raw_decision_early = {
		"action": "calendar.select_slot",
		"args": {"date": "2026-09-08", "time": "09:00", "doctor": "Dr. Rohit Verma"},
	}
	state_early = {"user_input": "book tomorrow at 9 am"}
	harness_res = ToolSelectionHarness.validate_and_repair(raw_decision_early, state_early, app)
	assert harness_res.action == "calendar.check_availability"
	print(f"✓ Degraded 9:00 AM out-of-hours request to '{harness_res.action}'")

	# I. Preserve select_slot when caller speaks a valid time (e.g. 11:00 am)
	raw_decision_valid = {
		"action": "calendar.select_slot",
		"args": {"date": "2026-09-08", "time": "11:00", "doctor": ""},
	}
	state_valid = {"user_input": "yes 11:00 am is right"}
	harness_res = ToolSelectionHarness.validate_and_repair(raw_decision_valid, state_valid, app)
	assert harness_res.action == "calendar.select_slot"
	assert harness_res.args["time"] == "11:00"
	print(f"✓ Preserved valid spoken slot '11:00 AM' as '{harness_res.action}'")

	# J. Offset beyond 8 days ("after 8 days at 3 pm")
	raw_decision_after_8 = {
		"action": "calendar.select_slot",
		"args": {"time": "15:00"},
	}
	state_after_8 = {"user_input": "book slot after 8 days at 3 pm"}
	harness_res_8 = ToolSelectionHarness.validate_and_repair(raw_decision_after_8, state_after_8, app)
	assert harness_res_8.action == "calendar.select_slot"
	assert harness_res_8.args["time"] == "15:00"
	expected_date_after_8 = (today + timedelta(days=9)).isoformat()
	assert harness_res_8.args["date"] == expected_date_after_8
	print(f"✓ Correctly resolved 'after 8 days' -> {expected_date_after_8} as '{harness_res_8.action}'")

	# K. Offset in Hindi ("10 din baad")
	raw_decision_hindi = {
		"action": "calendar.check_availability",
		"args": {},
	}
	state_hindi = {"user_input": "mujhe 10 din baad slot chahiye"}
	harness_res_hi = ToolSelectionHarness.validate_and_repair(raw_decision_hindi, state_hindi, app)
	assert harness_res_hi.action == "calendar.check_availability"
	expected_date_hindi = (today + timedelta(days=10)).isoformat()
	assert harness_res_hi.args["date"] == expected_date_hindi
	print(f"✓ Correctly resolved '10 din baad' -> {expected_date_hindi} as '{harness_res_hi.action}'")

	# L. Choosing a doctor without specifying a time degrades select_slot to check_availability
	raw_decision_choose_doc = {
		"action": "calendar.select_slot",
		"args": {"date": "2026-09-08", "time": "10:00", "doctor": "Dr. Rohit Verma"},
	}
	state_choose_doc = {
		"user_input": "I want to see Dr. Rohit",
		"history": [{"role": "assistant", "content": "We have 10:00 AM available."}],
	}
	harness_res_cd = ToolSelectionHarness.validate_and_repair(raw_decision_choose_doc, state_choose_doc, app)
	assert harness_res_cd.action == "calendar.check_availability"
	assert harness_res_cd.args["doctor"] == "Dr. Rohit Verma"
	print(f"✓ Degraded doctor choice without user confirmation of time to '{harness_res_cd.action}'")


def test_response_harness_fact_checking():
	print("\n--- 2. Testing Response Harness Fact-Checking & Anti-Hallucination ---")
	settings = get_settings()
	app = types.SimpleNamespace(state=types.SimpleNamespace(settings=settings, observability=None))

	# A. Anti-hallucination on free slots
	# The LLM fabricated 3:00 PM and 5:00 PM, but real tool result only has 10:00, 11:30, 12:00
	hallucinated_reply = "We have open slots at 3:00 PM and 5:00 PM tomorrow!"
	state_slots = {
		"tool_result": {
			"free_slots": ["10:00", "11:30", "12:00"],
			"date": "2026-09-08",
			"doctor": "Dr. Rohit Verma",
		},
		"payload_type": "response",
	}
	harness_res = ResponseHarness.verify_and_condition(hallucinated_reply, state_slots, app)
	assert harness_res.was_repaired is True
	assert "10:00" in harness_res.response
	assert "3:00 PM" not in harness_res.response
	print(f"✓ Caught hallucinated slot '3:00 PM' -> injected real slots: '{harness_res.response}'")

	# A2. Offering at least 2 slots when LLM only offered a single slot
	single_slot_reply = "Doctor Rohit 10:00 AM par available hain."
	state_multi = {
		"tool_result": {
			"free_slots": ["10:00", "11:30", "14:00"],
			"date": "2026-09-08",
			"doctor": "Dr. Rohit Verma",
		},
		"payload_type": "response",
	}
	harness_res_multi = ResponseHarness.verify_and_condition(single_slot_reply, state_multi, app)
	assert harness_res_multi.was_repaired is True
	assert "10:00" in harness_res_multi.response
	assert "11:30" in harness_res_multi.response
	assert "Aapko kaunsa slot book kar doon?" in harness_res_multi.response
	assert "Doctor Doctor" not in harness_res_multi.response
	print(f"✓ Expanded single slot to at least 2 slot options: '{harness_res_multi.response}'")

	# B. Doctor unavailable schedule override
	false_availability_reply = "Great news! Dr. Rohit Verma is available on Monday to see you."
	state_unavail = {
		"tool_result": {
			"doctor_unavailable": True,
			"note": "Dr. Rohit Verma only visits on Tuesday, Thursday, and Saturday.",
		},
		"payload_type": "response",
	}
	harness_res = ResponseHarness.verify_and_condition(false_availability_reply, state_unavail, app)
	assert harness_res.was_repaired is True
	assert "only visits on Tuesday" in harness_res.response
	print(f"✓ Overrode false doctor availability claim -> '{harness_res.response}'")

	# C. Fee price verification
	hallucinated_fee_reply = "Your slot is reserved. Please pay the 250 rupees fee to confirm."
	state_payment = {"payload_type": "payment_required"}
	harness_res = ResponseHarness.verify_and_condition(hallucinated_fee_reply, state_payment, app)
	assert "500 rupees" in harness_res.response
	assert "250" not in harness_res.response
	assert "Open payment" in harness_res.response
	print(f"✓ Corrected fabricated fee 250 -> 500 rupees with payment button prompt: '{harness_res.response}'")

	# D. Sunday Clinic Closure Reality Check
	false_sunday_open = "Yes, we are open on Sunday and we have slots available at 10:00 AM."
	state_sunday = {
		"tool_result": {
			"available": False,
			"clinic_closed": True,
			"requested_date": "2026-09-13",
			"requested_day": "Sunday",
			"note": "Bright Dental Clinic is closed on Sundays (emergencies are handled by phone only). We are open Monday through Saturday from 10:00 AM to 7:00 PM. Would you like to book an appointment for Monday, 2026-09-14 instead?",
		},
		"payload_type": "response",
	}
	harness_res_sun = ResponseHarness.verify_and_condition(false_sunday_open, state_sunday, app)
	assert harness_res_sun.was_repaired is True
	assert "closed on Sundays" in harness_res_sun.response
	assert "slots available" not in harness_res_sun.response
	print(f"✓ Overrode false Sunday open claim with verified closure policy: '{harness_res_sun.response}'")

	# E. Spoken Date-Day Consistency Reconciliation
	# LLM generated 'Saturday, 10th September', but tool result date is 2026-09-12 (Saturday 12th)
	inconsistent_spoken_reply = "Yes, Dr. Rohit Verma is available this Saturday, 10th September. Which slot would you prefer?"
	state_date_recon = {
		"tool_result": {
			"date": "2026-09-12",
			"doctor": "Dr. Rohit Verma",
			"free_slots": ["10:00", "11:30"],
		}
	}
	harness_res_date = ResponseHarness.verify_and_condition(inconsistent_spoken_reply, state_date_recon, app)
	assert harness_res_date.was_repaired is True
	assert "Saturday, September 12" in harness_res_date.response
	print(f"✓ Reconciled spoken date error: '{inconsistent_spoken_reply}' -> '{harness_res_date.response}'")

	# F. Beyond 8-Day Advance Booking Window Reality Check
	false_beyond_booking = "Sure, your appointment is confirmed for September 25th at 3:00 PM!"
	state_beyond = {
		"tool_result": {
			"available": False,
			"beyond_booking_window": True,
			"requested_date": "2026-09-25",
			"note": "Bright Dental Clinic accepts advance bookings only within the coming week (in 8 days only). Please select a date within the next 8 days.",
		},
		"payload_type": "response",
	}
	harness_res_beyond = ResponseHarness.verify_and_condition(false_beyond_booking, state_beyond, app)
	assert harness_res_beyond.was_repaired is True
	assert any(w in harness_res_beyond.response.lower() for w in ("8 days", "8 din", "coming week", "hafte"))
	assert "confirmed" not in harness_res_beyond.response
	print(f"✓ Overrode false booking beyond 8 days with verified 8-day advance policy: '{harness_res_beyond.response}'")

	# F. Spoken URL Replacement
	url_reply = "Please complete your payment at https://checkout.stripe.com/pay/cs_test_abc123 to secure your slot."
	harness_res_url = ResponseHarness.verify_and_condition(url_reply, {}, app)
	assert "https://" not in harness_res_url.response
	assert "secure link on your screen" in harness_res_url.response
	print(f"✓ Filtered raw HTTP URL into voice-friendly prompt: '{harness_res_url.response}'")

	# G. Anti-hallucination for unpaid bookings (forbid false confirmation claims)
	false_confirmed_reply = "Your appointment is scheduled with Dr. Ananya Sharma at 11:00 AM on Saturday."
	state_unpaid = {
		"tool_result": {
			"status": "unpaid",
			"date": "Saturday, Sep 12",
			"time": "11:00 AM",
			"fee_rupees": 500,
		},
		"payload_type": "response",
	}
	harness_res_unpaid = ResponseHarness.verify_and_condition(false_confirmed_reply, state_unpaid, app)
	assert harness_res_unpaid.was_repaired is True
	assert "held pending payment" in harness_res_unpaid.response
	assert "not confirmed yet" in harness_res_unpaid.response
	assert "Open payment" in harness_res_unpaid.response
	print(f"✓ Overrode false confirmation for unpaid booking -> '{harness_res_unpaid.response}'")


def test_response_harness_acoustic_conditioning():
	print("\n--- 3. Testing Acoustic Voice Conditioning & Brevity ---")
	settings = get_settings()
	app = types.SimpleNamespace(state=types.SimpleNamespace(settings=settings, observability=None))

	# A. Stripping Markdown
	markdown_reply = "### Consultation Details:\n- **Dr. Ananya Sharma** is ready.\n- Please bring your *X-rays*."
	harness_res = ResponseHarness.verify_and_condition(markdown_reply, {}, app)
	assert "**" not in harness_res.response
	assert "###" not in harness_res.response
	assert "- " not in harness_res.response
	print(f"✓ Stripped markdown formatting -> '{harness_res.response}'")

	# B. Acronyms & Currency
	raw_speech = "Your RCT consultation is ₹500 and takes around 30 mins."
	harness_res = ResponseHarness.verify_and_condition(raw_speech, {}, app)
	assert "root canal treatment" in harness_res.response
	assert "500 rupees" in harness_res.response
	assert "30 minutes" in harness_res.response
	assert "₹" not in harness_res.response
	print(f"✓ Expanded acronyms & currency -> '{harness_res.response}'")

	# B2. Comma-separated currency formatting (e.g. ₹3,500 must not become 3 rupees,500)
	raw_comma_currency = "Root canal treatment costs between ₹3,500 and ₹6,000 per tooth."
	harness_res_commas = ResponseHarness.verify_and_condition(raw_comma_currency, {}, app)
	assert "3,500 rupees" in harness_res_commas.response
	assert "6,000 rupees" in harness_res_commas.response
	assert "rupees,500" not in harness_res_commas.response
	print(f"✓ Formatted comma currency correctly -> '{harness_res_commas.response}'")

	# C. 24h to 12h Time Conversion
	raw_time = "Your appointment is set for 14:00 with Dr. Rohit Verma."
	harness_res = ResponseHarness.verify_and_condition(raw_time, {}, app)
	assert "2:00 PM" in harness_res.response
	print(f"✓ Converted 14:00 -> '{harness_res.response}'")

	# D. Full Spoken Response Preservation & Doctor Phonetic Expansion
	full_reply = "Oh, aapko gums mein problem ho rahi hai, main samajh sakti hoon, yeh kafi uncomfortable hota hai! Dr. Ananya Sharma se checkup karwake aapko complete relief mil jayega. Kya main kal ka slot book kar doon?"
	harness_res = ResponseHarness.verify_and_condition(full_reply, {}, app)
	assert "Doctor Ananya Sharma" in harness_res.response
	assert "Kya main kal ka slot book kar doon?" in harness_res.response
	assert not harness_res.response.endswith("Dr.")
	print(f"✓ Preserved complete LLM response with phonetic Doctor expansion -> '{harness_res.response}'")

	# E. Removal of 'baje' word in responses
	baje_reply = "Aap kal subah 11:30 baje baje aa sakte hain, ya shaam 4:00 PM baje."
	harness_res_baje = ResponseHarness.verify_and_condition(baje_reply, {}, app)
	assert "baje" not in harness_res_baje.response.lower()
	assert "11:30" in harness_res_baje.response
	assert "4:00" in harness_res_baje.response
	print(f"✓ Successfully removed 'baje' word in response -> '{harness_res_baje.response}'")


def test_end_to_end_agent_with_harness():
	print("\n--- 4. Testing End-to-End Agent with Harness Engineering ---")
	settings = get_settings()
	obs = ObservabilityService(settings)
	app = types.SimpleNamespace(
		state=types.SimpleNamespace(
			settings=settings,
			observability=obs,
			llm=LLMService(observability=obs),
			bookings=BookingStore(":memory:"),
			vectors=VectorStore(),
		)
	)
	app.state.graph = build_graph(app)

	sessions = SessionStore()
	session = sessions.create()
	session.profile["name"] = "Rahul Verma"
	session.profile["phone"] = "9876543210"

	async def run_calls():
		# Turn 1: Caller asking about braces
		res1 = await run_turn(app, session, "I have crooked front teeth and need braces. Can I book tomorrow?")
		assert res1.get("response")
		assert "**" not in res1["response"]
		assert "₹" not in res1["response"]
		print(f"✓ Turn 1 response: '{res1['response']}'")

		# Turn 2: Farewell
		res2 = await run_turn(app, session, "Thank you, goodbye!")
		assert res2.get("type") == "call_end"
		print(f"✓ Turn 2 response: '{res2['response']}' (Type: {res2.get('type')})")

	asyncio.run(run_calls())
	obs.flush()
	print(f"✓ End-to-end turns completed with full harness enforcement & Langfuse telemetry")


def test_calendar_tool_enhancements():
	print("\n--- 4. Testing Calendar Tool Rescheduling & Clinic Hours ---")
	settings = get_settings()
	booking_store = BookingStore(":memory:")
	app = types.SimpleNamespace(
		state=types.SimpleNamespace(
			settings=settings,
			bookings=booking_store,
		)
	)

	tomorrow = date.today() + timedelta(days=1)
	if tomorrow.weekday() == 6:  # Skip Sunday
		tomorrow += timedelta(days=1)
	tomorrow_str = tomorrow.isoformat()

	# A. Out of hours rejection in select_slot (e.g. 08:00 AM)
	state1 = {"profile": {"name": "Test User", "phone": "9999999999"}}
	res_early = select_slot(app, state1, {"date": tomorrow_str, "time": "08:00", "doctor": "Dr. Ananya Sharma"})
	assert res_early.data.get("out_of_hours") is True
	print(f"✓ select_slot rejected 08:00 AM out of hours: '{res_early.data.get('note')}'")

	# B. Cancelling previous pending booking when rescheduling to a new slot
	free_slots = CalendarService(settings).get_free_slots(tomorrow_str)
	assert len(free_slots) >= 2, "Need at least 2 free slots for rescheduling test"
	slot1, slot2 = free_slots[0], free_slots[1]

	res1 = select_slot(app, state1, {"date": tomorrow_str, "time": slot1, "doctor": "Dr. Ananya Sharma"})
	booking1_id = res1.pending_booking_id
	assert booking1_id is not None
	b1 = booking_store.get(booking1_id)
	assert b1.status == "pending"
	print(f"✓ Initial slot reservation held for {slot1}: id={booking1_id}")

	# Now caller reschedules to second slot
	state1["pending_booking_id"] = booking1_id
	res2 = select_slot(app, state1, {"date": tomorrow_str, "time": slot2, "doctor": "Dr. Ananya Sharma"})
	booking2_id = res2.pending_booking_id
	assert booking2_id is not None
	assert booking2_id != booking1_id

	# Verify old booking was cancelled and new booking is pending
	b1_after = booking_store.get(booking1_id)
	b2 = booking_store.get(booking2_id)
	assert b1_after.status == "cancelled"
	assert b2.status == "pending"
	print(f"✓ Rescheduling cancelled old booking '{booking1_id}' and created new pending booking '{booking2_id}'")

	# C. Rejection of dates beyond 8-day advance booking window in check_availability and select_slot
	beyond_date = (date.today() + timedelta(days=10)).isoformat()
	res_beyond_avail = check_availability(app, state1, {"date": beyond_date})
	assert res_beyond_avail.data.get("beyond_booking_window") is True
	assert res_beyond_avail.data.get("available") is False
	assert any(w in res_beyond_avail.data.get("note", "").lower() for w in ("8 din", "8 days", "coming week", "hafte"))
	print(f"✓ check_availability correctly rejected date beyond 8 days ({beyond_date}): '{res_beyond_avail.data.get('note')}'")

	res_beyond_select = select_slot(app, state1, {"date": beyond_date, "time": "11:00", "doctor": "Dr. Ananya Sharma"})
	assert res_beyond_select.data.get("beyond_booking_window") is True
	assert res_beyond_select.data.get("available") is False
	assert res_beyond_select.payload_type == "response"
	assert any(w in res_beyond_select.data.get("note", "").lower() for w in ("8 din", "8 days", "coming week", "hafte"))
	print(f"✓ select_slot correctly rejected slot beyond 8 days ({beyond_date}): '{res_beyond_select.data.get('note')}'")


def main():
	print("=" * 60)
	print("  Runtime Response & Tool Selection Harness Test Suite")
	print("=" * 60)
	test_tool_selection_harness()
	test_response_harness_fact_checking()
	test_response_harness_acoustic_conditioning()
	test_calendar_tool_enhancements()
	test_end_to_end_agent_with_harness()
	print("\n All Harness Engineering tests passed successfully!")
	print("=" * 60)


if __name__ == "__main__":
	import sys
	main()
	sys.exit(0)
