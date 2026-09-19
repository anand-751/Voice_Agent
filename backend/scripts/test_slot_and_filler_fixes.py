"""Targeted verification for slot alignment, anti-hallucination grounding, and gap filler gating."""

import sys
import types
from datetime import date
from pathlib import Path

# Ensure backend root on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.llm import LLMService
from app.agent.guardrails import is_angry_or_frustrated
from app.agent.harness import ToolSelectionHarness, ResponseHarness
from app.config import get_settings
from app.tools.calendar_tool import check_availability, select_slot
from app.services.calendar_service import CalendarService


def run_tests():
	settings = get_settings()
	llm = LLMService()
	app = types.SimpleNamespace(
		state=types.SimpleNamespace(
			settings=settings,
			observability=None,
			calendar=CalendarService(settings),
			bookings=types.SimpleNamespace(
				create=lambda **kw: types.SimpleNamespace(id="mock_b1", **kw),
				cancel=lambda id: None,
			),
		)
	)

	print("\n=== 1. Testing Intent Classification for Non-Dental Symptoms & Questions ===")
	# Non-dental symptoms must have intent=None so NO delay filler audio is sent
	res_back = llm._classify_intent_sync("मेरे कमर में भी दर्द है।")
	print(f"\"मेरे कमर में भी दर्द है।\" -> {res_back} (expected: None)")
	assert res_back is None, f"Expected None for back pain, got {res_back}"

	res_stomach = llm._classify_intent_sync("पेट में बहुत तेज दर्द हो रहा है")
	print(f"\"पेट में बहुत तेज दर्द हो रहा है\" -> {res_stomach} (expected: None)")
	assert res_stomach is None

	# Dental symptom must still have intent='info'
	res_jaw = llm._classify_intent_sync("मेरे जबड़े में दर्द है।")
	print(f"\"मेरे जबड़े में दर्द है।\" -> {res_jaw} (expected: info)")
	assert res_jaw == "info"

	# Reprimand/confusion must have intent=None
	res_why = llm._classify_intent_sync("ते वो क्यों कर रहे हो?")
	print(f"\"ते वो क्यों कर रहे हो?\" -> {res_why} (expected: None)")
	assert res_why is None

	# Pre-question query must have intent='short'
	res_query = llm._classify_intent_sync("Ek sawaal poochna tha")
	print(f"\"Ek sawaal poochna tha\" -> {res_query} (expected: short)")
	assert res_query == "short"

	print("\n=== 2. Testing Anger/Reprimand Detection for 'ते वो क्यों कर रहे हो?' ===")
	is_angry = is_angry_or_frustrated("ते वो क्यों कर रहे हो?")
	print(f"\"ते वो क्यों कर रहे हो?\" is_angry: {is_angry}")
	assert is_angry is True

	is_angry_aisa = is_angry_or_frustrated("ऐसा क्यों कर रहे हो")
	print(f"\"ऐसा क्यों कर रहे हो\" is_angry: {is_angry_aisa}")
	assert is_angry_aisa is True

	print("\n=== 3. Testing ToolSelectionHarness Slot Alignment (11:45 vs 11:00) ===")
	raw_unaligned = {"action": "calendar.select_slot", "args": {"date": "2026-09-14", "time": "11:45"}}
	state_unaligned = {"user_input": "11:45", "today": "2026-09-13"}
	harness_res1 = ToolSelectionHarness.validate_and_repair(raw_unaligned, state_unaligned, app)
	print(f"11:45 select_slot -> action: {harness_res1.action}, issues: {harness_res1.issues}")
	assert harness_res1.action == "calendar.check_availability", f"Expected degraded to check_availability, got {harness_res1.action}"
	assert any("not an aligned 30-minute" in issue for issue in harness_res1.issues)

	raw_aligned = {"action": "calendar.select_slot", "args": {"date": "2026-09-14", "time": "11:00"}}
	state_aligned = {"user_input": "11:00 AM", "today": "2026-09-13"}
	harness_res2 = ToolSelectionHarness.validate_and_repair(raw_aligned, state_aligned, app)
	print(f"11:00 select_slot -> action: {harness_res2.action}")
	assert harness_res2.action == "calendar.select_slot"

	print("\n=== 4. Testing calendar_tool.select_slot with Unaligned Time ===")
	res_tool = select_slot(app, {}, {"date": "2026-09-14", "time": "11:45", "doctor": "Dr. Ananya Sharma"})
	print(f"select_slot 11:45 result: available={res_tool.data.get('available')}, invalid={res_tool.data.get('invalid_slot')}")
	assert res_tool.data.get("available") is False
	assert res_tool.data.get("invalid_slot") is True
	assert len(res_tool.data.get("alternatives", [])) > 0

	print("\n=== 5. Testing check_availability all_free_slots Provider ===")
	avail_res = check_availability(app, {}, {"date": "2026-09-14", "doctor": "Dr. Ananya Sharma"})
	all_slots = avail_res.data.get("all_free_slots", [])
	free_top = avail_res.data.get("free_slots", [])
	print(f"free_slots: {free_top}")
	print(f"all_free_slots count: {len(all_slots)} (includes '14:00': {'14:00' in all_slots})")
	assert len(all_slots) >= len(free_top)
	assert "14:00" in all_slots

	print("\n=== 6. Testing ResponseHarness Non-Hallucination for Valid Afternoon Slot (14:00) ===")
	state_with_all = {
		"tool_result": {
			"date": "2026-09-14",
			"doctor": "Dr. Ananya Sharma",
			"free_slots": ["10:30", "11:00", "11:30"],
			"all_free_slots": ["10:30", "11:00", "11:30", "12:00", "14:00", "15:00"],
		},
		"payload_type": "response",
	}
	llm_reply = "Humare paas kal subah 10:30, 11:00 ya dopahar 2:00 (14:00) ka slot available hai."
	resp_res = ResponseHarness.verify_and_condition(llm_reply, state_with_all, app)
	print(f"Response with 14:00 was_repaired: {resp_res.was_repaired}, issues: {resp_res.issues}")
	assert resp_res.was_repaired is False
	assert "2:00" in resp_res.response

	# Genuine hallucination (e.g. 5:00 PM / 17:00, not in all_free_slots) must still be caught
	fake_reply = "Humare paas kal shaam 5:00 PM ka slot available hai."
	resp_fake = ResponseHarness.verify_and_condition(fake_reply, state_with_all, app)
	print(f"Fake slot 5:00 PM was_repaired: {resp_fake.was_repaired}, issues: {resp_fake.issues}")
	assert resp_fake.was_repaired is True
	assert any("Hallucinated slots detected" in i for i in resp_fake.issues)

	print("\n✓ ALL SLOT AND FILLER FIXES VERIFIED SUCCESSFULLY!\n")


if __name__ == "__main__":
	run_tests()
