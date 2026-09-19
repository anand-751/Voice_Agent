"""Test suite verifying:
1. Compliment detection ('बाय द वे निया, आपकी आवाज बहुत अच्छी है', 'यू आर वेरी ब्यूटीफुल') -> fast_category='compliment', NOT farewell.
2. Call cut commands ('कट द कॉल', 'यू कैन नाउ कट द कॉल', 'i am done with this') -> decision='CALL_END'.
3. Post-booking gratitude ('ओके थैंक यू आई एम डन विथ दिस', 'थैंक्स फॉर द हेल्प थैंक यू') -> decision='CALL_END'.
4. Pre-booking gratitude ('ठीक है शुक्रिया') -> stays active (acknowledgment).
5. ToolHarness auto-routing to end_call on post-booking gratitude or cut commands.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add backend root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings
from app.services.llm import LLMService, get_fast_chit_chat_response
from app.agent.harness import ToolSelectionHarness
from app.agent.nodes import _get_deterministic_template_reply

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("test_booking_farewell_cut")


async def main():
	print("=" * 70)
	print("TEST SUITE: Call Cut, Compliments, & Post-Booking Farewell Handling")
	print("=" * 70)

	settings = get_settings()
	llm = LLMService()

	# 1. Compliment vs Farewell ("बाय द वे...")
	print("\n--- 1. Testing Compliments (Should NOT trigger Farewell / Call End) ---")
	comp1 = "बाय द वे निया, आपकी आवाज बहुत अच्छी है।"
	res1 = await llm.decide_turn_action(comp1, provider="sarvam", has_pending_booking=True)
	print(f"Query: '{comp1}'")
	print(f"  Decision: {res1.get('decision')}, Fast Category: {res1.get('fast_category')}, Farewell: {res1.get('is_farewell')}")
	print(f"  Direct Response: {res1.get('direct_response')}")
	assert res1.get("decision") == "FAST_RESPONSE", f"Expected FAST_RESPONSE but got {res1.get('decision')}"
	assert res1.get("fast_category") == "compliment", f"Expected compliment but got {res1.get('fast_category')}"
	assert not res1.get("is_farewell"), "Compliment must NOT be flagged as farewell!"
	print("✓ Compliment 'बाय द वे...' recognized accurately without false farewell!")

	comp2 = "यू आर वेरी ब्यूटीफुल नियो।"
	res2 = await llm.decide_turn_action(comp2, provider="sarvam", has_pending_booking=False)
	print(f"\nQuery: '{comp2}'")
	print(f"  Decision: {res2.get('decision')}, Fast Category: {res2.get('fast_category')}")
	assert res2.get("fast_category") == "compliment"
	assert not res2.get("is_farewell")
	print("✓ Compliment 'यू आर वेरी ब्यूटीफुल' recognized accurately!")

	# 2. Call cut command in Hindi
	print("\n--- 2. Testing Direct Call Cut Commands ---")
	cut1 = "नहीं, कोई और सवाल नहीं है। ओके, यू कैन नाउ कट द कॉल।"
	res_cut1 = await llm.decide_turn_action(cut1, provider="sarvam", has_pending_booking=True)
	print(f"Query: '{cut1}'")
	print(f"  Decision: {res_cut1.get('decision')}, Fast Category: {res_cut1.get('fast_category')}, Farewell: {res_cut1.get('is_farewell')}")
	print(f"  Direct Response: {res_cut1.get('direct_response')}")
	assert res_cut1.get("decision") == "CALL_END", f"Expected CALL_END but got {res_cut1.get('decision')}"
	assert res_cut1.get("is_farewell") is True
	assert "slot" in res_cut1.get("direct_response").lower() or "hold" in res_cut1.get("direct_response").lower()
	print("✓ Explicit 'कट द कॉल' triggers CALL_END with slot-aware closing!")

	cut2 = "Ok you can now cut the call, bye"
	res_cut2 = await llm.decide_turn_action(cut2, provider="sarvam", has_pending_booking=False)
	print(f"\nQuery: '{cut2}'")
	print(f"  Decision: {res_cut2.get('decision')}, Farewell: {res_cut2.get('is_farewell')}")
	assert res_cut2.get("decision") == "CALL_END"
	assert res_cut2.get("is_farewell") is True
	print("✓ English 'cut the call' triggers CALL_END!")

	# 3. Post-booking 'I am done with this' / Gratitude
	print("\n--- 3. Testing Post-Booking Completion (has_pending_booking=True) ---")
	done1 = "ओके, थैंक यू, आई एम डन विथ दिस।"
	res_done1 = await llm.decide_turn_action(done1, provider="sarvam", has_pending_booking=True)
	print(f"Query: '{done1}'")
	print(f"  Decision: {res_done1.get('decision')}, Fast Category: {res_done1.get('fast_category')}, Farewell: {res_done1.get('is_farewell')}")
	print(f"  Direct Response: {res_done1.get('direct_response')}")
	assert res_done1.get("decision") == "CALL_END", f"Expected CALL_END but got {res_done1.get('decision')}"
	assert res_done1.get("is_farewell") is True
	assert "appointment" in res_done1.get("direct_response").lower() or "slot" in res_done1.get("direct_response").lower()
	print("✓ 'आई एम डन विथ दिस' triggers CALL_END with booking hold reminder!")

	grat1 = "थैंक्स फॉर द हेल्प थैंक यू"
	res_grat1 = await llm.decide_turn_action(grat1, provider="sarvam", has_pending_booking=True)
	print(f"\nQuery: '{grat1}' (with pending booking)")
	print(f"  Decision: {res_grat1.get('decision')}, Farewell: {res_grat1.get('is_farewell')}")
	assert res_grat1.get("decision") == "CALL_END"
	assert res_grat1.get("is_farewell") is True
	print("✓ Post-booking gratitude 'थैंक्स फॉर द हेल्प थैंक यू' triggers CALL_END!")

	# 4. Pre-booking gratitude vs Post-booking gratitude
	print("\n--- 4. Pre-Booking vs Post-Booking Gratitude Contrast ---")
	pre_grat = "ठीक है शुक्रिया"
	res_pre = await llm.decide_turn_action(pre_grat, provider="sarvam", has_pending_booking=False)
	print(f"Pre-booking: '{pre_grat}'")
	print(f"  Decision: {res_pre.get('decision')}, Category: {res_pre.get('fast_category')}, Farewell: {res_pre.get('is_farewell')}")
	assert res_pre.get("decision") == "FAST_RESPONSE"
	assert res_pre.get("fast_category") == "acknowledgment"
	assert not res_pre.get("is_farewell"), "Pre-booking 'theek hai shukriya' should keep listening!"
	print("✓ Pre-booking 'ठीक है शुक्रिया' stays in call (acknowledgment)!")

	res_post = await llm.decide_turn_action(pre_grat, provider="sarvam", has_pending_booking=True)
	print(f"\nPost-booking: '{pre_grat}'")
	print(f"  Decision: {res_post.get('decision')}, Category: {res_post.get('fast_category')}, Farewell: {res_post.get('is_farewell')}")
	assert res_post.get("decision") == "CALL_END"
	assert res_post.get("is_farewell") is True
	print("✓ Post-booking 'ठीक है शुक्रिया' triggers CALL_END!")

	# 5. ToolHarness enforcement
	print("\n--- 5. ToolHarness Post-Booking End-Call Routing ---")
	state_pending = {
		"pending_booking_id": "bk_test_active",
		"history": [],
		"user_input": "thank you for the help bye",
	}
	import types
	mock_app = types.SimpleNamespace(state=types.SimpleNamespace(settings=settings))
	harness_res = ToolSelectionHarness.validate_and_repair(
		decision={"action": "chitchat", "args": {}},
		state=state_pending,
		app=mock_app,
	)
	print(f"ToolHarness decision: {harness_res}")
	assert harness_res.action == "end_call", f"Expected end_call but got {harness_res.action}"
	print("✓ ToolHarness auto-routes post-booking farewell to 'end_call'!")

	reply = _get_deterministic_template_reply(
		action="end_call",
		user_input="thank you for the help bye",
		clinic_name="Bright Dental Clinic",
		provider="sarvam",
		has_pending_booking=True,
	)
	print(f"Deterministic parting reply: '{reply}'")
	assert "slot" in reply.lower() or "hold" in reply.lower() or "payment" in reply.lower()
	print("✓ Template reply reminds caller about payment link on screen!")

	print("\n" + "=" * 70)
	print("ALL VERIFICATION TESTS PASSED SUCCESSFULLY!")
	print("=" * 70)

	await llm.aclose()


if __name__ == "__main__":
	asyncio.run(main())
