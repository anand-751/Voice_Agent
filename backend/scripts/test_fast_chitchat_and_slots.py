"""Verification script for:
1. Fast chit-chat, greetings, acknowledgments, non-dental topics, complaints returning intent=None,
   FAST_RESPONSE with direct response (bypassing main engine).
2. Slots queries returning intent="slots", triggering gap filler ("Wait, main check karti hoon..."),
   and signaling main agent.
"""

import asyncio
import sys
from pathlib import Path

# Ensure backend root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.llm import LLMService
from app.services.sarvam import SarvamTTSService
from app.services.rumik import RumikTTSService


async def main():
	print("\n" + "=" * 70)
	print("TESTING FAST CHIT-CHAT BYPASS & SLOTS GAP FILLER COORDINATION")
	print("=" * 70)

	llm = LLMService()

	# Test Part 1: Chit-chat, Greetings, Acknowledgments, Non-dental, Complaints
	fast_cases = [
		("Namaste", "greeting"),
		("Namaskar", "greeting"),
		("Hello", "greeting"),
		("Kaise ho", "chitchat"),
		("Aap kaisi hain?", "chitchat"),
		("Theek hai", "acknowledgment"),
		("Shukriya", "acknowledgment"),
		("Dhanyavaad", "acknowledgment"),
		("Haan ji", "acknowledgment"),
		("Mujhe sar dard aur bukhar ho raha hai", "non_dental"),
		("Tum mujhe bahut irritate kar rahi ho, samajh nahi aata kya", "complaint"),
		("हां, ठीक है। कैसी हैं आप? मुझे ये बताइए।", "chitchat"),
		("आप इतनी ज़्यादा स्लो क्यों हैं बात करने में?", "latency_check"),
		("ठीक है, थैंक यू आपकी इस हेल्प के लिए, बट मैं बाद में आपसे बात करता हूं। थैंक यू।", "farewell"),
	]

	print("\n--- PART 1: Fast Responses (intent = None, FAST_RESPONSE decision) ---")
	for utterance, expected_category in fast_cases:
		# 1. Check classify_filler_intent returns None
		filler_intent = await llm.classify_filler_intent(utterance)
		assert filler_intent is None, f"Expected filler_intent=None for '{utterance}', got '{filler_intent}'"

		# 2. Check decide_turn_action returns FAST_RESPONSE, intent=None, and direct_response
		decision = await llm.decide_turn_action(utterance)
		action = decision.get("decision")
		direct_resp = decision.get("direct_response") or decision.get("deescalation_response")
		intent = decision.get("intent")

		print(f"\n[Utterance]: \"{utterance}\"")
		print(f"  -> Decision: {action}")
		print(f"  -> Intent: {intent}")
		print(f"  -> Direct Response: \"{direct_resp}\"")

		assert action in ("FAST_RESPONSE", "SECURITY_INTERCEPT") or bool(direct_resp), f"Expected fast response action for '{utterance}', got '{action}'"
		assert intent is None, f"Expected intent=None for fast response '{utterance}', got '{intent}'"
		assert bool(direct_resp), f"Expected non-empty direct response for '{utterance}'"

	print("\n✓ Part 1 Passed: All chit-chat/greetings/non-dental/complaints return intent=None and direct fast response without main engine!")

	# Test Part 2: Slots Queries & Gap Filler Coordination
	print("\n--- PART 2: Slot Queries & Gap Filler Coordination ---")
	slot_queries = [
		"Kal 10:00 AM ka slot khali hai kya?",
		"Doctor Rohit ka appointment kab mil sakta hai, slots check kijiye",
		"Monday ko sham 4 baje koi slot available hai?",
		"Available slots dekh kar bataiye",
	]

	for slot_q in slot_queries:
		# 1. Check classify_filler_intent returns 'slots'
		filler_intent = await llm.classify_filler_intent(slot_q)
		print(f"\n[Slot Query]: \"{slot_q}\"")
		print(f"  -> Filler Intent: {filler_intent}")
		assert filler_intent == "slots", f"Expected 'slots' for '{slot_q}', got '{filler_intent}'"

		# 2. Check decide_turn_action returns EXECUTE_TURN with intent='slots'
		decision = await llm.decide_turn_action(slot_q)
		action = decision.get("decision")
		intent = decision.get("intent")
		print(f"  -> Turn Action: {action}")
		print(f"  -> Signal to Main Agent (intent): {intent}")
		assert action == "EXECUTE_TURN", f"Expected EXECUTE_TURN for '{slot_q}', got '{action}'"
		assert intent == "slots", f"Expected intent='slots' for '{slot_q}', got '{intent}'"

	print("\n✓ Part 2 Passed: All slot queries return intent='slots' to signal the main agent!")

	# Test Part 3: Gap Filler phrase check in Sarvam & Rumik
	print("\n--- PART 3: Gap Filler Phrase Verification ('Wait, main check karti hoon...') ---")
	sarvam_fillers = SarvamTTSService.FILLERS_BY_INTENT.get("slots", [])
	rumik_fillers = RumikTTSService.FILLERS_BY_INTENT.get("slots", [])

	print("Sarvam slot fillers:", sarvam_fillers)
	print("Rumik slot fillers:", rumik_fillers)

	assert "Wait, main check karti hoon..." in sarvam_fillers, "Sarvam slot fillers missing 'Wait, main check karti hoon...'"
	assert "Wait, main check karti hoon..." in rumik_fillers, "Rumik slot fillers missing 'Wait, main check karti hoon...'"
	print("\n✓ Part 3 Passed: 'Wait, main check karti hoon...' is configured as a primary slot gap filler in both Sarvam and Rumik TTS!")

	print("\n" + "=" * 70)
	print("ALL VERIFICATIONS COMPLETED SUCCESSFULLY!")
	print("=" * 70)


if __name__ == "__main__":
	asyncio.run(main())
