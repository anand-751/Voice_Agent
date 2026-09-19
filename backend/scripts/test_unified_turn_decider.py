"""Verification test suite for the Unified First-Layer Turn Classifier."""

import asyncio
import sys
from pathlib import Path

# Ensure backend root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.llm import LLMService


async def main():
	llm = LLMService()

	print("\n=== Testing Unified First-Layer Turn Classifier (Groq + Context) ===")

	history = [
		{"role": "assistant", "content": "Namaste! Bright Dental Clinic mein aapka swagat hai. Main Niaa bol rahi hoon. Main aapki kaise madad kar sakti hoon?"}
	]

	# Test 1: Incomplete thought / hesitation starter
	res1 = await llm.decide_turn_action(
		incoming_text="देखो मुझे basically na",
		conversation_history=history,
		is_agent_speaking=False,
		pending_continuation="",
	)
	print("\nTest 1 (Hesitation starter):")
	print("Input: 'देखो मुझे basically na'")
	print("Result:", res1)
	assert res1["decision"] == "WAIT_AND_LISTEN", f"Expected WAIT_AND_LISTEN, got {res1['decision']}"
	assert res1.get("backchannel") is not None, "Expected backchannel phrase"

	# Test 2: Continuation completing the thought
	res2 = await llm.decide_turn_action(
		incoming_text="ब्लीडिंग हो रही है मेरे टीथ में, तो क्या इलाज हो सकता है?",
		conversation_history=history,
		is_agent_speaking=False,
		pending_continuation="देखो मुझे basically na",
	)
	print("\nTest 2 (Continuation completion):")
	print("Input: 'ब्लीडिंग हो रही है मेरे टीथ में, तो क्या इलाज हो सकता है?' with prefix 'देखो मुझे basically na'")
	print("Result:", res2)
	assert res2["decision"] == "EXECUTE_TURN", f"Expected EXECUTE_TURN, got {res2['decision']}"
	assert "ब्लीडिंग" in res2["clean_query"] and "basically" in res2["clean_query"], "Expected queries to be clubbed"
	assert res2.get("intent") == "info", f"Expected intent 'info', got {res2.get('intent')}"

	# Test 3: Stray noise while agent speaking
	res3 = await llm.decide_turn_action(
		incoming_text="अह अह।",
		conversation_history=history,
		is_agent_speaking=True,
		pending_continuation="",
	)
	print("\nTest 3 (Stray noise while agent speaking):")
	print("Input: 'अह अह।' (speaking=True)")
	print("Result:", res3)
	assert res3["decision"] == "DROP_NOISE", f"Expected DROP_NOISE, got {res3['decision']}"

	# Test 4: Trailing connector
	res4 = await llm.decide_turn_action(
		incoming_text="aur Doctor Rohit se",
		conversation_history=history,
		is_agent_speaking=False,
		pending_continuation="",
	)
	print("\nTest 4 (Trailing connector 'se'):")
	print("Input: 'aur Doctor Rohit se'")
	print("Result:", res4)
	assert res4["decision"] in ("WAIT_AND_LISTEN", "EXECUTE_TURN")

	# Test 5: Direct complete slot selection
	res5 = await llm.decide_turn_action(
		incoming_text="Kal Monday subah 10:30 baje ka slot book kar dijiye",
		conversation_history=history,
		is_agent_speaking=False,
		pending_continuation="",
	)
	print("\nTest 5 (Slot selection):")
	print("Input: 'Kal Monday subah 10:30 baje ka slot book kar dijiye'")
	print("Result:", res5)
	assert res5["decision"] == "EXECUTE_TURN", f"Expected EXECUTE_TURN, got {res5['decision']}"
	assert res5.get("intent") == "slots", f"Expected intent 'slots', got {res5.get('intent')}"

	# Test 6: Fast fallback rules (Groq disabled simulation)
	print("\n--- Testing Instant Fallback Mode (0ms Heuristics) ---")
	res_fb1 = llm._fast_turn_decide("dekho mujhe na", is_agent_speaking=False)
	assert res_fb1["decision"] == "WAIT_AND_LISTEN"

	res_fb2 = llm._fast_turn_decide("daant mein dard hai", pending_continuation="dekho mujhe na")
	assert res_fb2["decision"] == "EXECUTE_TURN"
	assert "dekho mujhe na daant mein dard hai" == res_fb2["clean_query"]
	assert res_fb2["intent"] == "info"

	res_fb3 = llm._fast_turn_decide("हम्म", is_agent_speaking=True)
	assert res_fb3["decision"] == "DROP_NOISE"

	print("✓ All tests passed successfully!")


if __name__ == "__main__":
	asyncio.run(main())
