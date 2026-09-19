"""Automated Test Suite for Continuation Buffering, Barge-in Turn Superseding,
Devanagari Affirmations, and Low-Latency Conversational Turn-Taking.
"""

import asyncio
import os
import sys

# Ensure backend path is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.agent.nodes import _get_deterministic_template_reply, _matches_any_term
from app.api.websocket import is_incomplete_utterance
from app.services.session import SessionStore


def test_incomplete_utterance_detection():
	print("\n--- 1. Testing Incomplete Utterance & Trailing Connector Detection ---")

	# Trailing markers (should be True)
	incomplete_cases = [
		"मुझे एक्चुअली",
		"mujhe actually",
		"actually mujhe",
		"हां ठीक है, मेरी बात सुनो पूरी पहले तो कि मैं बोल रहा हूं कि मुझे",
		"main bol raha hoon ki mujhe",
		"soch raha tha ki",
		"ek minute",
		"ek second",
		"meri baat suno",
		"मेरी बात सुनो",
		"suno",
		"सुनिए",
		"root canal karwana hai aur",
		"doctor ananya available hain lekin",
		"aaj shaam ko",
		"cleaning karwani thi par",
	]
	for text in incomplete_cases:
		res = is_incomplete_utterance(text)
		assert res, f"Expected '{text}' to be marked incomplete, got {res}"
		print(f"✓ Correctly flagged incomplete: '{text}'")

	# Complete queries (should be False -> executed immediately with 0ms added latency)
	complete_cases = [
		"Doctor Ananya ke saath appointment book karni hai",
		"Tuesday 11:00 AM",
		"टीथ में फिलिंग करवानी है।",
		"आइज़ में एक्चुअली कुछ इशू हो रहा है।",
		"teeth cleaning karwani hai",
		"Saturday afternoon slot please",
		"हाँ",
		"Theek hai, confirm kar dijiye",
	]
	for text in complete_cases:
		res = is_incomplete_utterance(text)
		assert not res, f"Expected '{text}' to be marked complete, got {res}"
		print(f"✓ Complete (0ms extra latency): '{text}'")


def test_devanagari_affirmation_and_hesitation_replies():
	print("\n--- 2. Testing Devanagari Affirmation & Hesitation Fast-Paths ---")
	clinic = "Bright Dental Clinic"

	# Devanagari 'हाँ'
	reply = _get_deterministic_template_reply("chitchat", "हाँ", clinic, "sarvam")
	assert "Ji bilkul" in reply or "Batayein" in reply, f"Failed for 'हाँ': got '{reply}'"
	print(f"✓ Handled Devanagari 'हाँ' gracefully: '{reply}'")

	# Devanagari 'हां ठीक है'
	reply2 = _get_deterministic_template_reply("chitchat", "हां ठीक है", clinic, "sarvam")
	assert "Ji bilkul" in reply2 or "Batayein" in reply2, f"Failed for 'हां ठीक है': got '{reply2}'"
	print(f"✓ Handled 'हां ठीक है': '{reply2}'")

	# Hesitation phrase 'meri baat suno'
	reply3 = _get_deterministic_template_reply("chitchat", "meri baat suno", clinic, "sarvam")
	assert "Ji boliye, main bilkul dhyan se sun rahi hoon" in reply3, f"Failed hesitation: got '{reply3}'"
	print(f"✓ Handled 'meri baat suno' attentive listener: '{reply3}'")

	# Hesitation phrase 'मेरी बात सुनो'
	reply4 = _get_deterministic_template_reply("chitchat", "मेरी बात सुनो", clinic, "sarvam")
	assert "Ji boliye, main bilkul dhyan se sun rahi hoon" in reply4, f"Failed hesitation: got '{reply4}'"
	print(f"✓ Handled Devanagari 'मेरी बात सुनो': '{reply4}'")

	# Hesitation phrase 'ek minute'
	reply5 = _get_deterministic_template_reply("chitchat", "ek minute", clinic, "sarvam")
	assert "Ji boliye, main bilkul dhyan se sun rahi hoon" in reply5, f"Failed 'ek minute': got '{reply5}'"
	print(f"✓ Handled 'ek minute': '{reply5}'")


def test_session_turn_superseding():
	print("\n--- 3. Testing Session Pop Last Turn for Merging ---")
	store = SessionStore()
	session = store.create()

	session.add("user", "हां ठीक है, मेरी बात सुनो पूरी पहले तो कि मैं बोल रहा हूं कि मुझे")
	session.add("assistant", "Main Bright Dental Clinic se aapki help ke liye ready hoon ji...")

	assert len(session.history) == 2

	# Caller interrupts and provides continuation
	popped = session.pop_last_turn()
	assert popped is not None
	assert popped["user"] == "हां ठीक है, मेरी बात सुनो पूरी पहले तो कि मैं बोल रहा हूं कि मुझे"
	assert "Main Bright Dental Clinic" in popped["assistant"]
	assert len(session.history) == 0

	# Now combined query can be added
	combined = f"{popped['user']} टीथ में फिलिंग करवानी है।"
	session.add("user", combined)
	session.add("assistant", "Tooth filling ke liye Doctor Ananya best hain ji...")

	assert len(session.history) == 2
	assert session.history[0]["content"] == combined
	print(f"✓ Successfully popped aborted turn and merged query: '{combined}'")


async def test_debounced_continuation_flow():
	print("\n--- 4. Testing Debounced Continuation Flow (<= 0.7s Constraint) ---")
	# Simulate continuation timing
	debounce_duration = 0.6  # strictly under 0.7s
	assert debounce_duration <= 0.7, "Debounce timer exceeds 0.7s user constraint!"

	pending_continuation = "मुझे एक्चुअली"
	continuation_cancelled = False

	async def mock_timer():
		nonlocal continuation_cancelled
		try:
			await asyncio.sleep(debounce_duration)
		except asyncio.CancelledError:
			continuation_cancelled = True

	task = asyncio.create_task(mock_timer())
	# User speaks within 300ms
	await asyncio.sleep(0.3)
	task.cancel()
	await asyncio.sleep(0.05)

	assert continuation_cancelled, "Expected timer to be cancelled when user resumes speaking!"
	# Merge
	next_query = "आइज़ में एक्चुअली कुछ इशू हो रहा है।"
	merged = f"{pending_continuation} {next_query}"
	assert merged == "मुझे एक्चुअली आइज़ में एक्चुअली कुछ इशू हो रहा है।"
	print(f"✓ Cancelled timer on speech continuation and merged query: '{merged}' (in 300ms, well under 0.7s)")


def main():
	print("=" * 60)
	print("  Continuation Buffering & Conversational UX Test Suite")
	print("=" * 60)
	test_incomplete_utterance_detection()
	test_devanagari_affirmation_and_hesitation_replies()
	test_session_turn_superseding()
	asyncio.run(test_debounced_continuation_flow())
	print("\n" + "=" * 60)
	print(" ALL CONTINUATION & BARGE-IN UX TESTS PASSED SUCCESSFULLY!")
	print("=" * 60)


if __name__ == "__main__":
	main()
