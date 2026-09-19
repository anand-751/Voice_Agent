"""Test Suite: Problem-based Doctor Recommendation & Days Availability in Same Response.

Verifies that when a caller describes a problem:
1. The appropriate specialist doctor is identified.
2. Their working/visiting days are provided.
3. The spoken response smoothly incorporates both the doctor recommendation and availability days in the same reply.
"""

import asyncio
import os
import sys
import types

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.config import get_settings
from app.services.doctor_service import (
	recommend_doctor_by_problem,
	get_doctor_recommendation_context,
	DOCTORS,
)
from app.tools.rag import rag_answer
from app.agent.runner import run_turn
from app.services.booking_store import BookingStore
from app.services.vectorstore import VectorStore
from app.services.llm import LLMService
from app.agent.graph import build_graph


async def main():
	settings = get_settings()
	print("=" * 70)
	print("TEST SUITE: PROBLEM-BASED DOCTOR RECOMMENDATION & DAYS AVAILABILITY")
	print("=" * 70)

	# 1. Test Problem Keyword Matching
	print("\n--- 1. Testing Problem Matcher (Hindi, Hinglish, & English) ---")
	cases = [
		("mujhe na mere teeth mein kuch issue aa raha hai, cavity aur bleeding ho rahi hai", "Dr. Ananya Sharma", "Monday se Saturday"),
		("Mere daant mein bohot dard hai aur masudo se khoon aa raha hai", "Dr. Ananya Sharma", "Monday se Saturday"),
		("I need a root canal treatment done for my molar tooth", "Dr. Ananya Sharma", "Monday se Saturday"),
		("Teeth cleaning aur scaling ke charges kya hain", "Dr. Ananya Sharma", "Monday se Saturday"),
		("Mere daant thode tedhe hain mujhe braces lagwane hain", "Dr. Rohit Verma", "Tuesday, Thursday aur Saturday"),
		("I have crooked teeth and gaps, want to inquire about aligners", "Dr. Rohit Verma", "Tuesday, Thursday aur Saturday"),
		("Invisalign clear aligners lagte hain kya clinic mein?", "Dr. Rohit Verma", "Tuesday, Thursday aur Saturday"),
	]

	for query, expected_doc, expected_days_sub in cases:
		doc = recommend_doctor_by_problem(query)
		assert doc is not None, f"Failed to match doctor for: {query}"
		assert doc.name == expected_doc, f"Expected {expected_doc}, got {doc.name} for query: {query}"
		ctx = get_doctor_recommendation_context(query)
		assert ctx is not None
		assert ctx["recommended_doctor"] == expected_doc
		assert expected_days_sub.split()[0] in ctx["doctor_available_days_hindi"]
		print(f"✓ '{query[:55]}...' -> {doc.name} ({ctx['doctor_available_days_hindi']})")

	# 2. Test rag_answer tool enrichment
	print("\n--- 2. Testing rag_answer Context & Doctor Enrichment ---")
	mock_vectors = VectorStore()
	mock_app = types.SimpleNamespace(
		state=types.SimpleNamespace(
			settings=settings,
			vectors=mock_vectors,
		)
	)

	rag_query = "Mujhe bleeding gums aur cavity ki problem hai"
	res = rag_answer(mock_app, {"user_input": rag_query}, {"question": rag_query})
	assert res.data.get("recommended_doctor") == "Dr. Ananya Sharma"
	assert "Monday" in res.data.get("doctor_available_days_hindi")
	print(f"✓ rag_answer data contains recommended_doctor: {res.data.get('recommended_doctor')}")
	print(f"✓ rag_answer data contains days: {res.data.get('doctor_available_days_hindi')}")

	# 3. Test End-to-End Agent Response (Niaa Spoken Reply)
	print("\n--- 3. Testing End-to-End Agent Spoken Responses ---")
	store = BookingStore(":memory:")
	llm = LLMService()
	full_app = types.SimpleNamespace(
		state=types.SimpleNamespace(
			settings=settings,
			bookings=store,
			vectors=mock_vectors,
			llm=llm,
		)
	)
	full_app.state.graph = build_graph(full_app)

	test_turns = [
		(
			"मुझे ना मेरे टीथ में कुछ इशू आ रहा है, कैविटी और ब्लीडिंग हो रही है, तो उसकी हेल्प चाहिए थी।",
			("ananya", "अनन्या"),
			("monday", "सोमवार", "saturday", "शनिवार"),
		),
		(
			"Mere teeth crooked hain aur mujhe braces lagwane hain, iska kya process hai?",
			("rohit", "रोहित"),
			("tuesday", "thursday", "saturday", "मंगलवार", "गुरुवार", "शनिवार"),
		),
	]

	from app.services.session import Session

	for user_msg, expected_doc_tokens, expected_day_tokens in test_turns:
		print(f"\n[User Query]: {user_msg}")
		sess = Session(id="test_doc_rec", profile={"name": "Anand", "phone": "9876543210"})
		sess.tts_provider = "sarvam"
		result = await run_turn(full_app, sess, user_input=user_msg, caller_sentiment="neutral")
		reply = result.get("response", "")
		print(f"[Niaa Reply]: {reply}")

		# Check 1: Doctor recommendation present
		has_doc = any(tok.lower() in reply.lower() for tok in expected_doc_tokens)
		assert has_doc, f"Expected doctor (any of {expected_doc_tokens}) in response: {reply}"
		# Check 2: Availability days present in the SAME response
		has_days = any(tok.lower() in reply.lower() for tok in expected_day_tokens)
		assert has_days, f"Expected availability days (any of {expected_day_tokens}) in response: {reply}"
		print(f"✓ Both doctor recommendation and availability days present in same reply!")

	print("\n" + "=" * 70)
	print("ALL PROBLEM-BASED DOCTOR RECOMMENDATION TESTS PASSED 100%!")
	print("=" * 70)


if __name__ == "__main__":
	asyncio.run(main())
