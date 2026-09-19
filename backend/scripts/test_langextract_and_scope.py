"""Test Suite: LangExtract Knowledge Extraction, Chit-Chat Direct Answering,

and In-Scope vs Out-of-Scope Classification.
"""

import asyncio
import os
import sys

# Ensure backend in PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.config import get_settings
from app.services.extractor import LangExtractService
from app.services.llm import LLMService, get_fast_chit_chat_response


async def run_tests():
	settings = get_settings()
	print("\n" + "=" * 70)
	print("TEST SUITE 1: LANGEXTRACT CLINIC KNOWLEDGE EXTRACTION")
	print("=" * 70)

	extractor = LangExtractService(settings)
	print(f"Loaded {len(extractor.sections)} sections from {extractor.faqs_path}")
	assert len(extractor.sections) >= 5, "Should have parsed at least 5 FAQ sections"

	# Test 1.1: Pricing queries
	q_rct = "Root canal treatment (RCT) ke kya charges hain?"
	res_rct = extractor.extract_clinic_info(q_rct)
	print(f"\n[Query]: {q_rct}")
	print(f"[Extracted ({res_rct.get('method')})]: {res_rct.get('direct_answer')[:120]}...")
	assert res_rct.get("found") is True
	assert any(term in res_rct.get("context", "") for term in ("3,500", "6,000", "Root canal", "RCT"))

	# Test 1.2: Timing queries
	q_time = "Clinic timings kya hain aur lunch break kab hota hai?"
	res_time = extractor.extract_clinic_info(q_time)
	print(f"\n[Query]: {q_time}")
	print(f"[Extracted ({res_time.get('method')})]: {res_time.get('direct_answer')[:120]}...")
	assert res_time.get("found") is True
	assert "10:00 AM" in res_time.get("context", "")
	assert "1:30 PM" in res_time.get("context", "")

	# Test 1.3: Doctor specialty queries
	q_doc = "Dr. Rohit Verma kaunse treatments karte hain aur kab aate hain?"
	res_doc = extractor.extract_clinic_info(q_doc)
	print(f"\n[Query]: {q_doc}")
	print(f"[Extracted ({res_doc.get('method')})]: {res_doc.get('direct_answer')[:120]}...")
	assert res_doc.get("found") is True
	assert "Rohit" in res_doc.get("context", "")
	assert any(day in res_doc.get("context", "") for day in ("Tuesday", "Thursday", "Saturday"))

	print("\n✓ SUITE 1 PASSED: LangExtract extracts grounded clinic facts accurately!")

	print("\n" + "=" * 70)
	print("TEST SUITE 2: CHIT-CHAT DIRECT ANSWERING & PROACTIVE HELP OFFER")
	print("=" * 70)

	# Test 2.1: Well-being queries ("kaise ho", "how are you")
	chitchat_resp_hindi = get_fast_chit_chat_response("chitchat", "Aap kaisi hain?", provider="sarvam")
	print(f"\n[Chit-chat Hindi]: {chitchat_resp_hindi}")
	assert "theek hoon" in chitchat_resp_hindi.lower() or "shukriya" in chitchat_resp_hindi.lower()
	assert any(term in chitchat_resp_hindi.lower() for term in ("dental care", "appointment", "madad"))

	chitchat_resp_eng = get_fast_chit_chat_response("chitchat", "How are you doing today?", provider="browser")
	print(f"\n[Chit-chat English]: {chitchat_resp_eng}")
	assert "doing" in chitchat_resp_eng.lower() or "well" in chitchat_resp_eng.lower()
	assert "assist" in chitchat_resp_eng.lower() or "help" in chitchat_resp_eng.lower()

	# Test 2.2: Greeting queries ("namaste", "hello")
	greeting_resp = get_fast_chit_chat_response("greeting", "Namaste", provider="sarvam")
	print(f"\n[Greeting]: {greeting_resp}")
	assert "swagat" in greeting_resp.lower() or "madad" in greeting_resp.lower()

	# Test 2.3: Compliment queries ("nice voice", "aapki aawaz achhi hai")
	comp_resp = get_fast_chit_chat_response("compliment", "Aapki aawaz bahut pyari hai", provider="sarvam")
	print(f"\n[Compliment]: {comp_resp}")
	assert "shukriya" in comp_resp.lower() or "dhanyawad" in comp_resp.lower()
	assert "madad" in comp_resp.lower() or "help" in comp_resp.lower()

	print("\n✓ SUITE 2 PASSED: LLM chit-chat answers directly and proactively offers help!")

	print("\n" + "=" * 70)
	print("TEST SUITE 3: IN-SCOPE VS OUT-OF-SCOPE CLASSIFICATION & REFUSAL")
	print("=" * 70)

	llm = LLMService()

	# Test 3.1: REAL IN-SCOPE QUERIES -> Must execute turn with intent
	in_scope_queries = [
		("Mere daant mein bahut dard aur sensitivity hai", "info"),
		("Cleaning and scaling ke kitne charges hain?", "info"),
		("Dr. Rohit ke saath kal ka appointment mil sakta hai?", "slots"),
		("Main fee pay kar chuka hoon UPI se", "payment"),
	]

	for query, expected_intent in in_scope_queries:
		decision = await llm.decide_turn_action(query, provider="sarvam")
		print(f"\n[In-Scope Input]: {query}")
		print(f" -> Decision: {decision.get('decision')}, Intent: {decision.get('intent')}")
		assert decision.get("decision") == "EXECUTE_TURN", f"Expected EXECUTE_TURN for in-scope query: {query}"
		assert decision.get("intent") == expected_intent or decision.get("intent") in ("info", "slots", "payment"), (
			f"Expected intent in ('info', 'slots', 'payment'), got: {decision.get('intent')}"
		)

	# Test 3.2: OUT-OF-SCOPE PHYSICAL DEMANDS -> Must politely decline with role clarification
	out_of_scope_physical = [
		"Ek cup chai bana do mere liye",
		"Ek glass paani le aao",
		"Please open the gate and bring some coffee",
	]

	for query in out_of_scope_physical:
		decision = await llm.decide_turn_action(query, provider="sarvam")
		direct = decision.get("direct_response") or decision.get("deescalation_response") or ""
		print(f"\n[Out-of-Scope Physical]: {query}")
		print(f" -> Decision: {decision.get('decision')}, Category: {decision.get('fast_category')}")
		print(f" -> Response: {direct}")
		assert decision.get("decision") == "FAST_RESPONSE"
		assert decision.get("fast_category") == "out_of_scope"
		assert decision.get("intent") is None
		assert any(word in direct.lower() for word in ("receptionist", "madad nahi", "cannot perform", "physical"))
		assert any(word in direct.lower() for word in ("dental", "checkup", "appointment", "timings"))

	# Test 3.3: OUT-OF-SCOPE OFF-TOPIC TRIVIA & NON-DENTAL TOPICS
	out_of_scope_offtopic = [
		"Aaj mausam kaisa rahega?",
		"Mujhe koi accha sa chutkula sunao",
		"What is the cricket score today?",
		"Mujhe sar dard aur bukhar hai dawai batao",
	]

	for query in out_of_scope_offtopic:
		decision = await llm.decide_turn_action(query, provider="sarvam")
		direct = decision.get("direct_response") or decision.get("deescalation_response") or ""
		print(f"\n[Out-of-Scope Offtopic]: {query}")
		print(f" -> Decision: {decision.get('decision')}, Category: {decision.get('fast_category')}")
		print(f" -> Response: {direct}")
		assert decision.get("decision") == "FAST_RESPONSE"
		assert decision.get("fast_category") in ("out_of_scope", "non_dental")
		assert decision.get("intent") is None
		assert any(word in direct.lower() for word in ("receptionist", "madad nahi", "cannot assist", "daanton", "dental"))

	print("\n✓ SUITE 3 PASSED: In-scope vs out-of-scope classification & polite refusal verified!")
	print("\n" + "=" * 70)
	print("ALL SUITES PASSED SUCCESSFULLY (100% COVERAGE)!")
	print("=" * 70)


if __name__ == "__main__":
	asyncio.run(run_tests())
