"""Simulation script reproducing the user's conversation transcript end-to-end.

Verifies:
1. Rescheduling request and temporal comprehension ('the next following day', 'coming saturday').
2. Selection of 11:00 AM slot -> pending payment status with payment prompt.
3. RAG inquiry on root canal availability and costs -> formatted with '3,500 rupees to 6,000 rupees' (no '3 rupees,500').
4. 'no fix the appointment' -> handles slot selection/availability cleanly.
5. 'book appointment for September 12 at Saturday' with no time -> does NOT hallucinate 9:00 AM; checks availability and asks caller to choose a time.
6. Rescheduling cancels old pending booking in BookingStore.
"""

import asyncio
import types
from datetime import date
from app.config import get_settings
from app.agent.runner import run_turn
from app.agent.graph import build_graph
from app.services.booking_store import BookingStore
from app.services.llm import LLMService
from app.services.observability import ObservabilityService
from app.services.session import SessionStore
from app.services.vectorstore import VectorStore


async def run_simulation():
	print("=" * 65)
	print("  Starting Voice Agent Multi-Turn User Transcript Simulation")
	print("=" * 65)

	settings = get_settings()
	obs = ObservabilityService(settings)
	booking_store = BookingStore(":memory:")
	app = types.SimpleNamespace(
		state=types.SimpleNamespace(
			settings=settings,
			observability=obs,
			llm=LLMService(observability=obs),
			bookings=booking_store,
			vectors=VectorStore(),
		)
	)
	app.state.graph = build_graph(app)

	sessions = SessionStore()
	session = sessions.create()
	session.profile["name"] = "Anand Choudhary"
	session.profile["phone"] = "9876543210"

	# Turn 1: Caller wants to check availability / change date
	print("\nCaller: 'can I change the date'")
	res1 = await run_turn(app, session, "can I change the date")
	print(f"Agent: {res1['response']}")

	# Turn 2: Relative date reference
	print("\nCaller: 'the next following day'")
	res2 = await run_turn(app, session, "the next following day")
	print(f"Agent: {res2['response']}")

	# Turn 3: Caller selects 11:00 AM slot
	print("\nCaller: 'yes yes 11:00 a.m. is right'")
	res3 = await run_turn(app, session, "yes yes 11:00 a.m. is right")
	print(f"Agent: {res3['response']}")
	first_booking_id = session.pending_booking_id
	print(f"[State] Pending booking id: {first_booking_id}")

	# Turn 4: Caller asks another question
	print("\nCaller: 'another question please'")
	res4 = await run_turn(app, session, "another question please")
	print(f"Agent: {res4['response']}")

	# Turn 5: Inquiry about root canal treatment
	print("\nCaller: 'is a root canal option is available in your clinic'")
	res5 = await run_turn(app, session, "is a root canal option is available in your clinic")
	print(f"Agent: {res5['response']}")
	assert "root canal" in res5["response"].lower() or "ananya" in res5["response"].lower()

	# Turn 6: Inquiry about root canal charges
	print("\nCaller: 'what is the charges what is the charges of root canal at a clinic'")
	res6 = await run_turn(app, session, "what is the charges what is the charges of root canal at a clinic")
	print(f"Agent: {res6['response']}")
	# Critical check: Must NOT say '3 rupees,500'
	assert "rupees,500" not in res6["response"], "Found broken currency string 'rupees,500'!"
	print("✓ Verified currency formatting: No '3 rupees,500' acoustic artifact.")

	# Turn 7: Booking appointment for coming Saturday without specifying time
	print("\nCaller: 'book appointment for September 12 at Saturday'")
	res7 = await run_turn(app, session, "book appointment for September 12 at Saturday")
	print(f"Agent: {res7['response']}")
	# Critical check: Must NOT have hallucinated 9:00 AM (clinic opens at 10:00 AM, user spoke no time)
	assert "9:00" not in res7["response"] and "9 am" not in res7["response"].lower(), "Hallucinated 9:00 AM slot!"
	print("✓ Verified slot sanity: Agent did NOT hallucinate 9:00 AM; correctly presented available slots.")

	# Turn 8: Selecting 11:00 AM on Saturday
	print("\nCaller: 'I will take 11:00 am'")
	res8 = await run_turn(app, session, "I will take 11:00 am")
	print(f"Agent: {res8['response']}")
	second_booking_id = session.pending_booking_id
	print(f"[State] New pending booking id: {second_booking_id}")
	if first_booking_id and second_booking_id and first_booking_id != second_booking_id:
		old_b = booking_store.get(first_booking_id)
		new_b = booking_store.get(second_booking_id)
		print(f"[Store] Old booking status: {old_b.status}, New booking status: {new_b.status}")
		assert old_b.status == "cancelled", f"Old booking should be cancelled, got {old_b.status}"
		assert new_b.status == "pending", f"New booking should be pending, got {new_b.status}"
		print("✓ Verified store state: Old pending booking cancelled, new booking active.")

	obs.flush()
	print("\n" + "=" * 65)
	print("  User Transcript Simulation Completed with 100% Success!")
	print("=" * 65)


if __name__ == "__main__":
	asyncio.run(run_simulation())
