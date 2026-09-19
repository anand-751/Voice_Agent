import asyncio
import os
import sys

sys.path.insert(0, os.path.abspath("backend"))

from app.main import create_app
from app.services.session import SessionStore
from app.agent.runner import run_turn
from starlette.testclient import TestClient

def test_payment_and_confirmation_flow():
    app = create_app()
    with TestClient(app) as client:
        sessions = SessionStore()
        session = sessions.create()
        session.profile = {"name": "Test User", "phone": "9876543210", "email": "test@example.com"}

        from datetime import date, timedelta
        from app.services.calendar_service import CalendarService
        tomorrow = (date.today() + timedelta(days=1)).isoformat()
        free_slots = CalendarService(app.state.settings).get_free_slots(tomorrow)
        target_slot = free_slots[0] if free_slots else "15:00"

        # 1. Ask for slot booking
        res1 = asyncio.run(run_turn(app, session, f"I want to book an appointment tomorrow at {target_slot}"))
        print("Step 1 (Select Slot):", res1["type"], res1.get("response")[:60])
        assert res1["type"] == "payment_required", f"Expected payment_required, got {res1['type']}"
        assert "checkout_url" in res1, "Expected checkout_url in payload"
        booking_id = res1["booking"]["id"]
        print("Booking ID:", booking_id)

        # 2. Try sending 'paid' BEFORE clicking the open payment button (booking is pending_payment)
        res2 = asyncio.run(run_turn(app, session, "paid"))
        print("Step 2 (Premature 'paid'):", res2["type"], res2.get("response"))
        assert res2["type"] == "response", f"Expected response (unpaid), got {res2['type']}"
        assert "pending" in res2.get("response", "").lower() or "open payment" in res2.get("response", "").lower() or "unpaid" in str(res2), "Expected unpaid warning"
        # Booking must NOT be confirmed
        booking_record = app.state.bookings.get(booking_id)
        assert booking_record.status != "confirmed", "Booking should NOT be confirmed yet"
        print("✓ Premature 'paid' correctly rejected. Slot remained pending.")

        # 3. Simulate user clicking 'Open payment' button (hits /api/dev/simulate-payment/{booking_id})
        sim_res = client.post(f"/api/dev/simulate-payment/{booking_id}")
        assert sim_res.status_code == 200, f"Simulate payment failed: {sim_res.text}"
        booking_record = app.state.bookings.get(booking_id)
        assert booking_record.status in ("paid", "confirmed"), f"Expected status 'paid' or 'confirmed', got {booking_record.status}"
        print("✓ Open payment button clicked: simulated payment successfully marked booking as paid/confirmed.")

        # 4. Now the automatic 'paid' trigger runs
        res3 = asyncio.run(run_turn(app, session, "paid"))
        print("Step 4 (Automated 'paid' trigger):", res3["type"], res3.get("booking"))
        assert res3["type"] == "booking_confirmed", f"Expected booking_confirmed, got {res3['type']}"
        assert res3["booking"]["id"] == booking_id
        booking_record = app.state.bookings.get(booking_id)
        assert booking_record.status == "confirmed"
        print(f"✓ Booking confirmed successfully for {res3['booking']['date']} at {res3['booking']['time']}!")

if __name__ == "__main__":
    test_payment_and_confirmation_flow()
