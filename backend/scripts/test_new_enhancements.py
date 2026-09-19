import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock
from app.services.call_queue import CallQueueManager, ActiveCall
from app.services.llm import _strip_think_tags
from app.agent.guardrails import apply_output_guardrails
from app.agent.harness import ResponseHarness
from app.agent.nodes import HISTORY_TAIL


class TestNewEnhancements(unittest.TestCase):
    def test_history_tail_is_3(self):
        self.assertEqual(HISTORY_TAIL, 3, "HISTORY_TAIL must be exactly 3")

    def test_strip_think_tags_complete(self):
        text = "<think>This is internal reasoning.\nSecond line.</think>Hello! How can I help you?"
        cleaned = _strip_think_tags(text)
        self.assertEqual(cleaned, "Hello! How can I help you?")

    def test_strip_think_tags_unclosed(self):
        text = "Hello! <think>Unfinished thought"
        cleaned = _strip_think_tags(text)
        self.assertEqual(cleaned, "Hello!")

    def test_guardrails_strip_think(self):
        text = "<think>Patient needs appointment</think>Bright Dental Clinic is open Monday to Saturday."
        cleaned = apply_output_guardrails(text)
        self.assertNotIn("<think>", cleaned)
        self.assertNotIn("Patient needs appointment", cleaned)
        self.assertIn("Bright Dental Clinic", cleaned)

    def test_response_harness_strip_think(self):
        state = {
            "tool_result": {},
            "decision": {"action": "chitchat"},
            "payload_type": "response",
        }
        app_mock = MagicMock()
        app_mock.state.settings.BOOKING_FEE_RUPEES = 500
        app_mock.state.observability = None

        reply = "<think>The user wants to book Dr. Rohit Verma.</think>Assistant: Dr. Rohit Verma is available on Tuesdays and Thursdays."
        res = ResponseHarness.verify_and_condition(reply, state, app_mock)
        self.assertNotIn("<think>", res.response)
        self.assertNotIn("The user wants to book", res.response)
        self.assertNotIn("Assistant:", res.response)
        self.assertTrue("Doctor Rohit Verma" in res.response or "Dr. Rohit Verma" in res.response)


class TestCallQueueExtension(unittest.IsolatedAsyncioTestCase):
    async def test_payment_hold_extension_and_timeout(self):
        manager = CallQueueManager(max_concurrent=2, max_duration_seconds=120)
        ws_mock = AsyncMock()

        session_id = "test_session_123"
        await manager.enter_or_wait(ws_mock, session_id)
        self.assertIn(session_id, manager.active_calls)
        active = manager.active_calls[session_id]

        # Initial timeout task should be active
        self.assertIsNotNone(active.timeout_task)

        # Trigger payment hold
        await manager.pause_timeout(session_id)

        # Regular timeout should be cancelled, extended timeout should be running
        self.assertIsNone(active.timeout_task)
        self.assertTrue(active.in_payment_hold)
        self.assertIsNotNone(active.extended_timeout_task)

        # Test cancel on resume (e.g. payment confirmed)
        await manager.resume_timeout(session_id)
        self.assertFalse(active.in_payment_hold)
        self.assertIsNone(active.extended_timeout_task)

        await manager.leave(session_id)
        self.assertEqual(manager.active_count, 0)

    async def test_extended_timeout_watcher_parting_message(self):
        manager = CallQueueManager(max_concurrent=2, max_duration_seconds=120)
        ws_mock = AsyncMock()
        session_id = "test_session_watcher"

        await manager.enter_or_wait(ws_mock, session_id)
        
        # Test watcher with 0.05s duration
        watcher_task = asyncio.create_task(
            manager._extended_timeout_watcher(session_id, ws_mock, duration=0.05)
        )
        await watcher_task

        # Verify call_end was sent with the exact requested parting message
        ws_mock.send_json.assert_called()
        last_sent = ws_mock.send_json.call_args[0][0]
        self.assertEqual(last_sent["type"], "call_end")
        self.assertEqual(last_sent["reason"], "extended_timeout")
        self.assertIn("hope i have answered all of your questions", last_sent["response"].lower())
        self.assertIn("whenever you require to book for the slot, please call us again", last_sent["response"].lower())
        self.assertIn("bright dental clinic", last_sent["response"].lower())

        # Verify ws was closed
        ws_mock.close.assert_called_with(code=1000)

        # Verify active calls cleaned up
        self.assertNotIn(session_id, manager.active_calls)


if __name__ == "__main__":
    unittest.main()
