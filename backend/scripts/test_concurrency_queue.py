"""Automated tests for CallQueueManager: 2-concurrent limit, FIFO queue, and timeout."""

import asyncio
from app.services.call_queue import CallQueueManager


class MockWebSocket:
	def __init__(self, name: str):
		self.name = name
		self.sent_messages = []
		self.closed = False

	async def send_json(self, data: dict):
		self.sent_messages.append(data)

	async def close(self, code: int = 1000):
		self.closed = True


async def run_queue_tests():
	print("--- 1. Testing 2-Concurrency Admission and Queuing ---")
	manager = CallQueueManager(max_concurrent=2, max_duration_seconds=3)

	ws1 = MockWebSocket("client-1")
	ws2 = MockWebSocket("client-2")
	ws3 = MockWebSocket("client-3")
	ws4 = MockWebSocket("client-4")

	# Client 1 joins -> immediate admission
	task1 = asyncio.create_task(manager.enter_or_wait(ws1, "c1"))
	await asyncio.sleep(0.01)
	assert task1.done() and task1.result() is True
	assert manager.active_count == 1
	assert manager.queue_count == 0
	print("✓ Client 1 admitted immediately (active: 1/2)")

	# Client 2 joins -> immediate admission
	task2 = asyncio.create_task(manager.enter_or_wait(ws2, "c2"))
	await asyncio.sleep(0.01)
	assert task2.done() and task2.result() is True
	assert manager.active_count == 2
	assert manager.queue_count == 0
	print("✓ Client 2 admitted immediately (active: 2/2, slots full)")

	# Client 3 joins -> must be QUEUED at position 1
	task3 = asyncio.create_task(manager.enter_or_wait(ws3, "c3"))
	await asyncio.sleep(0.01)
	assert not task3.done()  # Still waiting in queue
	assert manager.active_count == 2
	assert manager.queue_count == 1
	assert len(ws3.sent_messages) == 1
	assert ws3.sent_messages[0]["type"] == "queued"
	assert ws3.sent_messages[0]["position"] == 1
	assert "Agents are busy kindly wait" in ws3.sent_messages[0]["message"]
	print("✓ Client 3 placed in FIFO queue at Position #1 with message: 'Agents are busy kindly wait'")

	# Client 4 joins -> must be QUEUED at position 2
	task4 = asyncio.create_task(manager.enter_or_wait(ws4, "c4"))
	await asyncio.sleep(0.01)
	assert not task4.done()
	assert manager.active_count == 2
	assert manager.queue_count == 2
	assert ws4.sent_messages[-1]["type"] == "queued"
	assert ws4.sent_messages[-1]["position"] == 2
	print("✓ Client 4 placed in FIFO queue at Position #2")

	print("\n--- 2. Testing Automatic Pop-in on Call End ---")
	# Client 1 hangs up -> Client 3 must immediately pop in!
	await manager.leave("c1")
	await asyncio.sleep(0.05)

	assert task3.done() and task3.result() is True
	assert manager.active_count == 2  # c2 and c3 are active
	assert manager.queue_count == 1   # only c4 remains in queue
	print("✓ Client 3 automatically popped into active call upon Client 1 leaving!")

	# Client 4 should have received an updated queue position (#1)
	assert ws4.sent_messages[-1]["position"] == 1
	print("✓ Client 4 received position update to Position #1")

	# Client 2 hangs up -> Client 4 must pop in
	await manager.leave("c2")
	await asyncio.sleep(0.05)
	assert task4.done() and task4.result() is True
	assert manager.active_count == 2  # c3 and c4 active
	assert manager.queue_count == 0
	print("✓ Client 4 automatically popped into active call upon Client 2 leaving!")

	print("\n--- 3. Testing 120s (Fast-forwarded) Call Duration Limit ---")
	# Create manager with short timeout (0.5s) to test timeout auto-hangup
	timeout_mgr = CallQueueManager(max_concurrent=1, max_duration_seconds=1)
	ws_timeout = MockWebSocket("client-timeout")
	admitted = await timeout_mgr.enter_or_wait(ws_timeout, "timeout_call")
	assert admitted is True
	assert timeout_mgr.active_count == 1

	print("Waiting 1.2s for timeout to trigger...")
	await asyncio.sleep(1.2)
	assert timeout_mgr.active_count == 0
	assert ws_timeout.closed is True
	timeout_msgs = [m for m in ws_timeout.sent_messages if m.get("type") == "call_end"]
	assert len(timeout_msgs) == 1
	assert timeout_msgs[0]["reason"] == "timeout"
	assert "session has ended" in timeout_msgs[0]["response"]
	print(f"✓ Call automatically terminated after duration limit: '{timeout_msgs[0]['response']}'")

	# Clean up remaining tasks
	await manager.leave("c3")
	await manager.leave("c4")

	print("\n All CallQueueManager concurrency, FIFO queue, and duration limit tests passed successfully!")


if __name__ == "__main__":
	asyncio.run(run_queue_tests())
