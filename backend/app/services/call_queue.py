"""Call Concurrency & FIFO Queue Manager for Voice Agent.

Enforces:
1. Max concurrent calls (e.g. 2 calls).
2. FIFO Waiting queue when all receptionist slots are occupied with live queue positions.
3. Automatic pop-in: When an active call ends, the next user in queue is automatically admitted.
4. Per-call session timeout (e.g. 120 seconds) to conserve LLM quotas.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Coroutine, Dict, List, Optional
from fastapi import WebSocket

log = logging.getLogger("call_queue")


@dataclass
class ActiveCall:
	session_id: str
	ws: WebSocket
	started_at: float = field(default_factory=time.time)
	timeout_task: Optional[asyncio.Task] = None
	extended_timeout_task: Optional[asyncio.Task] = None
	in_payment_hold: bool = False
	payment_hold_started_at: Optional[float] = None
	is_speech_in_progress: Optional[Callable[[], bool]] = None
	wait_for_speech_complete: Optional[Callable[[], Coroutine]] = None


@dataclass
class QueuedCaller:
	session_id: str
	ws: WebSocket
	admitted_event: asyncio.Event = field(default_factory=asyncio.Event)
	cancelled: bool = False
	queued_at: float = field(default_factory=time.time)


class CallQueueManager:
	def __init__(self, max_concurrent: int = 2, max_duration_seconds: int = 180):
		self.max_concurrent = max_concurrent
		self.max_duration_seconds = max_duration_seconds
		self.active_calls: Dict[str, ActiveCall] = {}
		self.wait_queue: List[QueuedCaller] = []
		self._lock = asyncio.Lock()

	@property
	def active_count(self) -> int:
		return len(self.active_calls)

	@property
	def queue_count(self) -> int:
		return len(self.wait_queue)

	async def enter_or_wait(self, ws: WebSocket, session_id: str) -> bool:
		"""Attempt to join an active call immediately, or wait in the FIFO queue.
		
		Returns True when admitted, or False if aborted/cancelled.
		"""
		caller: Optional[QueuedCaller] = None

		async with self._lock:
			if len(self.active_calls) < self.max_concurrent:
				self._admit_active(session_id, ws)
				log.info("Call %s admitted immediately (Active: %d/%d)", session_id, len(self.active_calls), self.max_concurrent)
				return True

			# Otherwise, place in FIFO queue
			caller = QueuedCaller(session_id=session_id, ws=ws)
			self.wait_queue.append(caller)
			pos = len(self.wait_queue)
			log.info("Call %s queued at position #%d", session_id, pos)

			try:
				await ws.send_json({
					"type": "queued",
					"position": pos,
					"message": "Agents are busy kindly wait",
				})
			except Exception:
				self.wait_queue.remove(caller)
				return False

		# Await until popped into active slot by an ending call
		try:
			await caller.admitted_event.wait()
			return not caller.cancelled
		except asyncio.CancelledError:
			async with self._lock:
				caller.cancelled = True
				if caller in self.wait_queue:
					self.wait_queue.remove(caller)
				await self._broadcast_queue_updates()
			return False

	def _admit_active(self, session_id: str, ws: WebSocket):
		"""Admit a caller into an active slot and launch the duration timer."""
		active = ActiveCall(session_id=session_id, ws=ws)
		# Start 120s timeout watcher
		active.timeout_task = asyncio.create_task(
			self._timeout_watcher(session_id, ws, self.max_duration_seconds)
		)
		self.active_calls[session_id] = active

	async def _timeout_watcher(self, session_id: str, ws: WebSocket, duration: int):
		"""Automatically terminates call when max duration expires, gracefully waiting for any speech in progress."""
		try:
			await asyncio.sleep(duration)
			log.info("Call %s reached max duration limit (%ds). Checking for in-flight turn or speech...", session_id, duration)

			active = self.active_calls.get(session_id)
			if active and active.is_speech_in_progress and active.is_speech_in_progress():
				log.info(
					"Call %s timer ended while speech or response turn is in-flight. "
					"Gracefully waiting for answer speech to finish before sending closing message...",
					session_id,
				)
				if active.wait_for_speech_complete:
					try:
						# Wait up to 35s max grace period for the agent's answer to finish speaking
						await asyncio.wait_for(active.wait_for_speech_complete(), timeout=35.0)
						log.info("Call %s in-flight answer speech finished completely.", session_id)
					except (asyncio.TimeoutError, Exception) as w_err:
						log.warning("Grace period wait ended for session %s: %s", session_id, w_err)

			mins = duration // 60
			log.info("Call %s max duration reached and speech completed. Ending call with closing message.", session_id)
			try:
				await ws.send_json({
					"type": "call_end",
					"reason": "timeout",
					"response": (
						f"Your {mins}-minute call session has ended. "
						f"Thank you for contacting Bright Dental Clinic!"
					),
				})
			except Exception:
				pass

			# Flush call_end message before closing websocket
			try:
				await asyncio.sleep(0.1)
				await ws.close(code=1000)
			except Exception:
				pass
			await self.leave(session_id)
		except asyncio.CancelledError:
			# Normal hangup before timeout
			pass

	async def _extended_timeout_watcher(self, session_id: str, ws: WebSocket, duration: float):
		"""Automatically terminates call when extended payment hold is reached, gracefully waiting for speech."""
		try:
			await asyncio.sleep(duration)
			log.info("Call %s reached maximum extended duration limit (%.1fs). Checking for in-flight speech...", session_id, duration)

			active = self.active_calls.get(session_id)
			if active and active.is_speech_in_progress and active.is_speech_in_progress():
				if active.wait_for_speech_complete:
					try:
						await asyncio.wait_for(active.wait_for_speech_complete(), timeout=30.0)
					except Exception:
						pass

			parting_msg = (
				"I hope I have answered all of your questions! "
				"Whenever you require to book for the slot, please call us again. "
				"Thank you for contacting Bright Dental Clinic, have a wonderful day!"
			)
			try:
				await ws.send_json({
					"type": "call_end",
					"reason": "extended_timeout",
					"response": parting_msg,
				})
				await asyncio.sleep(12.0)
				await ws.close(code=1000)
			except Exception:
				pass
			await self.leave(session_id)
		except asyncio.CancelledError:
			pass

	async def leave(self, session_id: str):
		"""Handle caller disconnection or manual hangup."""
		async with self._lock:
			# Case 1: Caller was active
			if session_id in self.active_calls:
				active = self.active_calls.pop(session_id)
				if active.timeout_task and not active.timeout_task.done():
					active.timeout_task.cancel()
				if active.extended_timeout_task and not active.extended_timeout_task.done():
					active.extended_timeout_task.cancel()
				log.info("Active call %s ended (Active: %d/%d)", session_id, len(self.active_calls), self.max_concurrent)

				# Admit next queued caller if any
				while self.wait_queue:
					next_caller = self.wait_queue.pop(0)
					if not next_caller.cancelled:
						self._admit_active(next_caller.session_id, next_caller.ws)
						next_caller.admitted_event.set()
						log.info("Queued caller %s automatically popped into active call", next_caller.session_id)
						break

				await self._broadcast_queue_updates()
				return

			# Case 2: Caller was in wait queue
			matching = [c for c in self.wait_queue if c.session_id == session_id]
			for c in matching:
				c.cancelled = True
				self.wait_queue.remove(c)
				log.info("Queued caller %s left the queue", session_id)
			if matching:
				await self._broadcast_queue_updates()

	async def pause_timeout(self, session_id: str):
		"""Pause the initial call limit while the user completes payment.

		The call is extended in the backend by up to 2 additional minutes
		(with a hard cap of 4 minutes total from call start). If the user doesn't book
		and keeps chatting, the background timer automatically cuts the call at 4 minutes.
		"""
		async with self._lock:
			active = self.active_calls.get(session_id)
			if not active:
				return

			# Cancel standard initial timer
			if active.timeout_task and not active.timeout_task.done():
				active.timeout_task.cancel()
				active.timeout_task = None

			# If already on payment hold with active extension watcher, don't restart it
			if active.in_payment_hold and active.extended_timeout_task and not active.extended_timeout_task.done():
				return

			active.in_payment_hold = True
			active.payment_hold_started_at = time.time()

			# Strict hard cap of max_duration_seconds (180s / 3 min) from initial started_at, with 1.0s minimum hold
			now = time.time()
			elapsed_total = now - active.started_at
			remaining_to_max = max(1.0, float(self.max_duration_seconds) - elapsed_total)
			extension_duration = remaining_to_max

			log.info(
				"Call %s payment hold started. Extending by up to %.1fs (total elapsed: %.1fs)",
				session_id, extension_duration, elapsed_total,
			)

			active.extended_timeout_task = asyncio.create_task(
				self._extended_timeout_watcher(session_id, active.ws, extension_duration)
			)

	async def resume_timeout(self, session_id: str):
		"""Cancel extended timeout when booking is confirmed or payment completes."""
		async with self._lock:
			active = self.active_calls.get(session_id)
			if active:
				if active.extended_timeout_task and not active.extended_timeout_task.done():
					active.extended_timeout_task.cancel()
					active.extended_timeout_task = None
				active.in_payment_hold = False

	async def _broadcast_queue_updates(self):
		"""Send updated queue position to all waiting callers."""
		for idx, waiter in enumerate(self.wait_queue):
			if not waiter.cancelled:
				try:
					await waiter.ws.send_json({
						"type": "queued",
						"position": idx + 1,
						"message": "Agents are busy kindly wait",
					})
				except Exception:
					pass
