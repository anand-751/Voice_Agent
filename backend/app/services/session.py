import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional
from uuid import uuid4


@dataclass
class Session:
	id: str
	profile: dict = field(default_factory=dict)
	history: deque = field(default_factory=lambda: deque(maxlen=20))
	pending_booking_id: Optional[str] = None
	tts_provider: Optional[str] = None
	created_at: float = field(default_factory=time.time)
	touched_at: float = field(default_factory=time.time)

	last_turn_completed_at: float = 0.0
	last_turn_was_fallback: bool = False
	last_user_input: str = ""

	def add(self, role: str, content: str):
		if content:
			self.history.append({"role": role, "content": content})
			if role == "user":
				self.last_user_input = content
		self.touched_at = time.time()

	def pop_last_turn(self) -> Optional[dict]:
		"""Pop the most recent assistant (and optionally user) turn if aborted/interrupted."""
		if not self.history:
			return None
		asst_text = None
		user_text = None
		if self.history[-1]["role"] == "assistant":
			asst_text = self.history.pop()["content"]
		if self.history and self.history[-1]["role"] == "user":
			user_text = self.history.pop()["content"]
		if user_text or asst_text:
			return {"user": user_text or "", "assistant": asst_text or ""}
		return None


class SessionStore:
	"""In-memory call sessions with TTL garbage collection."""

	def __init__(self, ttl_minutes: int = 30):
		self._sessions = {}
		self._ttl = ttl_minutes * 60

	def create(self) -> Session:
		self._gc()
		session = Session(id=uuid4().hex)
		self._sessions[session.id] = session
		return session

	def get(self, session_id: str):
		return self._sessions.get(session_id)

	def drop(self, session_id: str):
		self._sessions.pop(session_id, None)

	def _gc(self):
		now = time.time()
		stale = [session_id for session_id, session in self._sessions.items()
				 if now - session.touched_at > self._ttl]
		for session_id in stale:
			self._sessions.pop(session_id, None)
