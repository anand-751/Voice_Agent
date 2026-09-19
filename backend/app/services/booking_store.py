import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from uuid import uuid4


@dataclass
class Booking:
	id: str
	name: str
	phone: str
	email: str
	date: str
	time: str
	start_iso: str
	end_iso: str
	status: str
	stripe_session_id: str
	calendar_event_id: str
	created_at: str
	doctor: str = ""


class BookingStore:
	"""SQLite-backed booking store and Stripe event ledger."""

	def __init__(self, path: str):
		Path(path).parent.mkdir(parents=True, exist_ok=True)
		self._lock = threading.Lock()
		self._db = sqlite3.connect(path, check_same_thread=False)
		self._db.row_factory = sqlite3.Row
		with self._lock, self._db:
			self._db.executescript(
				"""
				CREATE TABLE IF NOT EXISTS bookings (
					id TEXT PRIMARY KEY,
					name TEXT, phone TEXT, email TEXT,
					date TEXT, time TEXT,
					start_iso TEXT, end_iso TEXT,
					status TEXT NOT NULL DEFAULT 'pending',
					stripe_session_id TEXT DEFAULT '',
					calendar_event_id TEXT DEFAULT '',
					created_at TEXT,
					doctor TEXT DEFAULT ''
				);
				CREATE TABLE IF NOT EXISTS stripe_events (
					event_id TEXT PRIMARY KEY,
					created_at TEXT
				);
				CREATE TABLE IF NOT EXISTS profiles (
					phone TEXT PRIMARY KEY,
					name TEXT, email TEXT, created_at TEXT
				);
				"""
			)
			try:
				self._db.execute("ALTER TABLE bookings ADD COLUMN doctor TEXT DEFAULT ''")
			except Exception:
				pass

	def create(self, name, phone, email, date, time, start_iso, end_iso, doctor="") -> Booking:
		booking = Booking(
			id=uuid4().hex[:12],
			name=name, phone=phone, email=email,
			date=date, time=time,
			start_iso=start_iso, end_iso=end_iso,
			status="pending", stripe_session_id="", calendar_event_id="",
			created_at=datetime.now(timezone.utc).isoformat(),
			doctor=doctor,
		)
		with self._lock, self._db:
			self._db.execute(
				"INSERT INTO bookings (id, name, phone, email, date, time, start_iso, end_iso, status, stripe_session_id, calendar_event_id, created_at, doctor) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
				(booking.id, booking.name, booking.phone, booking.email,
				 booking.date, booking.time, booking.start_iso, booking.end_iso,
				 booking.status, booking.stripe_session_id,
				 booking.calendar_event_id, booking.created_at, booking.doctor))
		return booking

	def get(self, booking_id):
		with self._lock:
			row = self._db.execute(
				"SELECT * FROM bookings WHERE id=?", (booking_id,)).fetchone()
		return self._to_booking(row)

	def latest_pending(self, phone=None):
		query = "SELECT * FROM bookings WHERE status IN ('pending','paid')"
		args = []
		if phone:
			query += " AND phone=?"
			args.append(phone)
		query += " ORDER BY created_at DESC LIMIT 1"
		with self._lock:
			row = self._db.execute(query, args).fetchone()
		return self._to_booking(row)

	def attach_session(self, booking_id, session_id):
		self._update(booking_id, stripe_session_id=session_id)

	def mark_paid(self, booking_id, stripe_session_id=""):
		self._update(booking_id, status="paid",
					 stripe_session_id=stripe_session_id or None)

	def mark_confirmed(self, booking_id, calendar_event_id):
		self._update(booking_id, status="confirmed",
					 calendar_event_id=calendar_event_id)

	def cancel(self, booking_id):
		self._update(booking_id, status="cancelled")

	def _update(self, booking_id, **fields):
		fields = {key: value for key, value in fields.items() if value is not None}
		if not fields:
			return
		columns = ", ".join(f"{key}=?" for key in fields)
		with self._lock, self._db:
			self._db.execute(
				f"UPDATE bookings SET {columns} WHERE id=?",
				(*fields.values(), booking_id))

	def event_processed(self, event_id: str) -> bool:
		with self._lock:
			row = self._db.execute(
				"SELECT 1 FROM stripe_events WHERE event_id=?",
				(event_id,)).fetchone()
		return row is not None

	def record_event(self, event_id: str):
		with self._lock, self._db:
			self._db.execute(
				"INSERT OR IGNORE INTO stripe_events VALUES (?,?)",
				(event_id, datetime.now(timezone.utc).isoformat()))

	def save_profile(self, name, phone, email):
		with self._lock, self._db:
			self._db.execute(
				"INSERT OR REPLACE INTO profiles VALUES (?,?,?,?)",
				(phone, name, email, datetime.now(timezone.utc).isoformat()))

	def get_profile(self, phone: str) -> Optional[dict]:
		with self._lock:
			cur = self._db.execute(
				"SELECT name, phone, email FROM profiles WHERE phone = ?", (phone,))
			row = cur.fetchone()
			return dict(row) if row else None

	def get_latest_profile(self) -> Optional[dict]:
		with self._lock:
			cur = self._db.execute(
				"SELECT name, phone, email FROM profiles ORDER BY created_at DESC LIMIT 1")
			row = cur.fetchone()
			return dict(row) if row else None

	@staticmethod
	def _to_booking(row):
		return Booking(**dict(row)) if row else None
