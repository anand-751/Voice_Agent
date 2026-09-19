import json as _json
import logging
import os
from datetime import datetime, timedelta
from uuid import uuid4
from zoneinfo import ZoneInfo

log = logging.getLogger("calendar")


class CalendarService:
	"""Google Calendar service supporting OAuth2 User Token (token.json), Service Accounts, or Mock mode."""

	def __init__(self, settings):
		self.s = settings
		self.tz = ZoneInfo(settings.CLINIC_TIMEZONE)
		self._service = None
		if settings.google_configured:
			self._service = self._build_service()

	def _build_service(self):
		from google.auth.transport.requests import Request
		from google.oauth2.credentials import Credentials
		from google.oauth2 import service_account
		from googleapiclient.discovery import build

		SCOPES = [
			"https://www.googleapis.com/auth/calendar",
			"https://www.googleapis.com/auth/calendar.events",
		]

		# 1. Primary: OAuth2 User Token (token.json) — Best for personal @gmail.com accounts
		token_candidates = [
			self.s.GOOGLE_TOKEN_FILE,
			os.path.join("backend", self.s.GOOGLE_TOKEN_FILE),
			os.path.join(os.path.dirname(__file__), "..", "..", self.s.GOOGLE_TOKEN_FILE),
		]
		token_path = next((p for p in token_candidates if p and os.path.exists(p)), None)

		if token_path:
			try:
				creds = Credentials.from_authorized_user_file(token_path, SCOPES)
				if creds and creds.expired and creds.refresh_token:
					log.info("Refreshing expired Google Calendar OAuth token from %s", token_path)
					creds.refresh(Request())
					# Save back refreshed access token
					with open(token_path, "w") as token_file:
						token_file.write(creds.to_json())
				log.info("Google Calendar client built successfully using OAuth2 token (%s)", token_path)
				return build("calendar", "v3", credentials=creds, cache_discovery=False)
			except Exception as tok_err:
				log.warning("Google Calendar OAuth2 token (%s) could not be refreshed (%s). Falling back to Service Account.", token_path, tok_err)

		# 2. Secondary: Service Account JSON
		if self.s.GOOGLE_SERVICE_ACCOUNT_JSON:
			try:
				info = _json.loads(self.s.GOOGLE_SERVICE_ACCOUNT_JSON)
				credentials = service_account.Credentials.from_service_account_info(info, scopes=SCOPES)
				return build("calendar", "v3", credentials=credentials, cache_discovery=False)
			except Exception:
				log.exception("Failed to build Google Calendar service from GOOGLE_SERVICE_ACCOUNT_JSON")

		# 3. Tertiary: Service Account File
		sa_candidates = [
			self.s.GOOGLE_SERVICE_ACCOUNT_FILE,
			os.path.join("backend", self.s.GOOGLE_SERVICE_ACCOUNT_FILE) if self.s.GOOGLE_SERVICE_ACCOUNT_FILE else "",
			os.path.join(os.path.dirname(__file__), "..", "..", self.s.GOOGLE_SERVICE_ACCOUNT_FILE) if self.s.GOOGLE_SERVICE_ACCOUNT_FILE else "",
		]
		sa_path = next((p for p in sa_candidates if p and os.path.exists(p)), None)
		if sa_path:
			try:
				credentials = service_account.Credentials.from_service_account_file(
					sa_path, scopes=SCOPES
				)
				log.info("Google Calendar client built successfully using Service Account (%s)", sa_path)
				return build("calendar", "v3", credentials=credentials, cache_discovery=False)
			except Exception:
				log.exception("Failed to build Google Calendar service from file %s", sa_path)

		log.warning("Google Calendar credentials not found — running in mock mode")
		return None

	def slot_to_iso(self, date_str: str, time_str: str):
		start = datetime.fromisoformat(f"{date_str}T{time_str}").replace(tzinfo=self.tz)
		end = start + timedelta(minutes=self.s.SLOT_MINUTES)
		return start.isoformat(), end.isoformat()

	def get_free_slots(self, date_str: str):
		# Bright Dental Clinic is closed on Sundays and books up to 8 days in advance
		try:
			target_date = datetime.fromisoformat(date_str).date()
			today = datetime.now(self.tz).date()
			if target_date < today or target_date > today + timedelta(days=8):
				return []
			if datetime.fromisoformat(date_str).weekday() == 6:
				return []
		except Exception:
			pass

		busy = self._busy_intervals(date_str)
		slot = timedelta(minutes=self.s.SLOT_MINUTES)
		current = datetime.fromisoformat(
			f"{date_str}T{self.s.CLINIC_OPEN_HOUR:02d}:00"
		).replace(tzinfo=self.tz)
		close = datetime.fromisoformat(
			f"{date_str}T{self.s.CLINIC_CLOSE_HOUR:02d}:00"
		).replace(tzinfo=self.tz)
		now = datetime.now(self.tz)
		# Slots must be at least 1.5 hours (90 minutes) in advance from current Indian time
		min_slot_time = now + timedelta(minutes=90)

		free = []
		while current + slot <= close:
			end = current + slot
			if current >= min_slot_time and not any(
				self._overlaps(current, end, start, finish)
				for start, finish in busy
			):
				free.append(current.strftime("%H:%M"))
			current += slot
		return free

	def is_slot_free(self, start_iso: str, end_iso: str) -> bool:
		start = datetime.fromisoformat(start_iso)
		end = datetime.fromisoformat(end_iso)
		min_slot_time = datetime.now(self.tz) + timedelta(minutes=90)
		if start < min_slot_time:
			return False
		busy = self._busy_intervals(start.date().isoformat())
		return not any(
			self._overlaps(start, end, start_busy, end_busy)
			for start_busy, end_busy in busy
		)

	@staticmethod
	def _overlaps(first_start, first_end, second_start, second_end) -> bool:
		return first_start < second_end and second_start < first_end

	def _busy_intervals(self, date_str: str):
		if not self._service:
			return []
		day = datetime.fromisoformat(date_str).replace(tzinfo=self.tz)
		time_min = day.replace(hour=0, minute=0, second=0).isoformat()
		time_max = (day + timedelta(days=1)).replace(hour=0, minute=0, second=0).isoformat()
		try:
			response = self._service.events().list(
				calendarId=self.s.GOOGLE_CALENDAR_ID,
				timeMin=time_min,
				timeMax=time_max,
				singleEvents=True,
				orderBy="startTime",
			).execute()
			busy = []
			for event in response.get("items", []):
				start_raw, end_raw = event.get("start", {}), event.get("end", {})
				if "dateTime" not in start_raw:
					continue
				busy.append((
					datetime.fromisoformat(start_raw["dateTime"]),
					datetime.fromisoformat(end_raw["dateTime"]),
				))
			return busy
		except Exception:
			log.exception("Failed to fetch busy intervals from Google Calendar")
			return []

	def create_event(
		self, summary: str, description: str, start_iso: str, end_iso: str, attendee_email: str = None
	) -> dict:
		if not self._service:
			log.info("mock calendar event created: %s @ %s", summary, start_iso)
			return {"id": f"mock_evt_{uuid4().hex[:10]}", "htmlLink": ""}

		body = {
			"summary": summary,
			"description": description,
			"start": {"dateTime": start_iso, "timeZone": self.s.CLINIC_TIMEZONE},
			"end": {"dateTime": end_iso, "timeZone": self.s.CLINIC_TIMEZONE},
			"reminders": {
				"useDefault": False,
				"overrides": [
					{"method": "email", "minutes": 24 * 60},
					{"method": "popup", "minutes": 60},
				],
			},
		}
		if attendee_email:
			body["attendees"] = [{"email": attendee_email}]

		try:
			event = self._service.events().insert(
				calendarId=self.s.GOOGLE_CALENDAR_ID,
				body=body,
				sendUpdates="all" if attendee_email else "none",
			).execute()
			log.info("Google Calendar event created: %s (%s)", event.get("htmlLink"), event.get("id"))
			return event
		except Exception:
			log.exception("Failed to insert event into Google Calendar")
			return {"id": f"mock_evt_{uuid4().hex[:10]}", "htmlLink": ""}
