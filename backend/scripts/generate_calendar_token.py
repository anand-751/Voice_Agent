"""Interactive Google Calendar OAuth2 Token Generator.

Generates `token.json` for the Voice Agent to access Google Calendar APIs.

Usage:
    python backend/scripts/generate_calendar_token.py
"""

import json
import os
import sys
from pathlib import Path

SCOPES = [
	"https://www.googleapis.com/auth/calendar",
	"https://www.googleapis.com/auth/calendar.events",
]


def find_credentials_file() -> str:
	candidates = [
		"credentials.json",
		"backend/credentials.json",
		os.path.join(os.path.dirname(__file__), "..", "credentials.json"),
		os.path.join(os.path.dirname(__file__), "..", "..", "credentials.json"),
	]
	for p in candidates:
		if os.path.exists(p):
			return os.path.abspath(p)
	return ""


def prompt_for_credentials() -> str:
	print("\n" + "=" * 70)
	print("  GOOGLE CALENDAR OAUTH2 SETUP")
	print("=" * 70)
	print("\nTo allow the AI receptionist to access your Google Calendar, you need")
	print("an OAuth 2.0 Client ID from the Google Cloud Console.")
	print("\nQuick 3-step setup:")
	print("  1. Go to: https://console.cloud.google.com/apis/credentials")
	print("  2. Create Credentials -> OAuth Client ID -> Application Type: Desktop app")
	print("  3. Download the JSON and save it as 'credentials.json' in this folder.")
	print("=" * 70)

	choice = input("\nDo you have a 'credentials.json' file ready? (y/n): ").strip().lower()
	if choice != "y":
		print("\nOr, you can paste your Client ID and Client Secret directly:")
		client_id = input("Enter Google OAuth Client ID: ").strip()
		client_secret = input("Enter Google OAuth Client Secret: ").strip()
		if not client_id or not client_secret:
			print("Error: Client ID and Secret are required.")
			sys.exit(1)

		cred_data = {
			"installed": {
				"client_id": client_id,
				"client_secret": client_secret,
				"auth_uri": "https://accounts.google.com/o/oauth2/auth",
				"token_uri": "https://oauth2.googleapis.com/token",
				"redirect_uris": ["http://localhost"],
			}
		}
		target_path = os.path.abspath("credentials.json")
		with open(target_path, "w") as f:
			json.dump(cred_data, f, indent=2)
		print(f"Saved credentials to {target_path}")
		return target_path

	path_input = input("Enter path to credentials.json (or press Enter for ./credentials.json): ").strip()
	path = path_input if path_input else "credentials.json"
	if not os.path.exists(path):
		print(f"Error: File '{path}' does not exist.")
		sys.exit(1)
	return os.path.abspath(path)


def generate_token():
	try:
		from google_auth_oauthlib.flow import InstalledAppFlow
		from googleapiclient.discovery import build
	except ImportError:
		print("Installing missing dependencies: google-auth-oauthlib...")
		os.system(f"{sys.executable} -m pip install google-auth-oauthlib")
		from google_auth_oauthlib.flow import InstalledAppFlow
		from googleapiclient.discovery import build

	cred_file = find_credentials_file()
	if not cred_file:
		cred_file = prompt_for_credentials()

	print(f"\nUsing credentials file: {cred_file}")
	print("Launching browser for Google Sign-In and Calendar permission consent...")

	flow = InstalledAppFlow.from_client_secrets_file(cred_file, SCOPES)
	# Run local server to capture the redirect callback
	creds = flow.run_local_server(
		port=0,
		prompt="consent",
		access_type="offline",
		authorization_prompt_message="Please visit this URL to authorize Google Calendar: {url}",
		success_message="Authentication successful! You may now close this browser tab.",
	)

	token_json = creds.to_json()

	# Save token to project root and backend folder for seamless resolution
	out_paths = [
		os.path.abspath("token.json"),
		os.path.abspath("backend/token.json"),
	]

	for p in out_paths:
		os.makedirs(os.path.dirname(p), exist_ok=True)
		with open(p, "w") as f:
			f.write(token_json)
		print(f"✓ Saved OAuth token to: {p}")

	print("\n--- Verifying Google Calendar Connection ---")
	service = build("calendar", "v3", credentials=creds, cache_discovery=False)
	primary_cal = service.calendars().get(calendarId="primary").execute()
	print(f"✓ Success! Connected to primary calendar: {primary_cal.get('summary')} ({primary_cal.get('id')})")
	print(f"Timezone: {primary_cal.get('timeZone')}")
	print("\nGoogle Calendar is now fully configured and active for the Voice Agent!")


if __name__ == "__main__":
	generate_token()
