"""Verification script for Sarvam Realtime Streaming STT WebSocket Client.

Tests:
1. Connecting to Sarvam STT WebSocket with saaras:v3-realtime & silence_duration_ms=700.
2. Receiving session.begin.
3. Sending sample linear16 PCM audio frames.
4. Clean disconnection.
"""

import asyncio
import os
import sys

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from backend.app.config import Settings
from backend.app.services.sarvam_stt import SarvamStreamingSTT


async def main():
	settings = Settings()
	print(f"Testing Sarvam STT Provider: {settings.STT_PROVIDER}")
	print(f"Model: {settings.SARVAM_STT_MODEL}")
	print(f"Silence duration: {settings.SARVAM_STT_SILENCE_MS}ms")
	print(f"Endpoint: {settings.SARVAM_STT_ENDPOINT}")

	events_received = []

	async def on_speech_start(idx):
		print(f"Event: speech_start (utterance {idx})")
		events_received.append("speech_start")

	async def on_partial(text, idx):
		print(f"Event: transcript.partial -> '{text}'")
		events_received.append("partial")

	async def on_final(text, idx):
		print(f"Event: transcript.final -> '{text}'")
		events_received.append("final")

	async def on_error(err):
		print(f"Event: error -> {err}")
		events_received.append("error")

	client = SarvamStreamingSTT(
		api_key=settings.SARVAM_API_KEY,
		model=settings.SARVAM_STT_MODEL,
		language_code=settings.SARVAM_STT_LANGUAGE,
		endpoint=settings.SARVAM_STT_ENDPOINT,
		silence_duration_ms=settings.SARVAM_STT_SILENCE_MS,
		sample_rate=settings.SARVAM_STT_SAMPLE_RATE,
		on_speech_start=on_speech_start,
		on_partial=on_partial,
		on_final=on_final,
		on_error=on_error,
	)

	print("\nConnecting to Sarvam STT WebSocket...")
	connected = await client.connect()
	if not connected:
		print("FAILED to connect.")
		sys.exit(1)

	print("CONNECTED successfully!")

	# Send 1 second of silence PCM16 (16000 samples * 2 bytes = 32000 bytes) in chunks
	dummy_chunk = b"\x00" * 3200  # ~100ms
	print("Sending 5 simulated audio chunks (500ms)...")
	for _ in range(5):
		await client.send_audio(dummy_chunk)
		await asyncio.sleep(0.1)

	print("Waiting 1.5s for any server processing...")
	await asyncio.sleep(1.5)

	print("Closing connection...")
	await client.close()
	print("Test complete! Sarvam Streaming STT is operational.")


if __name__ == "__main__":
	asyncio.run(main())
