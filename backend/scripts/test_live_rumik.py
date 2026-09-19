"""Test live synthesis with Rumik AI Silk TTS API."""

import asyncio
import sys
from pathlib import Path

# Ensure backend root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings
from app.services.rumik import RumikTTSService


async def main():
	settings = get_settings()
	print(f"Testing Rumik AI with key: {settings.RUMIK_API_KEY[:10]}... (configured={settings.rumik_configured})")

	service = RumikTTSService(
		api_key=settings.RUMIK_API_KEY,
		model=settings.RUMIK_TTS_MODEL,
		default_speaker=settings.RUMIK_TTS_SPEAKER,
		default_description=settings.RUMIK_TTS_DESCRIPTION,
	)

	text = "Namaste! Bright Dental Clinic mein aapka swagat hai. Main Niaa bol rahi hoon."
	print(f"Synthesizing: '{text}'...")

	try:
		# 1. Binary synthesis
		wav_bytes = await service.synthesize(text)
		print(f"✓ Binary WAV synthesized successfully: {len(wav_bytes)} bytes")

		# 2. Base64 synthesis (with cache check)
		b64 = await service.synthesize_b64(text)
		print(f"✓ Base64 synthesized successfully: {len(b64)} characters")

		# 3. Cache verification
		wav_cached = await service.synthesize(text)
		assert wav_cached == wav_bytes
		print("✓ Cache hit verified (0ms latency)")

	finally:
		await service.aclose()


if __name__ == "__main__":
	asyncio.run(main())
