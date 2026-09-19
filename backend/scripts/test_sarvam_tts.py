"""Unit and integration verification for Sarvam TTS module & routing switch."""

import asyncio
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient

from app.main import app
from app.services.sarvam import SarvamTTSService


def test_sarvam_text_cleaning():
	raw_text = "<think>Patient is asking about cost</think> **RCT** costs ₹500 | 10:00 AM <br> #Appointment"
	cleaned = SarvamTTSService.clean_text_for_speech(raw_text)
	assert "<think>" not in cleaned
	assert "**" not in cleaned
	assert "<br>" not in cleaned
	assert "#" not in cleaned
	assert "|" not in cleaned
	assert "RCT costs ₹500, 10:00 AM Appointment" in cleaned
	print("✓ Sarvam speech text cleaner verified successfully")


def test_sarvam_service_configuration():
	svc_empty = SarvamTTSService(api_key="")
	assert not svc_empty.is_configured

	svc_with_key = SarvamTTSService(api_key="sarvam_mock_key_123")
	assert svc_with_key.is_configured
	assert svc_with_key.default_speaker == "simran"
	assert svc_with_key.default_pace == 1.0
	assert svc_with_key.model == "bulbul:v3"
	assert svc_with_key.default_language == "hi-IN"
	print("✓ Sarvam configuration flags verified successfully")


def test_audio_config_endpoint():
	client = TestClient(app)
	res = client.get("/audio/config")
	assert res.status_code == 200
	data = res.json()
	assert "tts_provider" in data
	assert "sarvam_available" in data
	assert data["default_speaker"] == "simran"
	assert data["default_pace"] == 1.0
	assert data["model"] == "bulbul:v3"
	print(f"✓ GET /audio/config returned: {data}")


def test_audio_tts_endpoint_validation():
	client = TestClient(app)
	# Case 1: Empty text
	res = client.post("/audio/tts", json={"text": "   "})
	# If key is missing, 503; if key is present, 400
	assert res.status_code in (400, 503)
	print("✓ /audio/tts empty payload validation verified")


def test_sarvam_mocked_synthesis():
	async def _run():
		service = SarvamTTSService(api_key="mock_key")
		fake_audio_bytes = b"RIFFmockwavheaderanddata"

		with patch("httpx.AsyncClient.post") as mock_post:
			mock_response = AsyncMock()
			mock_response.status_code = 200
			import base64
			b64_val = base64.b64encode(fake_audio_bytes).decode("utf-8")
			mock_response.json = lambda: {"audios": [b64_val]}
			mock_post.return_value = mock_response

			audio = await service.synthesize("Hello, welcome to Bright Dental Clinic")
			assert audio == fake_audio_bytes
			print("✓ Sarvam mocked async audio synthesis verified successfully")
			await service.aclose()

	asyncio.run(_run())


def test_sarvam_persistent_client():
	async def _run():
		service = SarvamTTSService(api_key="mock_key")
		client1 = service._get_client()
		client2 = service._get_client()
		assert client1 is client2, "Client should be reused for persistent connection pooling"
		assert not client1.is_closed
		await service.aclose()
		assert service._client is None
		print("✓ Sarvam persistent client connection pooling & aclose verified successfully")

	asyncio.run(_run())


def test_sarvam_b64_and_warmup():
	async def _run():
		service = SarvamTTSService(api_key="mock_key")
		fake_audio_bytes = b"RIFFmockwavheaderanddata"
		import base64
		b64_val = base64.b64encode(fake_audio_bytes).decode("utf-8")

		with patch("httpx.AsyncClient.post") as mock_post:
			mock_response = AsyncMock()
			mock_response.status_code = 200
			mock_response.json = lambda: {"audios": [b64_val]}
			mock_post.return_value = mock_response

			# 1. Direct b64 synthesis
			b64_res = await service.synthesize_b64("Namaste! Bright Dental Clinic")
			assert b64_res == b64_val
			# 2. Check cache hit (mock should not be called again)
			mock_post.reset_mock()
			cached_b64 = await service.synthesize_b64("Namaste! Bright Dental Clinic")
			assert cached_b64 == b64_val
			assert mock_post.call_count == 0, "Second call must be 0ms cache hit"

			# 3. Warmup helper
			await service.warmup(["Another phrase to warm up"])
			await service.aclose()
			print("✓ Sarvam synthesize_b64 & cache warmup verified successfully")

	asyncio.run(_run())


if __name__ == "__main__":
	print("Running Sarvam TTS module verification tests...\n")
	test_sarvam_text_cleaning()
	test_sarvam_service_configuration()
	test_sarvam_mocked_synthesis()
	test_sarvam_b64_and_warmup()
	test_sarvam_persistent_client()
	test_audio_config_endpoint()
	test_audio_tts_endpoint_validation()
	print("\nALL SARVAM TTS VERIFICATIONS PASSED!")


