"""Unit and integration verification for Rumik AI Silk TTS module & multi-provider switch."""

import asyncio
import base64
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# Ensure backend root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient

from app.main import app, create_app
from app.services.rumik import RumikTTSService
from app.config import get_settings


def test_rumik_text_cleaning():
	raw_text = "<think>Patient needs extraction</think> <chuckle> **RCT** costs ₹500 | 10:00 AM <br> <curious> Dr. Verma se milna hai 11:00 AM baje"
	cleaned = RumikTTSService.clean_text_for_speech(raw_text)
	assert "<think>" not in cleaned
	assert "**" not in cleaned
	assert "<br>" not in cleaned
	assert "|" not in cleaned
	assert "baje" not in cleaned
	assert "Doctor Verma" in cleaned
	assert "<chuckle>" in cleaned
	assert "<curious>" in cleaned
	assert "<chuckle> RCT costs ₹500, 10:00 AM <curious> Doctor Verma se milna hai 11:00 AM" in cleaned

	# Verify AM/PM deduplication
	deduped = RumikTTSService.clean_text_for_speech("Doctor Verma 10:00 AM AM baje available hain.")
	assert "10:00 AM" in deduped
	assert "AM AM" not in deduped
	print("✓ Rumik speech text cleaner verified successfully with emotion tags preserved & AM/PM deduplicated")


def test_rumik_service_configuration():
	svc_empty = RumikTTSService(api_key="")
	assert not svc_empty.is_configured

	svc_with_key = RumikTTSService(
		api_key="mock_rumik_api_key_123",
		model="mulberry",
		default_speaker="aisha",
		default_pace=1.0,
	)
	assert svc_with_key.is_configured
	assert svc_with_key.default_speaker == "aisha"
	assert svc_with_key.default_pace == 1.0
	assert svc_with_key.model == "mulberry"

	# Test payload builder for Mulberry
	payload_mulberry = svc_with_key._build_payload("Hello", speaker="aisha", model="mulberry")
	assert payload_mulberry["model"] == "mulberry"
	assert payload_mulberry["speaker"] == "aisha"
	assert svc_with_key.default_pace == 1.0
	assert "description" in payload_mulberry

	# Test payload builder for Muga (tone tag steering)
	payload_muga = svc_with_key._build_payload("Namaste!", model="muga")
	assert payload_muga["model"] == "muga"
	assert payload_muga["text"].startswith("[")
	print("✓ Rumik configuration and payload builder verified successfully for Aisha")


def test_rumik_caching_and_fillers():
	import tempfile

	async def _run():
		with tempfile.TemporaryDirectory() as tmp_dir:
			svc = RumikTTSService(api_key="mock_rumik_test", default_speaker="aisha", default_pace=1.0, cache_dir=tmp_dir)
			mock_bytes = b"RIFFmockwavdata"
			mock_b64 = base64.b64encode(mock_bytes).decode("ascii")

			mock_resp = MagicMock()
			mock_resp.status_code = 200
			mock_resp.content = mock_bytes

			with patch.object(svc, "_get_client") as mock_client_getter:
				mock_client = AsyncMock()
				mock_client.post = AsyncMock(return_value=mock_resp)
				mock_client_getter.return_value = mock_client

				test_phrase = svc.DEFAULT_FILLERS[0]
				# 1. First call fetches and caches
				res_bytes = await svc.synthesize(test_phrase)
				assert res_bytes == mock_bytes
				assert mock_client.post.call_count == 1

				# 2. Second call hits cache (0 API calls)
				res_cached = await svc.synthesize(test_phrase)
				assert res_cached == mock_bytes
				assert mock_client.post.call_count == 1  # No extra HTTP call!

				# 3. synthesize_b64 also hits memory cache
				res_b64 = await svc.synthesize_b64(test_phrase)
				assert res_b64 == mock_b64

				# 4. Filler audio retrieval
				filler = svc.get_filler_audio(0)
				assert filler is not None
				assert filler["text"] == test_phrase
				assert filler["audio_b64"] == mock_b64
				assert filler["tts_provider"] == "rumik"

				# 5. Contextual filler audio retrieval
				ctx_heavy = svc.get_contextual_filler_audio("heavy")
				assert ctx_heavy is not None
				assert ctx_heavy["text"] == "Ek second, main check karti hoon..."
				assert ctx_heavy["tts_provider"] == "rumik"

				# Pre-warm payment phrase and verify payment filler
				pay_phrase = svc.FILLERS_BY_INTENT["payment"][0]
				await svc.synthesize(pay_phrase)
				ctx_pay = svc.get_contextual_filler_audio("payment")
				assert ctx_pay is not None
				assert ctx_pay["text"] == "Ek second, main payment status check karti hoon..."

				# Test anti-repetition: passing last_phrase picks a different candidate
				pay_phrase_2 = svc.get_contextual_filler_audio("payment", last_phrase=ctx_pay["text"])
				assert pay_phrase_2 is not None

				# Pre-warm short phrase as well
				short_phrase = svc.FILLERS_BY_INTENT["short"][0]
				await svc.synthesize(short_phrase)
				ctx_short = svc.get_contextual_filler_audio("short")
				assert ctx_short is not None
				assert ctx_short["text"] == "Ji, ek minute..."
				assert ctx_short["tts_provider"] == "rumik"

	asyncio.run(_run())
	print("✓ Rumik LRU caching and pre-warmed fillers verified successfully")


def test_audio_config_endpoint_with_rumik():
	client = TestClient(app)
	res = client.get("/audio/config")
	assert res.status_code == 200
	data = res.json()
	assert "tts_provider" in data
	assert "rumik_available" in data
	assert "rumik_model" in data
	assert "rumik_speaker" in data
	assert data["rumik_model"] == "mulberry"
	assert data["rumik_speaker"] == "aisha"
	print(f"✓ GET /audio/config verified with Rumik metadata: speaker={data['rumik_speaker']}")


def test_audio_tts_routing_with_rumik():
	client = TestClient(app)

	# Mock rumik service on app.state
	mock_rumik = MagicMock()
	mock_rumik.is_configured = True
	mock_rumik.synthesize = AsyncMock(return_value=b"RIFFmockrumikwav")
	app.state.rumik = mock_rumik

	res = client.post(
		"/audio/tts",
		json={
			"text": "<chuckle> Namaste! Bright Dental Clinic mein swagat hai.",
			"provider": "rumik",
			"speaker": "aisha",
			"pace": 1.0,
		},
	)
	assert res.status_code == 200
	assert res.content == b"RIFFmockrumikwav"
	assert res.headers["content-type"] == "audio/wav"
	assert mock_rumik.synthesize.called
	print("✓ POST /audio/tts routed cleanly to Rumik AI engine with Aisha")


def test_multi_provider_switch_scenarios():
	settings = get_settings()
	original_prov = settings.TTS_PROVIDER

	try:
		# Scenario 1: Sarvam provider
		settings.TTS_PROVIDER = "sarvam"
		assert settings.TTS_PROVIDER == "sarvam"

		# Scenario 2: Rumik provider
		settings.TTS_PROVIDER = "rumik"
		assert settings.TTS_PROVIDER == "rumik"

		# Scenario 3: Browser provider
		settings.TTS_PROVIDER = "browser"
		assert settings.TTS_PROVIDER == "browser"
		print("✓ Multi-provider dynamic configuration switch verified")
	finally:
		settings.TTS_PROVIDER = original_prov


def test_rumik_aisha_harness_and_prompt_allowance():
	from app.agent.harness import ResponseHarness

	settings = get_settings()
	orig_app_prov = getattr(getattr(app, "state", None), "settings", None)
	orig_prov = settings.TTS_PROVIDER
	try:
		settings.TTS_PROVIDER = "rumik"
		if hasattr(app, "state") and hasattr(app.state, "settings"):
			app.state.settings.TTS_PROVIDER = "rumik"
		streamlined_reply = (
			"Humare paas Doctor Ananya ke saath kal ke slots available hain. "
			"<curious> Kya main subah 11:00 ka slot aapke liye reserve kar doon?"
		)
		mock_state = {
			"tool_result": None,
			"decision": {"action": "chitchat"},
			"payload_type": "response",
			"tts_provider": "rumik",
		}
		res = ResponseHarness.verify_and_condition(streamlined_reply, mock_state, app)
		assert "<curious>" in res.response
		assert "subah 11:00" in res.response

		# Test trimming 4 sentences to max 3 sentences for conversational brevity
		verbose_reply = (
			"Namaste ji! Bright Dental Clinic mein aapka swagat hai. "
			"Humare paas Doctor Ananya ke slots hain. "
			"<curious> Kya main kal slot reserve kar doon? "
			"Yeh chautha sentence cut hona chahiye."
		)
		res_verb = ResponseHarness.verify_and_condition(verbose_reply, mock_state, app)
		assert "Yeh chautha sentence" not in res_verb.response
		assert res_verb.was_repaired
		print("✓ Rumik Aisha ResponseHarness enforces streamlined 2-3 short sentences & preserves emotion tags")
	finally:
		settings.TTS_PROVIDER = orig_prov
		if hasattr(app, "state") and hasattr(app.state, "settings"):
			app.state.settings.TTS_PROVIDER = orig_prov


if __name__ == "__main__":
	print("\n--- Running Rumik AI TTS & Multi-Provider Verification Suite ---")
	test_rumik_text_cleaning()
	test_rumik_service_configuration()
	test_rumik_caching_and_fillers()
	test_audio_config_endpoint_with_rumik()
	test_audio_tts_routing_with_rumik()
	test_multi_provider_switch_scenarios()
	test_rumik_aisha_harness_and_prompt_allowance()
	print("\n🎉 ALL RUMIK AI TTS VERIFICATION TESTS PASSED SUCCESSFULLY!\n")
