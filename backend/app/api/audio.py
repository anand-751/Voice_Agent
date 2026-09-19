import base64
import logging
from fastapi import APIRouter, HTTPException, Request

from ..config import get_settings

log = logging.getLogger("audio")
router = APIRouter(prefix="/audio", tags=["audio"])


@router.post("/transcribe")
async def transcribe_audio_endpoint(request: Request):
	"""Transcribe audio recorded in the browser (Groq Whisper fallback).
	
	Accepts either:
	- Raw binary audio in body (e.g. audio/webm, audio/wav, audio/ogg)
	- JSON with {"audio_base64": "..."}
	"""
	content_type = request.headers.get("content-type", "")
	app = request.app
	audio_bytes = b""
	filename = "speech.webm"

	if "application/json" in content_type:
		try:
			data = await request.json()
			b64_data = data.get("audio_base64", "")
			# strip data uri prefix if present
			if "," in b64_data:
				b64_data = b64_data.split(",", 1)[1]
			audio_bytes = base64.b64decode(b64_data)
			filename = data.get("filename", "speech.webm")
		except Exception as exc:
			raise HTTPException(status_code=400, detail=f"Invalid base64 audio: {exc}")
	else:
		# Raw body
		audio_bytes = await request.body()
		if "wav" in content_type:
			filename = "speech.wav"
		elif "ogg" in content_type:
			filename = "speech.ogg"
		else:
			filename = "speech.webm"

	if not audio_bytes:
		raise HTTPException(status_code=400, detail="Empty audio payload")

	if not getattr(app.state, "llm", None):
		raise HTTPException(status_code=503, detail="LLM service unavailable")

	transcript = app.state.llm.transcribe_audio(audio_bytes, filename=filename)
	return {"text": transcript}


@router.get("/config")
async def get_audio_config(request: Request):
	"""Return audio configuration and available TTS providers."""
	settings = getattr(request.app.state, "settings", None) or get_settings()
	sarvam_service = getattr(request.app.state, "sarvam", None)
	sarvam_configured = bool(sarvam_service and sarvam_service.is_configured)
	rumik_service = getattr(request.app.state, "rumik", None)
	rumik_configured = bool(rumik_service and rumik_service.is_configured)

	tts_provider = getattr(settings, "TTS_PROVIDER", "sarvam").lower()
	if tts_provider == "rumik" and not rumik_configured:
		tts_provider = "sarvam" if sarvam_configured else "browser"
	stt_provider = getattr(settings, "STT_PROVIDER", "sarvam")
	return {
		"tts_provider": tts_provider,
		"default_tts_provider": tts_provider,
		"stt_provider": stt_provider,
		"sarvam_available": sarvam_configured,
		"sarvam_stt_available": bool(settings and getattr(settings, "SARVAM_API_KEY", None)),
		"sarvam_stt_sample_rate": getattr(settings, "SARVAM_STT_SAMPLE_RATE", 16000),
		"sarvam_stt_silence_ms": getattr(settings, "SARVAM_STT_SILENCE_MS", 550),
		"default_speaker": getattr(settings, "SARVAM_TTS_SPEAKER", "simran"),
		"default_pace": getattr(settings, "SARVAM_TTS_PACE", 1.0),
		"default_language": getattr(settings, "SARVAM_TTS_LANGUAGE", "hi-IN"),
		"model": getattr(settings, "SARVAM_TTS_MODEL", "bulbul:v3"),
		# Rumik AI Silk TTS metadata
		"rumik_available": rumik_configured,
		"rumik_model": getattr(settings, "RUMIK_TTS_MODEL", "mulberry"),
		"rumik_speaker": getattr(settings, "RUMIK_TTS_SPEAKER", "aisha"),
		"rumik_pace": getattr(settings, "RUMIK_TTS_PACE", 1.15),
		"rumik_description": getattr(settings, "RUMIK_TTS_DESCRIPTION", ""),
	}


@router.post("/tts")
async def text_to_speech_endpoint(request: Request):
	"""Synthesize speech using active TTS provider (Rumik Silk or Sarvam Bulbul) and return audio stream."""
	settings = getattr(request.app.state, "settings", None) or get_settings()
	sarvam_service = getattr(request.app.state, "sarvam", None)
	rumik_service = getattr(request.app.state, "rumik", None)

	try:
		data = await request.json()
	except Exception:
		raise HTTPException(status_code=400, detail="Invalid JSON body")

	text = (data.get("text") or "").strip()
	if not text:
		raise HTTPException(status_code=400, detail="Text field is required.")

	provider = (data.get("provider") or getattr(settings, "TTS_PROVIDER", "sarvam")).lower()
	speaker = data.get("speaker")
	language = data.get("language")
	pace = data.get("pace") or 1.0
	model = data.get("model")
	description = data.get("description")
	audio_format = data.get("audio_format")

	from fastapi.responses import Response

	# 1. Rumik AI Route
	if provider == "rumik":
		if rumik_service and rumik_service.is_configured:
			try:
				audio_bytes = await rumik_service.synthesize(
					text=text,
					speaker=speaker or getattr(settings, "RUMIK_TTS_SPEAKER", "aisha"),
					model=model or getattr(settings, "RUMIK_TTS_MODEL", "mulberry"),
					description=description or getattr(settings, "RUMIK_TTS_DESCRIPTION", None),
					audio_format=audio_format,
				)
				media_type = "audio/mpeg" if audio_format == "mp3" else "audio/wav"
				ext = "mp3" if audio_format == "mp3" else "wav"
				log.info("Rumik TTS successfully synthesized %d bytes for %d chars: '%.35s...'", len(audio_bytes), len(text), text)
				return Response(
					content=audio_bytes,
					media_type=media_type,
					headers={"Content-Disposition": f"inline; filename=speech.{ext}"},
				)
			except ValueError as val_err:
				raise HTTPException(status_code=400, detail=str(val_err))
			except Exception as err:
				log.warning("Rumik TTS failed (%s), attempting fallback to Sarvam...", err)
		# Fallback to Sarvam if Rumik unconfigured or failed
		if sarvam_service and sarvam_service.is_configured:
			try:
				sarvam_spk = getattr(settings, "SARVAM_TTS_SPEAKER", "simran")
				audio_bytes = await sarvam_service.synthesize(text=text, speaker=sarvam_spk, target_language_code=language, pace=pace)
				return Response(content=audio_bytes, media_type="audio/wav", headers={"Content-Disposition": "inline; filename=speech.wav"})
			except Exception as fb_err:
				log.error("Sarvam fallback also failed: %s", fb_err)
		raise HTTPException(status_code=503, detail="Rumik TTS service is not configured. Missing RUMIK_API_KEY.")

	# 2. Sarvam AI Route (Default)
	if not sarvam_service or not sarvam_service.is_configured:
		# Fallback to Rumik if Sarvam not configured
		if rumik_service and rumik_service.is_configured:
			try:
				audio_bytes = await rumik_service.synthesize(text=text, speaker=speaker)
				return Response(content=audio_bytes, media_type="audio/wav", headers={"Content-Disposition": "inline; filename=speech.wav"})
			except Exception:
				pass
		raise HTTPException(
			status_code=503,
			detail="Sarvam TTS service is not configured. Missing SARVAM_API_KEY.",
		)

	try:
		audio_bytes = await sarvam_service.synthesize(
			text=text,
			speaker=speaker,
			target_language_code=language,
			pace=pace,
		)
		log.info("Sarvam TTS successfully synthesized %d bytes for %d chars: '%.35s...'", len(audio_bytes), len(text), text)
		return Response(
			content=audio_bytes,
			media_type="audio/wav",
			headers={"Content-Disposition": "inline; filename=speech.wav"},
		)
	except ValueError as val_err:
		raise HTTPException(status_code=400, detail=str(val_err))
	except Exception as err:
		log.error("TTS synthesis error: %s", err)
		raise HTTPException(status_code=502, detail=f"TTS synthesis failed: {err}")
