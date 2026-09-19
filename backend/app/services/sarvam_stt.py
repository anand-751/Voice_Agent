"""Sarvam AI Streaming Speech-to-Text (STT) WebSocket Client.

Connects to Sarvam's Realtime Speech-to-Text WebSocket API (saaras:v3-realtime)
using raw linear16 16kHz PCM audio streaming, server-side VAD, interim transcripts,
and event callbacks for sub-second turn detection and instant barge-in.
"""

import asyncio
import base64
import json
import logging
import urllib.parse
from typing import Awaitable, Callable, Optional

import websockets

log = logging.getLogger("sarvam_stt")


class SarvamStreamingSTT:
	"""Manages a single real-time streaming STT session with Sarvam AI."""

	def __init__(
		self,
		api_key: str,
		model: str = "saaras:v3-realtime",
		language_code: str = "hi-IN",
		endpoint: str = "wss://api.sarvam.ai/speech-to-text-realtime/ws",
		silence_duration_ms: int = 550,
		sample_rate: int = 16000,
		prompt: Optional[str] = "Bright Dental Clinic, Dr. Ananya, Dr. Rohit, appointment, booking, RCT, braces",
		on_speech_start: Optional[Callable[[int], Awaitable[None]]] = None,
		on_partial: Optional[Callable[[str, int], Awaitable[None]]] = None,
		on_speech_end: Optional[Callable[[int], Awaitable[None]]] = None,
		on_final: Optional[Callable[[str, int], Awaitable[None]]] = None,
		on_error: Optional[Callable[[dict], Awaitable[None]]] = None,
	):
		self.api_key = api_key
		self.model = model
		self.language_code = language_code
		self.endpoint = endpoint
		self.silence_duration_ms = silence_duration_ms
		self.sample_rate = sample_rate
		self.prompt = prompt

		# Callbacks
		self.on_speech_start = on_speech_start
		self.on_partial = on_partial
		self.on_speech_end = on_speech_end
		self.on_final = on_final
		self.on_error = on_error

		self._ws: Optional[websockets.ClientConnection] = None
		self._recv_task: Optional[asyncio.Task] = None
		self._is_active = False
		self._connected = asyncio.Event()

	@property
	def is_connected(self) -> bool:
		return bool(self._ws and self._is_active)

	def _build_url(self) -> str:
		params = {
			"language_code": self.language_code,
			"model": self.model,
			"stream_type": "balanced",
			"mode": "transcribe",
			"endpointing": "vad",
			"encoding": "linear16",
			"sample_rate": str(self.sample_rate),
			"silence_duration_ms": str(self.silence_duration_ms),
			"prefix_padding_ms": "300",
			"threshold": "0.38",
			"min_speech_duration_ms": "180",
		}
		if self.prompt:
			params["prompt"] = self.prompt
		query_string = urllib.parse.urlencode(params)
		return f"{self.endpoint}?{query_string}"

	async def connect(self) -> bool:
		"""Connect to the Sarvam STT WebSocket and begin listening for events."""
		if not self.api_key:
			log.warning("Cannot connect to Sarvam STT: Missing SARVAM_API_KEY")
			return False

		url = self._build_url()
		headers = {"api-subscription-key": self.api_key}

		try:
			log.info("Connecting to Sarvam Realtime STT at %s (silence=%sms)...", url.split("?")[0], self.silence_duration_ms)
			self._ws = await websockets.connect(
				url,
				additional_headers=headers,
				open_timeout=10,
				ping_interval=20,
				ping_timeout=20,
			)
			self._is_active = True
			self._connected.set()
			self._recv_task = asyncio.create_task(self._receive_loop())
			log.info("Sarvam Realtime STT connected successfully.")
			return True
		except Exception as exc:
			log.error("Failed to connect to Sarvam Realtime STT: %s", exc)
			self._is_active = False
			return False

	async def send_audio(self, chunk: bytes | str) -> bool:
		"""Send a linear16 16kHz PCM audio chunk to Sarvam STT.
		
		Accepts either raw bytes or base64-encoded string.
		"""
		if not self._is_active or not self._ws:
			return False

		try:
			if isinstance(chunk, bytes):
				b64_audio = base64.b64encode(chunk).decode("utf-8")
			else:
				b64_audio = chunk

			payload = {
				"event": "audio_input",
				"audio": b64_audio,
			}
			await self._ws.send(json.dumps(payload))
			return True
		except Exception as exc:
			log.warning("Error sending audio chunk to Sarvam STT: %s", exc)
			return False

	async def _receive_loop(self) -> None:
		"""Continuously read and dispatch events from Sarvam STT."""
		try:
			while self._is_active and self._ws:
				raw = await self._ws.recv()
				try:
					data = json.loads(raw)
				except json.JSONDecodeError:
					continue

				event = data.get("event")
				if event == "session.begin":
					log.info("Sarvam STT session begun (request_id=%s)", data.get("request_id"))
				elif event == "speech_start":
					idx = data.get("utterance_idx", 0)
					log.debug("Sarvam STT: speech_start detected (utterance=%s)", idx)
					if self.on_speech_start:
						try:
							await self.on_speech_start(idx)
						except Exception as cb_err:
							log.exception("Error in on_speech_start callback: %s", cb_err)
				elif event == "transcript.partial":
					text = (data.get("text") or "").strip()
					idx = data.get("utterance_idx", 0)
					if text and self.on_partial:
						try:
							await self.on_partial(text, idx)
						except Exception as cb_err:
							log.exception("Error in on_partial callback: %s", cb_err)
				elif event == "speech_end":
					idx = data.get("utterance_idx", 0)
					log.debug("Sarvam STT: speech_end detected (utterance=%s)", idx)
					if self.on_speech_end:
						try:
							await self.on_speech_end(idx)
						except Exception as cb_err:
							log.exception("Error in on_speech_end callback: %s", cb_err)
				elif event == "transcript.final":
					text = (data.get("text") or "").strip()
					idx = data.get("utterance_idx", 0)
					log.info("Sarvam STT: transcript.final (utterance=%s): %s", idx, text)
					if text and self.on_final:
						try:
							await self.on_final(text, idx)
						except Exception as cb_err:
							log.exception("Error in on_final callback: %s", cb_err)
				elif event == "error":
					log.error("Sarvam STT error event: %s", data)
					if self.on_error:
						try:
							await self.on_error(data)
						except Exception as cb_err:
							log.exception("Error in on_error callback: %s", cb_err)
				elif event == "session.end":
					log.info("Sarvam STT session ended normally (request_id=%s)", data.get("request_id"))
					break
		except asyncio.CancelledError:
			pass
		except websockets.ConnectionClosed:
			log.info("Sarvam STT WebSocket connection closed.")
		except Exception as exc:
			log.exception("Unexpected error in Sarvam STT receive loop: %s", exc)
		finally:
			self._is_active = False

	async def close(self) -> None:
		"""Close the STT session and release resources."""
		self._is_active = False
		if self._recv_task and not self._recv_task.done():
			self._recv_task.cancel()
			try:
				await self._recv_task
			except (asyncio.CancelledError, Exception):
				pass
			self._recv_task = None

		if self._ws:
			try:
				await self._ws.send(json.dumps({"event": "end"}))
			except Exception:
				pass
			try:
				await self._ws.close()
			except Exception:
				pass
			self._ws = None
		log.info("Sarvam STT session closed.")
