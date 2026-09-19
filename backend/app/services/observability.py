import contextvars
import logging
import os
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

import httpx
from langfuse import Langfuse

from ..config import Settings, get_settings

log = logging.getLogger("observability")

_current_trace_var: contextvars.ContextVar = contextvars.ContextVar("langfuse_current_trace", default=None)


class ObservabilityService:
	"""Langfuse observability service for local or production tracing of AI agents.
	Compatible with Langfuse v2.x OSS server and SDK.
	"""

	def __init__(self, settings: Optional[Settings] = None):
		self.settings = settings or get_settings()
		self.host = self.settings.LANGFUSE_HOST.rstrip("/")
		self.public_key = self.settings.LANGFUSE_PUBLIC_KEY
		self.secret_key = self.settings.LANGFUSE_SECRET_KEY
		self.is_active = False
		self.client: Optional[Langfuse] = None

		if not self.settings.LANGFUSE_ENABLED:
			log.info("Langfuse observability is disabled via configuration.")
			return

		# Fast probe to check if Langfuse server is responding (<350ms)
		server_online = self._probe_server()
		if not server_online:
			log.info(
				"Langfuse server at %s is offline or not responding. Observability running in safe mode "
				"(0 latency, no network retries).",
				self.host,
			)
			return

		try:
			os.environ["LANGFUSE_PUBLIC_KEY"] = self.public_key
			os.environ["LANGFUSE_SECRET_KEY"] = self.secret_key
			os.environ["LANGFUSE_HOST"] = self.host

			self.client = Langfuse(
				public_key=self.public_key,
				secret_key=self.secret_key,
				host=self.host,
				debug=False,
			)
			# Validate credentials against server
			if self.client.auth_check():
				self.is_active = True
				log.info(
					"Langfuse observability active — connected to %s (project public key: %s...)",
					self.host,
					self.public_key[:12] if self.public_key else "none",
				)
			else:
				log.info(
					"Langfuse server at %s is online, but API keys are not valid or not set yet. "
					"Running in safe mode until keys are provided in .env.",
					self.host,
				)
		except Exception as exc:
			log.warning("Langfuse client auth check failed: %s. Safe mode active.", exc)
			self.is_active = False

	def _probe_server(self) -> bool:
		try:
			with httpx.Client(timeout=0.35) as client:
				res = client.get(f"{self.host}/api/public/health")
				return res.status_code in (200, 401, 403)
		except Exception:
			return False

	@contextmanager
	def observe_turn(
		self,
		session_id: str,
		user_input: str,
		profile: Optional[Dict[str, Any]] = None,
		metadata: Optional[Dict[str, Any]] = None,
	):
		"""Wrap an entire conversational turn. Groups all child spans/generations under one trace."""
		profile = profile or {}
		user_id = profile.get("phone") or profile.get("name") or session_id
		meta = {
			"clinic": self.settings.CLINIC_NAME,
			"environment": self.settings.ENV,
			"profile": profile,
		}
		if metadata:
			meta.update(metadata)

		tags = ["voice_agent", self.settings.ENV]
		if profile.get("source"):
			tags.append(str(profile["source"]))

		if not self.is_active or not self.client:
			yield None
			return

		trace = self.client.trace(
			name="voice_agent_turn",
			session_id=session_id,
			user_id=user_id,
			input=user_input,
			metadata=meta,
			tags=tags,
		)
		token = _current_trace_var.set(trace)
		try:
			yield trace
		finally:
			_current_trace_var.reset(token)

	@contextmanager
	def observe_guardrail(self, name: str, input_text: Optional[str] = None):
		"""Record a guardrail evaluation span."""
		current_trace = _current_trace_var.get()
		if not self.is_active or not current_trace:
			yield None
			return
		span = current_trace.span(
			name=f"guardrail:{name}",
			input=input_text,
		)
		try:
			yield span
		finally:
			span.end()

	@contextmanager
	def observe_generation(
		self,
		name: str,
		model: str,
		messages: List[Dict[str, Any]],
		model_parameters: Optional[Dict[str, Any]] = None,
	):
		"""Record an LLM generation call with model name, parameters, inputs, and usage details."""
		current_trace = _current_trace_var.get()
		if not self.is_active or not current_trace:
			yield None
			return
		gen = current_trace.generation(
			name=name,
			model=model,
			input=messages,
			model_parameters=model_parameters or {},
		)
		try:
			yield gen
		finally:
			gen.end()

	@contextmanager
	def observe_tool(self, tool_name: str, args: Optional[Dict[str, Any]] = None):
		"""Record a tool execution span."""
		current_trace = _current_trace_var.get()
		if not self.is_active or not current_trace:
			yield None
			return
		span = current_trace.span(
			name=f"tool:{tool_name}",
			input=args or {},
			metadata={"tool_name": tool_name},
		)
		try:
			yield span
		finally:
			span.end()

	def score(self, name: str, value: float, comment: Optional[str] = None):
		"""Record an evaluation or milestone score on the current active trace."""
		current_trace = _current_trace_var.get()
		if not self.is_active or not current_trace:
			return
		try:
			current_trace.score(name=name, value=value, comment=comment)
		except Exception as exc:
			log.debug("Failed to record score %s: %s", name, exc)

	def flush(self):
		"""Flush any queued trace batches."""
		if not self.is_active or not self.client:
			return
		try:
			self.client.flush()
		except Exception as exc:
			log.debug("Failed to flush Langfuse client: %s", exc)
