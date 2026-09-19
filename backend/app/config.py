from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
	model_config = SettingsConfigDict(
		env_file=(".env", "backend/.env"), env_file_encoding="utf-8", extra="ignore"
	)

	# ── Core ─────────────────────────────────────────────
	CLINIC_NAME: str = "Bright Dental Clinic"
	ENV: str = "development"
	HOST: str = "0.0.0.0"
	PORT: int = 8000
	FRONTEND_URL: str = "http://localhost:5173"
	BACKEND_PUBLIC_URL: str = "http://localhost:8000"

	# ── LLM Engine Switch ────────────────────────────────
	PRIMARY_LLM_PROVIDER: str = "gemini"  # "gemini" (primary while Groq free tier quota is in effect) or "groq"

	# ── Groq — models ────────────────────────────────────
	GROQ_API_KEY: str = ""
	GROQ_ROUTER_MODEL: str = "qwen/qwen3.8-27b"   # tool selection (ultra-fast, zero reasoning tokens)
	GROQ_RESPONDER_MODEL: str = "qwen/qwen3.8-27b"    # spoken replies (crisp, zero reasoning tokens)
	GROQ_CLASSIFIER_MODEL: str = "qwen/qwen3.8-27b"  # turn/intent classification

	# ── Gemini — models & failover ───────────────────────
	GEMINI_API_KEY: str = ""
	GEMINI_MODEL: str = "gemini-2.5-flash"  # Main agent (router and spoken responder)
	GEMINI_CLASSIFIER_MODEL: str = "gemini-2.5-flash"  # Initial LLM layer (turn classifier & gap filler)

	# ── RAG / embeddings ─────────────────────────────────
	EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
	CHROMA_DIR: str = "./data/chroma"
	CHROMA_COLLECTION: str = "clinic_kb"
	KB_DIR: str = "./app/knowledge"
	KB_FAQS_PATH: str = "./app/knowledge/clinic_faqs.md"
	RAG_TOP_K: int = 4
	USE_LANGEXTRACT: bool = True
	LANGEXTRACT_MODEL: str = "gemini-2.5-flash"

	# ── Google Calendar ──────────────────────────────────
	GOOGLE_CALENDAR_ID: str = "primary"
	GOOGLE_TOKEN_FILE: str = "token.json"
	GOOGLE_CREDENTIALS_FILE: str = "credentials.json"
	GOOGLE_SERVICE_ACCOUNT_FILE: str = ""
	GOOGLE_SERVICE_ACCOUNT_JSON: str = ""
	CLINIC_TIMEZONE: str = "Asia/Kolkata"
	CLINIC_OPEN_HOUR: int = 10
	CLINIC_CLOSE_HOUR: int = 19
	SLOT_MINUTES: int = 30

	# ── Stripe ───────────────────────────────────────────
	STRIPE_SECRET_KEY: str = ""
	STRIPE_WEBHOOK_SECRET: str = ""
	STRIPE_CURRENCY: str = "inr"
	BOOKING_FEE_RUPEES: int = 500
	PAYMENT_REQUIRED_FOR_BOOKING: bool = True

	# ── Sessions / data ──────────────────────────────────
	MAX_CONCURRENT_CALLS: int = 5
	CALL_MAX_DURATION_SECONDS: int = 180
	SESSION_TTL_MINUTES: int = 30
	DATABASE_PATH: str = "./data/bookings.db"

	# ── Voice Engine Switch (Config / Code Switch) ───────
	TTS_PROVIDER: str = "sarvam"  # "sarvam" (Bulbul v3, primary), "rumik" (Silk ₹0.4/min Aisha), or "browser" (dev / ₹0 cost)
	STT_PROVIDER: str = "sarvam"  # "browser" (client Web Speech) or "sarvam" (real-time streaming STT)
	SARVAM_API_KEY: str = ""
	SARVAM_TTS_MODEL: str = "bulbul:v3"
	SARVAM_TTS_SPEAKER: str = "simran"
	SARVAM_TTS_PACE: float = 1.0
	SARVAM_TTS_LANGUAGE: str = "hi-IN"
	SARVAM_STT_MODEL: str = "saaras:v3-realtime"
	SARVAM_STT_LANGUAGE: str = "hi-IN"
	SARVAM_STT_ENDPOINT: str = "wss://api.sarvam.ai/speech-to-text-realtime/ws"
	SARVAM_STT_SILENCE_MS: int = 750
	SARVAM_STT_SAMPLE_RATE: int = 16000

	# ── Rumik AI Silk TTS ────────────────────────────────
	RUMIK_API_KEY: str = ""
	RUMIK_TTS_MODEL: str = "mulberry"  # "mulberry" (faster, description-driven) or "muga" (tone-tagged)
	RUMIK_TTS_SPEAKER: str = "aisha"  # female: aisha, siya, emma, mia, sophia, ava, ira, zoya; male: lucas, noah, theo, adam
	RUMIK_TTS_PACE: float = 1.15
	RUMIK_TTS_DESCRIPTION: str = "a female 20s indian receptionist voice, crisp clear articulation, brisk and lively conversational pace, warm and professional"
	RUMIK_TTS_TEMPERATURE: float = 0.6

	# ── Dev mode: mock Calendar + Stripe when keys are missing ──
	DEV_MOCK_EXTERNALS: bool = True

	# ── Langfuse Observability ───────────────────────────
	LANGFUSE_ENABLED: bool = True
	LANGFUSE_PUBLIC_KEY: str = "pk-lf-local-voice-agent"
	LANGFUSE_SECRET_KEY: str = "sk-lf-local-voice-agent"
	LANGFUSE_HOST: str = "http://localhost:3000"

	@property
	def langfuse_configured(self) -> bool:
		return bool(self.LANGFUSE_ENABLED and self.LANGFUSE_PUBLIC_KEY and self.LANGFUSE_SECRET_KEY)

	@property
	def groq_configured(self) -> bool:
		return bool(self.GROQ_API_KEY)

	@property
	def stripe_configured(self) -> bool:
		return bool(self.STRIPE_SECRET_KEY)

	@property
	def google_configured(self) -> bool:
		import os
		token_exists = bool(
			self.GOOGLE_TOKEN_FILE
			and (
				os.path.exists(self.GOOGLE_TOKEN_FILE)
				or os.path.exists(os.path.join("backend", self.GOOGLE_TOKEN_FILE))
			)
		)
		return bool(
			token_exists
			or self.GOOGLE_SERVICE_ACCOUNT_FILE
			or self.GOOGLE_SERVICE_ACCOUNT_JSON
		)

	@property
	def sarvam_configured(self) -> bool:
		return bool(self.SARVAM_API_KEY and self.SARVAM_API_KEY.strip())

	@property
	def rumik_configured(self) -> bool:
		return bool(self.RUMIK_API_KEY and self.RUMIK_API_KEY.strip())


@lru_cache
def get_settings() -> Settings:
	return Settings()
