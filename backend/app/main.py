import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .agent.graph import build_graph
from .api.audio import router as audio_router
from .api.profile import router as profile_router
from .api.webhooks import router as webhook_router
from .api.websocket import router as ws_router
from .config import get_settings
from .services.booking_store import BookingStore
from .services.call_queue import CallQueueManager
from .services.llm import LLMService
from .services.observability import ObservabilityService
from .services.rumik import RumikTTSService
from .services.sarvam import SarvamTTSService
from .services.vectorstore import VectorStore


logging.basicConfig(
	level=logging.INFO,
	format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
	settings = get_settings()
	app.state.settings = settings
	app.state.observability = ObservabilityService(settings)
	app.state.llm = LLMService(observability=app.state.observability) # Groq router + responder
	app.state.bookings = BookingStore(settings.DATABASE_PATH)
	app.state.vectors = VectorStore()            # Chroma + embeddings
	app.state.vectors.ensure_ingested()          # auto-ingest KB on first boot
	app.state.graph = build_graph(app)           # compiled LangGraph, built once
	app.state.call_queue = CallQueueManager(
		max_concurrent=settings.MAX_CONCURRENT_CALLS,
		max_duration_seconds=settings.CALL_MAX_DURATION_SECONDS,
	)
	app.state.sarvam = SarvamTTSService(
		api_key=settings.SARVAM_API_KEY,
		default_speaker=settings.SARVAM_TTS_SPEAKER,
		default_language=settings.SARVAM_TTS_LANGUAGE,
		model=settings.SARVAM_TTS_MODEL,
		default_pace=settings.SARVAM_TTS_PACE,
	)
	app.state.rumik = RumikTTSService(
		api_key=settings.RUMIK_API_KEY,
		model=settings.RUMIK_TTS_MODEL,
		default_speaker=settings.RUMIK_TTS_SPEAKER,
		default_description=settings.RUMIK_TTS_DESCRIPTION,
		temperature=settings.RUMIK_TTS_TEMPERATURE,
		default_pace=getattr(settings, "RUMIK_TTS_PACE", 1.15),
	)

	greeting = f"Namaste! {settings.CLINIC_NAME} mein aapka swagat hai. Main Niaa bol rahi hoon, main aapki kaise madad kar sakti hoon?"
	goodbye = "Thank you for calling Bright Dental Clinic. Have a lovely day!"
	warmup_phrases = [greeting, goodbye]
	import asyncio
	if app.state.sarvam.is_configured:
		asyncio.create_task(app.state.sarvam.warmup(warmup_phrases))
	if app.state.rumik.is_configured:
		asyncio.create_task(app.state.rumik.warmup(warmup_phrases))

	yield
	if hasattr(app.state, "sarvam") and app.state.sarvam:
		await app.state.sarvam.aclose()
	if hasattr(app.state, "rumik") and app.state.rumik:
		await app.state.rumik.aclose()
	if hasattr(app.state, "observability") and app.state.observability:
		app.state.observability.flush()


def create_app() -> FastAPI:
	settings = get_settings()
	app = FastAPI(
		title=f"{settings.CLINIC_NAME} — AI Voice Agent",
		version="1.0.0",
		lifespan=lifespan,
	)
	app.add_middleware(
		CORSMiddleware,
		allow_origins=["*"],
		allow_methods=["*"],
		allow_headers=["*"],
		allow_credentials=True,
	)
	app.include_router(ws_router)
	app.include_router(webhook_router)
	app.include_router(profile_router)
	app.include_router(audio_router)

	@app.get("/health")
	def health():
		obs_active = getattr(getattr(app.state, "observability", None), "is_active", False)
		return {
			"status": "ok",
			"clinic": settings.CLINIC_NAME,
			"observability": {
				"enabled": settings.LANGFUSE_ENABLED,
				"active": obs_active,
				"host": settings.LANGFUSE_HOST,
			},
		}

	# ── Static SPA frontend serving (for single-tunnel ngrok & mobile access) ──
	import os
	from fastapi.staticfiles import StaticFiles
	from fastapi.responses import FileResponse

	dist_candidates = [
		os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "frontend", "dist")),
		os.path.abspath(os.path.join(os.getcwd(), "frontend", "dist")),
	]
	dist_dir = next((p for p in dist_candidates if os.path.isdir(p)), None)
	if dist_dir and os.path.isfile(os.path.join(dist_dir, "index.html")):
		assets_dir = os.path.join(dist_dir, "assets")
		if os.path.isdir(assets_dir):
			app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

		@app.api_route("/{full_path:path}", methods=["GET", "HEAD"])
		async def serve_spa(full_path: str):
			file_path = os.path.join(dist_dir, full_path)
			if full_path and os.path.isfile(file_path):
				return FileResponse(file_path)
			return FileResponse(os.path.join(dist_dir, "index.html"))

	return app


app = create_app()
