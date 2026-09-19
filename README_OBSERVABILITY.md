# 🔍 AI Voice Agent Observability with Open-Source Langfuse

This project includes end-to-end observability powered by the open-source version of **Langfuse** running locally via Docker.

It provides deep visibility into the AI Voice Agent's performance, safety guardrails, Groq LLM generations, tool executions, latencies, token consumption, and conversation session grouping.

---

## 🚀 Quick Start: Running Langfuse Locally with Docker

### 1. Ensure Docker Desktop is Running
Make sure Docker Desktop is running on your Mac (`open /Applications/Docker.app`).

### 2. Start the Langfuse Stack
From the project root, launch the self-contained Langfuse stack:
```bash
docker compose -f docker-compose.langfuse.yml up -d
```

This starts all necessary local services:
- **`langfuse-web`** on port `3000`: Web UI, Dashboard, and Ingestion API
- **`langfuse-worker`** on port `3030`: Asynchronous event processing
- **`clickhouse`** on ports `8123` & `9000`: Analytical storage for traces, generations, and token metrics
- **`postgres`** on port `5432`: Relational database for metadata and users
- **`redis`** on port `6379`: Job queue
- **`minio`** on port `9090`: Local object storage

### 3. Log In to the Langfuse Dashboard
Open your browser and navigate to:
👉 **[http://localhost:3000](http://localhost:3000)**

Use the pre-seeded administrator credentials:
- **Email:** `admin@brightdental.local`
- **Password:** `adminpassword123`

The default project **Bright Dental AI Voice Agent** is already pre-configured with the project keys used by the backend:
- **Public Key:** `pk-lf-local-voice-agent`
- **Secret Key:** `sk-lf-local-voice-agent`
- **Host:** `http://localhost:3000`

---

## 🛡️ Zero-Downtime Safe Mode (Offline Fallback)

If Docker is not running or Langfuse is stopped:
- The backend automatically executes a fast initial probe (<25ms) and activates **Safe Mode**.
- In Safe Mode, all tracing operations run as instantaneous local no-ops (0.1ms overhead, zero network retries).
- The Voice Agent continues running normally with **100% functionality and no latency penalty**.

---

## 📊 Metrics & Observability Features

### 1. 🤖 LLM Generations (`router_llm` & `responder_llm`)
- **Models Tracked:** `llama-3.3-70b-versatile`, `llama-3.1-8b-instant`, `openai/gpt-oss-20b`, and `whisper-large-v3-turbo`.
- **Token Consumption:** Prompt tokens, completion tokens, and total tokens tracked per turn and per conversation.
- **Latency:** Execution duration for routing tool selection vs. spoken reply generation.
- **Prompts & Completions:** Complete prompt history and structured JSON tool decisions.

### 2. 🛡️ Safety & Guardrail Monitoring
- **Input Guardrails (`guardrail:input`):**
  - Prompt injection attacks detected and blocked
  - Unauthorized medical diagnosis & prescription requests blocked
  - Out-of-domain and physical chore requests handled
  - Emergency medical triage alerts triggered
- **Output Guardrails (`guardrail:output`):**
  - System prompt leak redaction
  - Unauthorized prescription filters
- **Scoring:** Automated `guardrail_passed` metric (1.0 = passed, 0.0 = blocked).

### 3. 🛠️ Tool Execution Spans
- **`tool:calendar.check_availability`**: Available slot lookups and doctor scheduling checks.
- **`tool:calendar.select_slot`**: Slot reservations and appointment holds.
- **`tool:rag.query`**: Vector search queries against ChromaDB knowledge base.
- **`tool:payment.verify`**: Stripe booking fee verification and session status.
- **`tool:booking.cancel`**: Cancellation requests and slot releases.

### 4. 📞 Conversation Session Grouping
- Each phone call or WebSocket session maps to a persistent `session_id`.
- Multiple conversational turns are grouped under the same session thread in the Langfuse UI.
- Caller profile attributes (caller name, phone number, booking status) are linked to traces.
- **Milestone Scores:** `booking_milestone` tracks conversion funnels (0.3 = availability checked, 0.7 = payment link sent, 1.0 = booking confirmed).

---

## 🧪 Verifying Observability via Test Script

To verify that observability is properly recording turns and generations, run:
```bash
PYTHONPATH=backend ./.venv/bin/python backend/scripts/test_observability.py
```

You should see:
```text
============================================================
  Langfuse Observability & Metrics Verification Suite
============================================================

--- 1. Testing Safe Mode & Zero-Overhead Offline Fallback ---
✓ Offline probe finished safely in 14.2ms without blocking
✓ Complete observation turn completed in 0.11ms in safe mode

--- 2. Testing LLM Generation Metric Capture ---
✓ Router decision observed: action=rag.query
✓ Responder generation observed: reply snippet=...

--- 3. Testing End-to-End Agent Turn with Guardrail Interception ---
✓ Turn 1 (Legitimate query) traced successfully
✓ Turn 2 (Prompt Injection) safely blocked and scored: action=injection_blocked
✓ Turn 3 (Calendar query) traced successfully with session_id: ...
✓ Conversation session traced end-to-end with 3 turns

 All Langfuse Observability tests passed successfully!
============================================================
```

---

## 🛑 Managing the Docker Stack

### Check Container Status
```bash
docker compose -f docker-compose.langfuse.yml ps
```

### View Logs
```bash
docker compose -f docker-compose.langfuse.yml logs -f langfuse-web
```

### Stop Langfuse
```bash
docker compose -f docker-compose.langfuse.yml down
```

### Stop and Wipe Local Database Volumes (Fresh Reset)
```bash
docker compose -f docker-compose.langfuse.yml down -v
```
