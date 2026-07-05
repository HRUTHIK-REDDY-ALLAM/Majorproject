# Detective AI — Multi-Agent Forensic Reasoning System

> An uncertainty-aware, auditable multi-agent investigation pipeline that treats every conclusion as a hypothesis to be tested rather than a fact to be reported.

Detective AI ingests multi-modal evidence (camera sightings, access-control logs, witness statements), tracks multiple competing hypotheses in parallel, infers movement through camera blind spots with bounded uncertainty, subjects its leading conclusion to adversarial review by a dedicated critic agent, and produces a fully cited report — validated quantitatively against a purpose-built synthetic benchmark.

## Architecture

**6 specialized agents** orchestrated by LangGraph in an iterative loop:

| Agent | Role |
|---|---|
| **Orchestrator** | Lead investigator — routes tasks, manages the hypothesis tree |
| **Investigator** | Evidence gathering and cross-modal correlation |
| **Trajectory** | Blind-spot gap inference (Markov model over camera topology) |
| **Critic** | Adversarial review — structurally tasked with attacking the leading hypothesis |
| **Verifier** | Audit — every claim must trace to a cited evidence ID |
| **Reporter** | Assembles the final report with rejected alternatives and unresolved objections |

Two execution modes share the same core reasoning components (`HypothesisTracker`, `GapFiller`, `ConfidenceEngine`, `CounterfactualEngine`):

- **LLM mode** — the full LangGraph multi-agent pipeline (Groq / Llama 3.3 70B, free tier)
- **Offline mode** — a deterministic rule-based pipeline; no API calls, used for CI and calibration baselines

## Key Features

- **Multi-hypothesis tracking** with branching, pruning, and explicit rejection logging ("considered and rejected, because…")
- **Calibrated confidence** — ECE measured against benchmark ground truth; Platt scaling fitted from outcomes
- **Uncertainty-bounded gap inference** — inferred movement is always flagged distinctly from observed movement, with route distributions
- **Adversarial self-critique** as a first-class architectural role
- **Counterfactual exploration** — "What if evidence X were false?" re-runs hypothesis scoring with the evidence removed
- **Purpose-built synthetic benchmark** — 10+ scenarios, each with a ground-truth suspect path, camera blind spots, and planted false leads

## Measured Results

Latest benchmark runs on the 10-scenario suite (see `benchmarks/results/`):

| Metric | Offline pipeline (10 scenarios) | LLM pipeline (3-scenario sample) |
|---|---|---|
| Suspect identification accuracy | 100% | 100% |
| Timeline reconstruction (IoU) | 1.000 | 0.571 |
| Expected Calibration Error | 0.168 | 0.197 |
| Critic effectiveness (false leads flagged) | 100% | 67% |
| Mean time-to-resolution | <0.1 s | 67 s |

Reproduce with:

```bash
python -m benchmarks.runner --mode offline --count 10
python -m benchmarks.runner --mode llm --count 3      # needs GROQ_API_KEY
```

## Tech Stack

| Component | Technology |
|---|---|
| Agent orchestration | LangChain + LangGraph |
| LLM | Groq (Llama 3.3 70B) — free tier |
| Tool interface | Model Context Protocol (`python -m detective_ai.mcp`) |
| Database + vectors | PostgreSQL + pgvector (automatic SQLite fallback for dev/CI) |
| CV detection | YOLOv8n, CPU (`[cv]` extra) |
| Re-identification | MobileNetV2 embeddings (`[cv]` extra) |
| Text embeddings / RAG | sentence-transformers (`[embeddings]` extra) |
| API | FastAPI + Uvicorn |
| Experiment tracking | MLflow (`[tracking]` extra) |
| Deployment | Docker, docker-compose, Kubernetes manifests, GitHub Actions CI |

## Quick Start

### Local (no external services needed)

```bash
python -m venv .venv
.venv\Scripts\activate                  # Windows (source .venv/bin/activate on Unix)
pip install -e ".[dev]"

copy .env.example .env                  # optionally add your GROQ_API_KEY

pytest tests -q                         # 63 tests, all offline
python -m benchmarks.runner --mode offline --count 10

python -m uvicorn detective_ai.api.app:app --port 8000
```

If PostgreSQL is not running, storage automatically falls back to SQLite at `data/detective_ai.db`.

Visit **http://localhost:8000** for the dashboard, **http://localhost:8000/api/docs** for the API.

### Docker

```bash
GROQ_API_KEY=... docker compose up --build          # app + pgvector Postgres
docker compose --profile tracking up                # optional MLflow server on :5000
```

### Kubernetes

```bash
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/config.yaml        # set GROQ_API_KEY in the Secret first
kubectl apply -f k8s/postgres.yaml
kubectl apply -f k8s/app.yaml
```

## API Overview

| Endpoint | Purpose |
|---|---|
| `POST /api/v1/ingest/video` | Upload video for detection + ReID (`[cv]` extra) |
| `POST /api/v1/ingest/logs` | Ingest access-control logs (JSON/CSV) |
| `POST /api/v1/ingest/statements` | Ingest witness statements |
| `POST /api/v1/investigate/` | Start the multi-agent investigation |
| `GET /api/v1/investigate/{case_id}` | Poll investigation status |
| `GET /api/v1/report/{case_id}` | Retrieve the final cited report |
| `POST /api/v1/counterfactual/` | "What if evidence X were false?" |
| `GET /api/health` | Health probe |

## Project Structure

```
src/detective_ai/
├── core/           # Domain models, evidence graph, enums
├── ingestion/      # Video, log, and statement processing
├── cv/             # YOLOv8n detection + MobileNetV2 ReID (lazy-loaded)
├── agents/         # 6 LangGraph agents + prompts
├── hypothesis/     # Multi-hypothesis tracker, confidence engine, counterfactuals
├── trajectory/     # Camera topology, Markov model, blind-spot gap filler
├── pipeline/       # Deterministic offline investigation pipeline
├── storage/        # PostgreSQL + pgvector adapter (SQLite fallback)
├── mcp/            # Model Context Protocol server (evidence query tools)
└── api/            # FastAPI server + routes
benchmarks/         # Scenario generator, evaluator, runner
tests/              # Unit + integration tests (all offline)
k8s/                # Kubernetes manifests
```

## Evaluation Methodology

Each synthetic scenario has a known ground-truth suspect path, at least one deliberate camera blind spot, and at least one planted false lead. Reported metrics:

- **Timeline reconstruction accuracy** — IoU of predicted vs. ground-truth camera path
- **Suspect identification precision** and false-positive rate
- **Confidence calibration** — Expected Calibration Error + reliability diagrams (do 80%-confidence claims turn out correct ~80% of the time?)
- **Critic effectiveness** — proportion of planted false leads explicitly flagged
- **Mean time-to-resolution** per scenario

## Data & Ethics

All development, demonstration, and evaluation use **synthetic scenarios with constructed ground truth** — never footage of real, identifiable individuals. This is a design choice: controllable ground truth is what makes rigorous accuracy and calibration measurement possible.

## License

MIT