# Video Intelligence Agent

Video Intelligence Agent is a production-minded hybrid AI system for CCTV and offline video analytics. Instead of sending raw footage directly to an LLM, it first extracts structured evidence with computer vision modules such as motion detection, YOLO-based person detection, face identification, tracking, event extraction, and clip logging. A language model such as Gemini is then used only for reasoning over those structured events and generating operator-facing summaries.

This makes the project much more than a thin API wrapper. It behaves like a real AI pipeline with measurable intermediate outputs, auditable decisions, graceful failure handling, and deployable artifacts.

## Why This Project Feels Like Real AI Engineering

- Hybrid CV + LLM architecture instead of prompt-only inference
- Deterministic pipeline stages with inspectable outputs
- Event logs, clips, and summaries that look like operational artifacts
- Error handling and degradation paths for model failures
- Clear separation between perception, reasoning, and persistence

## System Architecture

```text
                    +----------------------+
                    |  Video File / CCTV   |
                    +----------+-----------+
                               |
                               v
                    +----------------------+
                    |   Frame Extraction   |
                    +----------+-----------+
                               |
                               v
                    +----------------------+
                    |   Motion Detection   |
                    +----------+-----------+
                               |
                +--------------+---------------+
                |                              |
                v                              v
     +----------------------+      +----------------------+
     | Scene / Person Det.  |      | Face Identification  |
     |   YOLOv8 Nano        |      | DeepFace / ArcFace   |
     +----------+-----------+      +----------+-----------+
                |                              |
                +--------------+---------------+
                               |
                               v
                    +----------------------+
                    |   Multi-Object       |
                    |     Tracking         |
                    +----------+-----------+
                               |
                               v
                    +----------------------+
                    |   Event Extraction   |
                    | enter/exit/loiter    |
                    +----+------------+----+
                         |            |
                         v            v
              +----------------+   +------------------+
              | JSON Event Log |   | Clip Extraction  |
              +--------+-------+   +---------+--------+
                       \                 /
                        \               /
                         v             v
                    +----------------------+
                    | AI Agent Reasoning   |
                    | Gemini / LLM Layer   |
                    +----------+-----------+
                               |
                               v
                    +----------------------+
                    | Summary Generation   |
                    +----------------------+
```

## Design Philosophy

- Use computer vision for perception, not an LLM.
- Send the LLM structured evidence, not raw frames.
- Keep the system lightweight enough for local and CPU-first demos.
- Favor observability and debugability over flashy black-box behavior.
- Make every module replaceable so the system can evolve incrementally.

## Current Weaknesses Found In The Original Direction

Before the current refactor, the project risked feeling like a demo because:

- the narrative was split between face recognition, CCTV analysis, and video summarization
- the LLM story was stronger than the engineering story
- there was no obvious production pipeline visible at the repo root
- intermediate artifacts were under-explained
- the system design was not framed as a hybrid AI pipeline

The refactor addresses that by centering the repo around a structured CV pipeline with an optional reasoning layer on top.

## Production Pipeline

```text
Video Input
  -> Frame Extraction
  -> Motion Detection
  -> Scene Detection
  -> Person / Face Detection
  -> Tracking
  -> Event Extraction
  -> Evidence Logging
  -> AI Reasoning
  -> Summary Generation
```

### Stage Breakdown

- `Frame Extraction`
  Reads offline video files, keeps frame indices and timestamps, and exposes deterministic replay.
- `Motion Detection`
  Filters inactive footage early so expensive inference is applied only where needed.
- `Scene / Person Detection`
  Uses YOLOv8 Nano for people and optional scene understanding.
- `Face Identification`
  Attempts known-vs-unknown identity matching using the local face-ID stack.
- `Tracking`
  Maintains track continuity with a lightweight tracker and a clean upgrade path to DeepSORT or ByteTrack.
- `Event Extraction`
  Converts track state into events such as entering, exiting, and loitering.
- `Evidence Logging`
  Stores JSON event records plus event clips for review and auditability.
- `AI Reasoning`
  Uses an LLM only for operator Q&A, explanation, and report generation.

## Project Structure

```text
project/
│── src/
│   └── video_intelligence_agent/
│       ├── video_processing/
│       ├── detection/
│       ├── tracking/
│       ├── agent/
│       ├── summarization/
│       ├── cctv_pipeline/
│       ├── cctv/
│       ├── surveillance/
│       └── engines/
│── docs/
│── outputs/
│── samples/
│── app.py
│── main.py
│── config.yaml
│── pyproject.toml
```

### Folder Responsibilities

- `src/video_intelligence_agent/video_processing/`
  Frame extraction and ingestion wrappers.
- `src/video_intelligence_agent/detection/`
  Person detection and optional scene analysis interfaces.
- `src/video_intelligence_agent/tracking/`
  Recruiter-friendly tracking entrypoint and adapters.
- `src/video_intelligence_agent/agent/`
  Hybrid reasoning layer for Gemini or other LLMs.
- `src/video_intelligence_agent/summarization/`
  Summary generation modules.
- `src/video_intelligence_agent/cctv_pipeline/`
  Production CCTV pipeline with logging, errors, events, and clips.
- `docs/`
  Architecture notes, interview guide, and system design documents.

## Tech Stack

### Computer Vision

- OpenCV for video IO, motion detection, frame annotation, and clip writing
- YOLOv8 Nano for person detection
- DeepFace-compatible face recognition stack
- Rule-based event extraction for explainable behavior
- Lightweight IoU tracker today, upgradeable to SORT, DeepSORT, or ByteTrack

### AI / Agent Layer

- Gemini-compatible or other LLM responder for reasoning and report generation
- Structured prompts built from extracted events rather than raw footage

### Engineering Stack

- Python
- Typed dataclasses and modular services
- JSON event persistence
- Streamlit operator demo via `app.py`
- Config-driven runtime via `config.yaml`

## Hybrid CV + LLM Strategy

This project avoids the usual “API wrapper” feel by treating the LLM as a reasoning layer, not as the perception engine.

### What Happens Before Gemini

- frames are extracted from video
- inactive footage is filtered with motion detection
- people are detected with YOLO
- tracks are maintained over time
- identity is attempted locally
- events are extracted and logged with clips

### What Gemini Should Do

- explain what happened in plain English
- answer operator questions over structured events
- generate executive summaries
- provide investigative reasoning over time windows and event sequences

That division gives lower cost, clearer debugging, and better system credibility.

## Sample Outputs

### Event Log Style

```text
[10:32] Unknown person detected
[10:35] Person stayed near entrance (loitering)
[10:40] Person exited
```

### Real Example From This Repo

```text
[10:32:04 AM] ALERT=medium CATEGORY=unknown_person_detected CAMERA=lobby-cam-01 EVENT=event-0003 PERSON=Unknown Person
[10:35:18 AM] ALERT=high CATEGORY=suspicious_presence CAMERA=lobby-cam-01 EVENT=event-0004 DETAIL=Unknown Person loitering near entrance for 21.4s
[10:41:52 AM] ALERT=low CATEGORY=entry CAMERA=lobby-cam-01 EVENT=event-0005 PERSON=Alan DETAIL=Known employee entered north door
```

### Event JSON Example

```json
{
  "person_id": "unknown_1",
  "action": "loitering",
  "start_time": "00:00:00.200",
  "end_time": "00:00:15.600",
  "duration_seconds": 15.4,
  "track_id": 1
}
```

### Available Generated Artifacts

- [outputs/lobby_demo/latest_analysis.json](/c:/Users/Alan_js/DEVELOPER/AI-KYRO/project/video-intelligence-agent/outputs/lobby_demo/latest_analysis.json)
- [outputs/lobby_demo/events.json](/c:/Users/Alan_js/DEVELOPER/AI-KYRO/project/video-intelligence-agent/outputs/lobby_demo/events.json)
- [outputs/lobby_demo/daily_summary.txt](/c:/Users/Alan_js/DEVELOPER/AI-KYRO/project/video-intelligence-agent/outputs/lobby_demo/daily_summary.txt)
- [outputs/examples/event_log.txt](/c:/Users/Alan_js/DEVELOPER/AI-KYRO/project/video-intelligence-agent/outputs/examples/event_log.txt)

## Use Cases

- CCTV footage review for offices and small facilities
- Unknown-person detection around entrances
- Occupancy and movement analytics
- Long-form video summarization for security and operations teams
- Research demos for hybrid AI system design

## Streamlit App

The repo includes a lightweight Streamlit operator shell:

```bash
streamlit run app.py
```

This is intentionally positioned as an operator interface on top of the real pipeline, not as the core intelligence layer.

## Quick Start

Install the base project:

```bash
pip install -e .
```

Install development tools:

```bash
pip install -e ".[dev]"
```

Install the full CV stack:

```bash
pip install -e ".[full]"
```

Run the production pipeline:

```bash
python main.py --config config.yaml
```

Enable debug-mode artifacts:

```bash
python main.py --config config.yaml --debug
```

Query stored events:

```bash
python main.py --config config.yaml --query-action loitering
```

## Engineering Improvements Added

- Modular `cctv_pipeline` package with centralized orchestration
- Stronger logging and error-handling boundaries
- Clip extraction for key events
- JSON event storage and query support
- Recruiter-facing package structure for processing, detection, tracking, agent, and summarization
- Hybrid reasoning module that prepares structured evidence for Gemini

## Limitations

- The default tracker is lightweight and can fragment IDs in crowded scenes.
- Face identification still depends on local enrollment quality and backend stability.
- Event logic is rule-based rather than learned action recognition.
- Current optimization target is offline review, not high-throughput real-time streaming.

## Future Work

- Multi-camera orchestration and cross-camera identity handoff
- Real-time streaming pipeline with per-camera workers
- SQLite or PostgreSQL event storage
- FastAPI or Flask backend with React dashboard
- ByteTrack or DeepSORT integration
- Quantized models and frame-skipping profiles for edge devices
- Better face-recognition backend using InsightFace

## Interview-Ready Explanations

### Explain The System In 60 Seconds

Video Intelligence Agent is a hybrid AI system for CCTV video analysis. It first extracts structured evidence from video using motion detection, YOLO-based person detection, tracking, event extraction, and optional face identification. Those outputs are saved as JSON logs and clips. Only after that does an LLM such as Gemini reason over the structured events to answer questions and generate summaries. That separation makes the system more efficient, explainable, and production-oriented than sending raw video straight to an API.

### Why Did You Combine CV And LLM?

CV is better for deterministic perception tasks like detection, tracking, and event extraction. LLMs are better for reasoning, summarization, and conversational explanations. Combining both gives you lower cost, clearer observability, and better engineering boundaries.

### How Do You Handle Real-Time Video?

The current repo is optimized for pre-recorded video, but the architecture is already modular. To support real-time streams, the ingestion layer can be swapped for a stream reader, tracking state can be kept per camera, and event outputs can be pushed to a queue or database without redesigning the entire system.

## Additional Docs

- [docs/SYSTEM_ARCHITECTURE.md](/c:/Users/Alan_js/DEVELOPER/AI-KYRO/project/video-intelligence-agent/docs/SYSTEM_ARCHITECTURE.md)
- [docs/INTERVIEW_GUIDE.md](/c:/Users/Alan_js/DEVELOPER/AI-KYRO/project/video-intelligence-agent/docs/INTERVIEW_GUIDE.md)
- [PROJECT_REVIEW.md](/c:/Users/Alan_js/DEVELOPER/AI-KYRO/project/video-intelligence-agent/PROJECT_REVIEW.md)
- [CCTV_AGENT_GUIDE.md](/c:/Users/Alan_js/DEVELOPER/AI-KYRO/project/video-intelligence-agent/CCTV_AGENT_GUIDE.md)

## Testing

```bash
python -m pytest -q
```
