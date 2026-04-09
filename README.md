# Video Intelligence Agent

Video Intelligence Agent is a hybrid computer-vision and AI-agent system for CCTV analytics, offline video review, and operator-facing event investigation. The system performs local perception first with OpenCV, YOLOv8, tracking, face recognition, and rule-based event extraction. An LLM is used only after that structured evidence exists, so the app behaves like a real AI pipeline instead of a thin API wrapper.

The current project includes:

- a production-style CCTV analysis pipeline
- a Streamlit operator app for browsing events and chatting over evidence
- a Sarvam-powered query agent for natural-language event investigation
- optional face-enrollment and live webcam demos

## Why This Feels Like A Real AI System

- CV-first, LLM-second architecture
- deterministic intermediate artifacts such as detections, tracks, events, clips, and summaries
- structured event logs before any model reasoning call
- modular pipeline stages with graceful failure handling
- deployable interfaces for offline batch processing and interactive review

## System Architecture

```text
Video Input / Webcam
        |
        v
Frame Extraction + Ingestion
        |
        v
Motion Detection / Preprocessing
        |
        v
Person Detection + Face Identification
        |
        v
Multi-Object Tracking
        |
        v
Event Engine
  - entering
  - exiting
  - loitering
  - repeated presence (extensible)
        |
        +--> JSON Event Log
        +--> Evidence Clips
        +--> Debug Frames
        |
        v
Agent Query Layer
  - query parser
  - time filter
  - local event retriever
  - Sarvam reasoning
        |
        v
Natural Language Summary / Chat Response
```

## Design Philosophy

- Use computer vision for perception and evidence extraction.
- Use the LLM for reasoning over structured evidence, not raw frames.
- Keep the baseline CPU-friendly and practical for local execution.
- Preserve auditable outputs such as JSON logs, clips, and summaries.
- Prefer modular stages that can be swapped independently.

## Current Pipeline

```text
Video Input
  -> Ingestion
  -> Preprocessing
  -> Detection
  -> Tracking
  -> Event Extraction
  -> Structured Storage
  -> Agent Reasoning
  -> Summary Generation
```

### Stage Breakdown

- `Ingestion`
  Reads frames from stored video files and preserves frame index and timestamp context.
- `Preprocessing`
  Filters inactive footage with motion detection before heavier inference.
- `Detection`
  Uses YOLOv8 Nano for people and the face-ID stack for identity attempts.
- `Tracking`
  Links detections over time with a lightweight tracker and a future upgrade path to DeepSORT or ByteTrack.
- `Event Engine`
  Converts track state into operator-friendly events such as entering, exiting, and loitering.
- `Structured Storage`
  Saves event JSON, evidence clips, and debug artifacts for later review.
- `Agent Reasoning`
  Parses user questions locally, filters matching events locally, and only then calls Sarvam when reasoning is needed.

## Professional Project Structure

```text
video-intelligence-agent/
|-- src/
|   `-- video_intelligence_agent/
|       |-- ingestion/
|       |-- preprocessing/
|       |-- detection/
|       |-- tracking/
|       |-- event_engine/
|       |-- agent/
|       |-- summarization/
|       |-- cctv_pipeline/
|       |-- engines/
|       `-- ...
|-- webcam_app/
|-- docs/
|-- models/
|-- logs/
|-- outputs/
|-- samples/
|-- app.py
|-- main.py
|-- config.yaml
`-- pyproject.toml
```

### Folder Responsibilities

- `src/video_intelligence_agent/ingestion/`
  Video-source adapters and replayable frame ingestion.
- `src/video_intelligence_agent/preprocessing/`
  Motion filtering and low-cost early transforms.
- `src/video_intelligence_agent/detection/`
  Person and scene detection interfaces.
- `src/video_intelligence_agent/tracking/`
  Tracking abstraction for multi-frame continuity.
- `src/video_intelligence_agent/event_engine/`
  Rule-based event extraction logic.
- `src/video_intelligence_agent/agent/`
  Query parser, time filter, local event retriever, Sarvam wrapper, and agent controller.
- `src/video_intelligence_agent/cctv_pipeline/`
  The production CCTV pipeline with config, logging, services, clips, and errors.
- `webcam_app/`
  Live webcam demo plus CLI-based face enrollment workflow.
- `outputs/`
  Generated analysis JSON, summaries, clips, snapshots, and debug frames.
- `logs/`
  Runtime log examples and deployment-oriented logging location.

## Tech Stack

### CV Stack

- OpenCV
- YOLOv8 Nano
- DeepFace-compatible face recognition stack
- rule-based event extraction
- lightweight IoU-based tracking fallback

### Agent Stack

- Sarvam AI for reasoning over filtered event data
- local rule-based query parsing
- local time-window resolution
- local event retrieval and deduplication before LLM use

### App / Engineering Stack

- Streamlit operator UI
- typed Python dataclasses and modular services
- JSON-based storage and evidence clip export
- config-driven runtime via `config.yaml`

## How The Agent Avoids Wrapper Feel

The event query flow is intentionally hybrid:

```text
User question
  -> QueryIntent parser
  -> Time window resolver
  -> Local event retrieval and filtering
  -> Optional Sarvam call
  -> Final answer with supporting clips
```

This means:

- the LLM does not parse the raw video
- the LLM does not scan the whole event database blindly
- structured evidence is filtered locally first
- the system can still answer without Sarvam in offline mode

## Sample Outputs

### Example Operator Log

```text
[10:32] Unknown person detected
[10:35] Person loitering near entrance
[10:40] Person exited
```

### Example Event JSON

```json
{
  "event_id": "event-00003",
  "person_id": "unknown_1",
  "action": "loitering",
  "start_time": "00:00:00.200",
  "end_time": "00:00:15.600",
  "duration_seconds": 15.4,
  "track_id": 1,
  "clip_path": "outputs/lobby_demo/clips/event-00003_track-0001.mp4"
}
```

### Example Generated Artifacts

- [outputs/lobby_demo/latest_analysis.json](/c:/Users/Alan_js/DEVELOPER/AI-KYRO/project/video-intelligence-agent/outputs/lobby_demo/latest_analysis.json)
- [outputs/lobby_demo/events.json](/c:/Users/Alan_js/DEVELOPER/AI-KYRO/project/video-intelligence-agent/outputs/lobby_demo/events.json)
- [outputs/lobby_demo/daily_summary.txt](/c:/Users/Alan_js/DEVELOPER/AI-KYRO/project/video-intelligence-agent/outputs/lobby_demo/daily_summary.txt)
- [outputs/lobby_demo/manifests/events.jsonl](/c:/Users/Alan_js/DEVELOPER/AI-KYRO/project/video-intelligence-agent/outputs/lobby_demo/manifests/events.jsonl)
- [logs/sample_runtime.log](/c:/Users/Alan_js/DEVELOPER/AI-KYRO/project/video-intelligence-agent/logs/sample_runtime.log)

### Debug Outputs

When debug mode is enabled, the pipeline also writes annotated frames under:

- `outputs/lobby_demo/debug/`

## Streamlit Operator App

The Streamlit UI is now an operator surface, not the intelligence layer itself.

Features:

- artifact dashboard
- event log browsing
- event filtering
- Sarvam-powered CCTV chat
- clip playback for matched events

Run it with:

```bash
python -m streamlit run app.py
```

Environment variable for Sarvam:

```bash
SARVAM_API_KEY=your_key_here
```

## Webcam Demo

The repo also includes a small live demo flow for face enrollment and webcam recognition:

Enroll a face:

```bash
python webcam_app/enroll_face.py --name Alan --image path/to/face.jpg
```

Run webcam recognition:

```bash
python webcam_app/main.py --device 0
```

## Installation

Base install:

```bash
pip install -e .
```

Development:

```bash
pip install -e ".[dev]"
```

Full pipeline plus UI and agent features:

```bash
pip install -e ".[full]"
```

Optional focused installs:

```bash
pip install -e ".[ui]"
pip install -e ".[agent]"
pip install -e ".[summary,scene]"
```

## Usage

Run the production CCTV pipeline:

```bash
python main.py --config config.yaml
```

Enable debug frames:

```bash
python main.py --config config.yaml --debug
```

Query stored events from the CLI:

```bash
python main.py --config config.yaml --query-action loitering
python main.py --config config.yaml --query-person unknown_3
```

## Real-World Use Cases

- CCTV review for offices and apartment entrances
- unknown-person investigation
- loitering and dwell-time analytics
- operator-friendly search over event history
- research demos for hybrid CV + LLM systems

## Limitations

- The default tracker is lightweight and may fragment IDs in crowded scenes.
- Face identification quality still depends on enrollment quality and backend stability.
- Event extraction is rule-based rather than learned action recognition.
- The main production path is currently offline video review, not full low-latency streaming.
- The Streamlit app is an operator console, not yet a multi-user production dashboard.

## Future Work

- multi-camera orchestration
- real-time streaming ingestion
- SQLite or PostgreSQL event storage
- FastAPI backend for programmatic access
- React or improved Streamlit dashboarding
- DeepSORT or ByteTrack integration
- repeated-presence analytics across longer time windows
- optimized CPU and edge-device model packaging

## Interview-Ready Explanations

### Explain The System In 60 Seconds

Video Intelligence Agent is a hybrid AI system for CCTV analytics. It first extracts structured evidence from video using motion detection, YOLO-based person detection, tracking, event extraction, and optional face recognition. Those results are saved as JSON logs, clips, and debug artifacts. Then an AI agent parses the user query, filters the event log locally, and uses Sarvam only for higher-level reasoning over the matched evidence. That separation makes the system more efficient, explainable, and production-oriented than sending raw video to an API.

### Why Not Rely Only On Sarvam?

Because perception and reasoning are different problems. Computer vision is better for frame-level tasks such as detection, tracking, and event extraction. If you push raw video directly into an LLM workflow, the system becomes more expensive, less observable, and harder to debug. Here the LLM only works on structured evidence after the heavy local processing is done.

### How Does The Pipeline Scale?

The architecture is stage-based, so scaling mostly means replacing the file-based ingestion layer with stream workers, keeping tracker state per camera, and storing events in a database or queue. Since the LLM is not on the frame-processing path, it does not bottleneck the perception pipeline.

### How Do You Detect Events?

Events are extracted by rules over track state. Entry and exit are inferred from movement between edge and center zones. Loitering is based on dwell time and low spatial drift. The same event engine can be extended with repeated-presence or zone-specific behavior rules without changing the rest of the pipeline.

## Additional Documentation

- [docs/SYSTEM_ARCHITECTURE.md](/c:/Users/Alan_js/DEVELOPER/AI-KYRO/project/video-intelligence-agent/docs/SYSTEM_ARCHITECTURE.md)
- [docs/INTERVIEW_GUIDE.md](/c:/Users/Alan_js/DEVELOPER/AI-KYRO/project/video-intelligence-agent/docs/INTERVIEW_GUIDE.md)
- [PROJECT_REVIEW.md](/c:/Users/Alan_js/DEVELOPER/AI-KYRO/project/video-intelligence-agent/PROJECT_REVIEW.md)
- [CCTV_AGENT_GUIDE.md](/c:/Users/Alan_js/DEVELOPER/AI-KYRO/project/video-intelligence-agent/CCTV_AGENT_GUIDE.md)

## Testing

```bash
python -m pytest -q
```
