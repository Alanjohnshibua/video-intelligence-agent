# Video Intelligence Agent

Video Intelligence Agent is a lightweight AI CCTV Surveillance Agent for offline video analysis. It detects faces, recognizes known vs unknown individuals, tracks people across frames, classifies security-relevant events, saves clips and snapshots, and generates daily summaries that are easy to review in an interview or demo.

The repository package name is still `video_intelligence_agent`, but the project is now framed as a production-minded surveillance system rather than a classroom-only face recognition script.

## Why This Project Stands Out

- Real pipeline, not just model inference: `video -> motion filtering -> face analysis -> tracking -> event decisions -> storage -> summary`
- CPU-friendly defaults: designed for offline review on modest hardware
- Recruiter-friendly artifacts: YAML config, production-style entrypoint, event logs, saved evidence, and summary reports
- Clear upgrade path: multi-camera orchestration, action detection, dashboarding, and database-backed audit trails

## System Architecture

```text
Camera / Video File
        |
        v
Video Ingestion
        |
        v
Motion Filtering
        |
        v
Face Detection
        |
        v
Face Recognition
        |
        v
Tracking
        |
        v
Event Classification + Alerting
        |
        +--> Clip / Snapshot Extraction
        |
        +--> JSONL Event Logging
        |
        v
Daily Summary Generation
```

## Refactored Project Structure

```text
project/
|-- src/
|   `-- video_intelligence_agent/
|       |-- surveillance/
|       |   |-- ingestion/
|       |   |-- detection/
|       |   |-- recognition/
|       |   |-- tracking/
|       |   |-- events/
|       |   |-- logging/
|       |   |-- summary/
|       |   |-- pipeline/
|       |   `-- config.py
|       |-- cctv/
|       |-- engines/
|       |-- core.py
|       `-- cli.py
|-- data/
|-- outputs/
|   `-- examples/
|-- tests/
|-- main.py
`-- config.yaml
```

## What Each Folder Does

- `src/video_intelligence_agent/surveillance/ingestion/`: video readers and camera/file input adapters
- `src/video_intelligence_agent/surveillance/detection/`: face detection stage interfaces and detection outputs
- `src/video_intelligence_agent/surveillance/recognition/`: known-vs-unknown identity resolution
- `src/video_intelligence_agent/surveillance/tracking/`: per-person track management across frames
- `src/video_intelligence_agent/surveillance/events/`: event rules, alert severity, and action labels
- `src/video_intelligence_agent/surveillance/logging/`: evidence persistence, event manifests, snapshots, and clips
- `src/video_intelligence_agent/surveillance/summary/`: daily reporting and operator-facing summaries
- `src/video_intelligence_agent/surveillance/pipeline/`: high-level surveillance runner
- `data/`: embedding database and local runtime artifacts
- `outputs/`: generated clips, unknown face crops, summaries, and example deliverables
- `tests/`: unit tests for the face pipeline, CCTV agent, and runtime config loader

## Tech Stack

- Python 3.10+
- NumPy
- OpenCV for video ingestion, motion filtering, and clip writing
- DeepFace-compatible backends for RetinaFace + ArcFace
- JSONL manifests for lightweight audit logging
- Optional Faster-Whisper and LLM summarization modules for richer video review
- Optional YOLOv8 Nano visual scene analysis from Hugging Face for multimodal summaries

## GPU Acceleration

- Face detection and recognition can use your TensorFlow GPU runtime automatically when DeepFace is configured with GPU support.
- The surveillance config now includes `prefer_gpu: true` and `tensorflow_memory_growth: true` to make GPU use the default while avoiding aggressive TensorFlow memory reservation.
- The video summarizer now defaults to `--whisper-device auto`, which selects `cuda` automatically when PyTorch detects a GPU.
- The local Transformers summarizer backend already prefers CUDA and 4-bit quantization when available.

Native Windows note:

- TensorFlow GPU acceleration is not supported on native Windows for TensorFlow 2.11+.
- On native Windows, the DeepFace surveillance pipeline will typically run on CPU even if your laptop has an NVIDIA GPU.
- For true TensorFlow GPU acceleration in the surveillance pipeline, use WSL2/Linux or a DirectML-compatible stack.
- The summarizer can still benefit from GPU through PyTorch when your CUDA environment is configured correctly.

What stays CPU-bound:

- OpenCV video decoding in this repo
- Motion filtering heuristics
- Lightweight event rules and JSON logging

## Recommended Models

### Face Detection

- `RetinaFace`: strongest default here because it handles varied pose and partial occlusion better than very lightweight Haar-style detectors
- `MTCNN`: good CPU fallback when absolute speed matters more than edge-case robustness

### Face Recognition

- `ArcFace`: strong embedding quality, mature ecosystem, and reliable performance for small-to-medium identity galleries
- `InsightFace`: good production choice when you want a cleaner deployment story around modern face-recognition tooling

### Tracking

- `SORT`: best lightweight baseline for CPU-only pipelines and demos
- `DeepSORT`: better when identity consistency matters more than raw simplicity

Why these choices:

- They are common enough that interviewers recognize them immediately
- They offer a realistic trade-off between quality and deployment cost
- They fit a modular architecture where each stage can be swapped later

## Real-World Features Included

- Unknown person alerting with severity labels
- Event classification for entry, exit, interaction, loitering, and suspicious presence
- Unknown face snapshot extraction
- Event clip extraction
- Timestamped JSONL event manifests
- Human-readable daily summary output

## Quick Start

Install the base project for tests and local development:

```bash
pip install -e .[dev]
```

Install transcript + visual-summary extras:

```bash
pip install -e .[summary,scene]
```

Run the recruiter-friendly surveillance entrypoint:

```bash
python main.py --config config.yaml
```

If your GPU stack is installed correctly, the face pipeline will prefer GPU automatically.

Override the video path at runtime:

```bash
python main.py --config config.yaml --video samples/lobby_demo.mp4
```

Generate a multimodal summary with YOLOv8 Nano scene analysis plus Qwen:

```bash
video-intelligence-agent summarize-video --video samples/demo.mp4 --no-person-id --scene-analysis
```

The run writes:

- `outputs/.../latest_analysis.json`
- `outputs/.../daily_summary.txt`
- `data/cctv_agent/manifests/events.jsonl` or your configured output manifest

## Sample Outputs

### Example Log Lines

```text
[10:32:04 AM] Unknown person detected near lobby entrance
[10:35:18 AM] Unknown person stayed near entrance for 21.4s (loitering)
[10:41:52 AM] Alan entered through north door
[10:44:07 AM] Alan exited through north door
```

### Example Event Manifest Snippet

```json
{
  "event_id": "event-0004",
  "event_category": "suspicious_presence",
  "alert_level": "high",
  "alert_reasons": [
    "unknown person detected",
    "loitering behavior detected"
  ],
  "start_timestamp": "2026-04-04T10:35:18",
  "end_timestamp": "2026-04-04T10:35:39"
}
```

### Example Daily Summary

See [outputs/examples/daily_summary.txt](/c:/Users/Alan_js/DEVELOPER/AI-KYRO/project/video-intelligence-agent/outputs/examples/daily_summary.txt) and [outputs/examples/event_log.txt](/c:/Users/Alan_js/DEVELOPER/AI-KYRO/project/video-intelligence-agent/outputs/examples/event_log.txt).

## Screenshots

Placeholders for portfolio or README assets:

- `docs/screenshots/dashboard-overview.png`
- `docs/screenshots/unknown-person-alert.png`
- `docs/screenshots/event-timeline.png`
- `docs/screenshots/summary-report.png`

## Use Cases

- Office entrance monitoring
- Visitor review for small businesses
- Warehouse after-hours surveillance
- Hostel or apartment lobby event review
- Smart evidence extraction from long CCTV footage

## Design Decisions

- Motion-first filtering reduces wasted compute on inactive footage
- Face recognition is separated from tracking so each stage can evolve independently
- Rule-based event classification keeps the system transparent and CPU-friendly
- JSONL logs are simple, grep-able, and easy to upgrade to SQLite later
- The `surveillance` package provides a cleaner architecture without breaking older modules

## Interview Talking Points

- The project optimizes for practical offline surveillance review, not just per-frame accuracy
- I kept the default pipeline modular so detection, recognition, and tracking can be swapped independently
- I added event severity and timestamp normalization because operators need actionable logs, not raw model outputs
- I kept the baseline CPU-friendly and reserved heavier models for clearly defined future upgrades

## Limitations

- The default tracker is intentionally lightweight and can drift in crowded scenes
- Face recognition quality still depends on enrollment image quality and lighting
- The current event classifier is heuristic, not learned action recognition
- This is best suited to single-camera offline review today

## Future Improvements

- YOLO-based person and action detection
- DeepSORT or ByteTrack integration for denser scenes
- SQLite or PostgreSQL event storage
- Flask/FastAPI backend with React dashboard
- Multi-camera correlation and handoff
- GPU acceleration profiles
- Webhook or email notifications for critical alerts

## Additional Documentation

- [PROJECT_REVIEW.md](/c:/Users/Alan_js/DEVELOPER/AI-KYRO/project/video-intelligence-agent/PROJECT_REVIEW.md)
- [CCTV_AGENT_GUIDE.md](/c:/Users/Alan_js/DEVELOPER/AI-KYRO/project/video-intelligence-agent/CCTV_AGENT_GUIDE.md)
- [VIDEO_SUMMARIZER_GUIDE.md](/c:/Users/Alan_js/DEVELOPER/AI-KYRO/project/video-intelligence-agent/VIDEO_SUMMARIZER_GUIDE.md)

## Testing

```bash
python -m compileall src tests main.py
python -m pytest -q
```
