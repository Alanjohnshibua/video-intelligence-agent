# Video Intelligence Agent System Architecture

## Executive View

Video Intelligence Agent is a hybrid CV + agentic reasoning system for CCTV analytics. The core principle is simple: perception happens locally first, and language reasoning happens later over structured evidence. This keeps the system inspectable, cost-aware, and production-friendly.

## End-To-End Pipeline

```text
Video File / Webcam
  -> Ingestion
  -> Preprocessing
  -> Person / Face Detection
  -> Tracking
  -> Event Engine
  -> Event JSON + Clips + Debug Frames
  -> Query Parsing + Time Filtering
  -> Local Event Retrieval
  -> Sarvam Reasoning
  -> Summary / Chat Answer
```

## Module Mapping

- `src/video_intelligence_agent/ingestion/`
  Video input adapters and replayable frame extraction.
- `src/video_intelligence_agent/preprocessing/`
  Motion detection and low-cost preprocessing before heavier inference.
- `src/video_intelligence_agent/detection/`
  Person detection and optional scene analysis interfaces.
- `src/video_intelligence_agent/tracking/`
  Track lifecycle management and continuity abstraction.
- `src/video_intelligence_agent/event_engine/`
  Rule-based event extraction for entry, exit, and loitering.
- `src/video_intelligence_agent/cctv_pipeline/`
  Production pipeline orchestration, config, services, clips, and logging.
- `src/video_intelligence_agent/agent/`
  Query parsing, time-window resolution, event retrieval, Sarvam client, and controller.
- `webcam_app/`
  Live webcam recognition demo and CLI enrollment workflow.

## Agent Query Flow

```text
User question
  -> QueryIntent parser
  -> Time window resolver
  -> Local event filtering
  -> Optional Sarvam call
  -> Final answer + matching clips
```

This local-first query path is what removes the "wrapper" feel. The LLM is not parsing raw video and is not asked to search the whole event store blindly.

## Design Principles

- Do cheap deterministic work before expensive model calls.
- Keep artifacts inspectable and reusable.
- Make CV perception independent from language reasoning.
- Support offline mode when the reasoning API is unavailable.
- Prefer CPU-friendly defaults with optional heavier upgrades.

## Operational Artifacts

The pipeline can produce:

- `events.json`
- `latest_analysis.json`
- `daily_summary.txt`
- event clips
- snapshots
- annotated debug frames

These outputs make the system useful even without the UI or the agent layer.
