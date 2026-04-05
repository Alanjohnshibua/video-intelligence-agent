# Video Intelligence Agent System Architecture

## Executive View

Video Intelligence Agent is a hybrid computer-vision and language-reasoning system for offline CCTV analysis. The system does not send raw video directly to an LLM. Instead, it extracts structured evidence with deterministic CV modules first, then uses an LLM only for high-level reasoning and natural-language summarization.

## Production Pipeline

```text
Video Input
  -> Frame Extraction
  -> Motion Detection
  -> Scene and Person Detection
  -> Face Identification
  -> Multi-Object Tracking
  -> Event Extraction
  -> Event Logging + Clip Export
  -> Agent Reasoning
  -> Summary Generation
```

## Module Mapping

- `src/video_intelligence_agent/video_processing/`
  Frame extraction and ingestion wrappers.
- `src/video_intelligence_agent/detection/`
  Person detection and optional scene understanding.
- `src/video_intelligence_agent/tracking/`
  Lightweight tracking abstractions for track continuity.
- `src/video_intelligence_agent/cctv_pipeline/core/`
  Production pipeline orchestration, motion detection, tracking, recognition, and event logic.
- `src/video_intelligence_agent/cctv_pipeline/services/`
  Event logging and evidence clip generation.
- `src/video_intelligence_agent/agent/`
  Hybrid reasoning layer that prepares structured evidence for Gemini or other LLMs.
- `src/video_intelligence_agent/summarization/`
  Final summarization utilities.

## Design Philosophy

- Run cheap deterministic modules first.
- Preserve structured intermediate outputs.
- Use the LLM for reasoning, not perception.
- Keep every stage replaceable.
- Default to CPU-friendly components and graceful degradation.

## Why This Is More Than An API Wrapper

- Motion detection reduces unnecessary inference before any model call.
- YOLO-based person detection and tracking create measurable intermediate state.
- Event extraction is rule-based and auditable.
- Clips and JSON logs create operational artifacts.
- The LLM receives already-structured events instead of raw unfiltered frames.
