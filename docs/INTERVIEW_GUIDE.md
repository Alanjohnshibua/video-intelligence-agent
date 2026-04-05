# Interview Guide

## Explain The System In 60 Seconds

Video Intelligence Agent is a hybrid CCTV analysis system. It ingests recorded video, filters inactive frames with motion detection, detects people with YOLO, tracks them across time, optionally identifies faces, and converts tracks into events like entering, exiting, and loitering. Those events are stored as JSON and evidence clips, and then an LLM such as Gemini is used only for higher-level reasoning and summary generation. That design keeps the system lightweight, explainable, and much more production-oriented than sending raw video directly to an API.

## Why Combine CV And LLM?

Computer vision is better for deterministic perception tasks such as detection, tracking, and event extraction. LLMs are better for reasoning over structured outputs, answering operator questions, and generating concise reports. Combining both gives you lower cost, better observability, and clearer failure boundaries than using an LLM for everything.

## How Do You Handle Real-Time Video?

The current repo is optimized for pre-recorded video, but the architecture already separates ingestion, detection, tracking, and reasoning. To support real-time video, I would replace the file reader with a stream reader, process frames in batches or micro-batches, maintain per-camera worker state, and push events into a queue or database for downstream consumers.

## What Makes This Production-Minded?

- Modular pipeline stages with clean handoffs
- Centralized logging and error handling
- Event persistence and evidence clips
- Hybrid CV + LLM design instead of prompt-only inference
- Clear upgrade path for multi-camera and real-time deployments

## Strong Follow-Up Improvements

- Multi-camera orchestration
- SQLite or PostgreSQL event storage
- FastAPI backend plus React dashboard
- ByteTrack or DeepSORT for stronger tracking
- Quantized models and frame skipping for low-resource hardware
