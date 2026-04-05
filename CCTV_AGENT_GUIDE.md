# Intelligent CCTV Agent Guide

This project now includes a modular CCTV analysis agent built for pre-recorded footage.

The design is:

- local-first for vision processing
- lightweight enough for student demos
- modular enough to look like a real production pipeline

## What It Does

The agent processes video in this order:

1. read video frames
2. detect motion
3. skip inactive footage
4. identify people using a face-recognition module
5. track simple actions with lightweight heuristics
6. save only relevant clips and unknown-face snapshots
7. generate a concise daily summary
8. answer natural-language questions over the extracted events

## Text Architecture Diagram

```text
                 +----------------------+
                 |  Pre-recorded Video  |
                 +----------+-----------+
                            |
                            v
                 +----------------------+
                 |  Video Ingestion     |
                 |  - frame sampling    |
                 |  - metadata loading  |
                 +----------+-----------+
                            |
                            v
                 +----------------------+
                 |  Motion Detection    |
                 |  - frame differencing|
                 |  - inactive filtering|
                 +----------+-----------+
                            |
                  no motion |         motion / activity
                            |                |
                            |                v
                            |     +----------------------+
                            |     | Face Recognition     |
                            |     | - known / unknown    |
                            |     | - human presence     |
                            |     +----------+-----------+
                            |                |
                            |                v
                            |     +----------------------+
                            |     | Tracking + Actions   |
                            |     | - entering           |
                            |     | - walking            |
                            |     | - loitering          |
                            |     | - interacting        |
                            |     +----------+-----------+
                            |                |
                            |                v
                            |     +----------------------+
                            |     | Event Storage        |
                            |     | - relevant clips     |
                            |     | - unknown snapshots  |
                            |     | - manifest/index     |
                            |     +----------+-----------+
                            |                |
                            |                v
                            |     +----------------------+
                            |     | Summary Generator    |
                            |     | - daily report       |
                            |     | - key events         |
                            |     +----------+-----------+
                            |                |
                            |                v
                            |     +----------------------+
                            |     | Chat / API Layer     |
                            |     | - ask questions      |
                            |     | - retrieve events    |
                            |     +----------------------+
                            |
                            v
                    inactive footage ignored
```

## Folder Structure

```text
video-intelligence-agent/
|- src/
|  \- video_intelligence_agent/
|     |- cctv/
|     |  |- __init__.py
|     |  |- config.py
|     |  |- models.py
|     |  |- ingestion.py
|     |  |- motion.py
|     |  |- person.py
|     |  |- actions.py
|     |  |- storage.py
|     |  |- summary.py
|     |  |- chat.py
|     |  \- pipeline.py
|     |- core face-id modules...
|     \- video_summarizer.py
|- tests/
|  |- test_core.py
|  |- test_video_summarizer.py
|  \- test_cctv_agent.py
|- README.md
|- VIDEO_SUMMARIZER_GUIDE.md
\- CCTV_AGENT_GUIDE.md
```

## Module Responsibilities

### `config.py`

Central place for thresholds, output folders, frame sampling, clip saving, and action heuristics.

### `models.py`

Shared data models for:

- video metadata
- frame packets
- motion analysis
- person observations
- activity records
- daily summary
- final analysis result

### `ingestion.py`

Reads pre-recorded video files with OpenCV and yields sampled frames plus metadata.

### `motion.py`

Implements lightweight motion detection using frame differencing.

Why this matters:

- cheap to run on CPU
- filters inactive footage early
- reduces storage and later compute

### `person.py`

Wraps the existing `FaceIdentifier` module and converts detections into CCTV-friendly `PersonObservation` objects.

It also includes a simple IoU tracker to keep lightweight track IDs across frames.

### `actions.py`

Uses low-cost heuristics to label events such as:

- entering
- walking
- loitering
- interacting

This avoids heavy action-recognition models for the demo version.

### `storage.py`

Stores only meaningful outputs:

- event clips
- unknown-face snapshots
- JSONL event manifest

### `summary.py`

Builds the structured daily summary:

- total activity duration
- number of people
- known vs unknown
- key events

### `chat.py`

Provides an interactive agent layer.

It works locally with rule-based retrieval first, and can optionally plug into a responder model for better phrasing.

This makes it easy to connect Gemini later without changing the vision pipeline.

### `pipeline.py`

The main orchestrator that connects all modules together.

## Recommended Libraries

These are good choices for a student-friendly but realistic build:

- OpenCV for video I/O, image processing, and motion detection.
  Official: https://opencv.org/
  Docs: https://docs.opencv.org/

- Ultralytics YOLO for optional lightweight person detection or tracking upgrades.
  Docs: https://docs.ultralytics.com/

- DeepFace for simple Python-side face recognition integration.
  Official repo: https://github.com/serengil/deepface

- InsightFace if you want a stronger face-recognition stack later.
  Official repo: https://github.com/deepinsight/insightface

- Faster-Whisper for efficient local transcription.
  Official repo: https://github.com/SYSTRAN/faster-whisper

- Gemini API or OpenAI-compatible chat layer for the conversational interface.
  Gemini docs: https://ai.google.dev/docs
  OpenAI-compat for Gemini: https://ai.google.dev/gemini-api/docs/openai

## Why This Design Is Lightweight

The system stays efficient by making a few practical choices:

- motion detection happens before expensive recognition
- only sampled frames are processed
- inactive footage is ignored
- action detection uses heuristics instead of heavy video transformers
- only relevant clips are stored
- unknown snapshots are saved only when needed

## Example Workflow

```python
from video_intelligence_agent import FaceIdentifier, FaceIdentifierConfig
from video_intelligence_agent.cctv import (
    CCTVAgentConfig,
    CCTVAnalysisPipeline,
    FaceIdentifierRecognizer,
    FootageQueryAgent,
)

face_identifier = FaceIdentifier(
    config=FaceIdentifierConfig(
        database_path="data/embeddings.pkl",
        unknown_dir="data/unknown",
        similarity_threshold=0.4,
    )
)

pipeline = CCTVAnalysisPipeline(
    config=CCTVAgentConfig(
        output_dir="data/cctv_agent",
        frame_step=5,
        save_event_clips=True,
        save_unknown_snapshots=True,
    ),
    person_recognizer=FaceIdentifierRecognizer(face_identifier),
)

result = pipeline.process_video("samples/cctv_day1.mp4")
print(result.summary.summary_text)

agent = FootageQueryAgent(result)
print(agent.ask("Show me unknown persons from yesterday"))
```

## Example Output

```text
CCTV Summary for samples/cctv_day1.mp4
Total activity duration: 42.0s across 3 event segments.
People detected: 4 unique (2 known, 2 unknown).
Known individuals: Alan, Sara
Key events:
1. event-0001 | 12.4s to 20.1s | People: Alan | Actions: Alan entering, Alan walking
2. event-0002 | 88.0s to 101.5s | People: Unknown Person | Actions: Unknown Person loitering
3. event-0003 | 144.2s to 155.8s | People: Sara, Unknown Person | Actions: people interacting
```

## What To Demonstrate In A Project Demo

For a clean demo, show this sequence:

1. run the pipeline on a pre-recorded CCTV video
2. show that inactive footage is skipped
3. show one or two saved event clips
4. show an unknown-face snapshot
5. print the daily summary
6. ask a natural-language question using `FootageQueryAgent`

## Good Upgrade Path

If you want to make this stronger later, the cleanest upgrades are:

- replace motion-only human filtering with YOLO person detection
- replace heuristic actions with pose or action recognition
- store event manifests in SQLite instead of JSONL
- add a FastAPI layer for querying clips and summaries
- connect `FootageQueryAgent` to Gemini for richer natural-language answers

## Industry-Style Modular View

Even though this is a lightweight student project, the code now follows an industry-style split:

- ingestion
- detection
- recognition
- tracking
- action reasoning
- storage
- summarization
- query interface

That makes it much easier to test, explain, and extend.

