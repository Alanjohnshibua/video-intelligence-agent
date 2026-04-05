# Video Summarizer Guide

This project now includes a lightweight video summarizer.

It can:

- take a video file
- pull out the audio with FFmpeg
- convert speech to text with Faster-Whisper
- optionally identify people using Video Intelligence Agent (VIA)
- optionally analyze sampled frames with YOLOv8 Nano from Hugging Face
- send everything to an LLM and return a short summary

Think of it as:

`video -> audio + sampled frames -> transcript + visual cues + optional person names -> final summary`

## When To Use It

Use this feature when you want a quick written summary of a meeting, demo, CCTV clip, lecture, or recorded interview.

It is especially useful when:

- you do not want to watch the full video
- you want a simple text recap
- you want names and timestamps included when face identification is available

## What You Need

### 1. Python packages

If you want transcript-only video summaries:

```bash
pip install -e .[summary]
```

If you want transcript + YOLOv8 Nano visual summaries:

```bash
pip install -e .[summary,scene]
```

If you want video summaries with person identification:

```bash
pip install -e .[full,summary]
```

If you already have a working face pipeline and only need OpenCV video support:

```bash
pip install -e .[vision,summary]
```

### 2. FFmpeg

FFmpeg must be installed and available in your system PATH.

Quick check:

```bash
ffmpeg -version
```

If that command fails, install FFmpeg first.

### 3. A lightweight LLM

The summarizer now defaults to your local Qwen lightweight model path:

```text
C:\Users\acer\.cache\huggingface\hub\models--Qwen--Qwen3.5-2B-Base\snapshots\982d64a16433b4ece3b33ee53b24a0b416bd979a
```

This model is loaded directly with `transformers`.

If you prefer Ollama or another OpenAI-compatible server, that is still supported too.

## How It Works

The pipeline has 4 steps:

### Step 1. Extract audio

The program runs FFmpeg and saves the video's audio as a temporary `.wav` file.

### Step 2. Transcribe speech

Faster-Whisper reads the `.wav` file and creates a transcript.

Example:

```text
Alan explains the attendance dashboard and shows the camera feed.
```

### Step 3. Identify people

If person identification is enabled, the program samples video frames and runs Video Intelligence Agent (VIA) on them.

It produces lines like this:

```text
- Alan at 2.4s (confidence: 0.93)
- Unknown Person at 7.1s (confidence: 0.41)
```

If you skip this step, the summarizer still works. It will summarize from audio only.

### Step 4. Ask the LLM for the final summary

The program sends:

- the transcript
- detected people and timestamps
- any optional custom context you provide

Then the LLM returns a short structured summary.

### Optional Step 3b. Analyze what is visible in the video

If scene analysis is enabled, the program samples frames and runs YOLOv8 Nano from Hugging Face through Ultralytics.

It produces coarse visual context like:

```text
Visual scene analysis generated from sampled frames with YOLOv8 Nano.
Likely activity cues:
- sustained human presence across the clip
Dominant detected objects: person (4), backpack (2)
Sampled timeline:
- 1.0s: person, backpack
```

## Simplest Python Example

Use this when you only want transcript-based summarization:

```python
from video_intelligence_agent.video_summarizer import VideoSummarizer

summarizer = VideoSummarizer(
    llm_backend="transformers",
    llm_model_path=r"C:\Users\acer\.cache\huggingface\hub\models--Qwen--Qwen3.5-2B-Base\snapshots\982d64a16433b4ece3b33ee53b24a0b416bd979a",
    whisper_model="base",
    whisper_device="auto",
)

summary = summarizer.summarize_video("samples/demo.mp4")
print(summary)
```

If you prefer the naming from your standalone example script, these aliases also work:

```python
summarizer = VideoSummarizer(
    llm_mode="local",
    model_path=r"C:\Users\acer\.cache\huggingface\hub\models--Qwen--Qwen3.5-2B-Base\snapshots\982d64a16433b4ece3b33ee53b24a0b416bd979a",
    whisper_model="base",
    whisper_device="auto",
)
```

## Python Example With Video Intelligence Agent (VIA) Person Detection

Use this when you want the summary to mention who appears in the video.

```python
from video_intelligence_agent import FaceIdentifierConfig
from video_intelligence_agent.video_summarizer import (
    VideoIntelligencePersonIdentifier,
    VideoSummarizer,
)

person_module = VideoIntelligencePersonIdentifier(
    config=FaceIdentifierConfig(
        database_path="data/embeddings.pkl",
        unknown_dir="data/unknown",
        similarity_threshold=0.4,
    ),
    frame_step=30,
)

summarizer = VideoSummarizer(
    llm_backend="transformers",
    llm_model_path=r"C:\Users\acer\.cache\huggingface\hub\models--Qwen--Qwen3.5-2B-Base\snapshots\982d64a16433b4ece3b33ee53b24a0b416bd979a",
    whisper_model="base",
    whisper_device="auto",
    person_id_module=person_module,
    dataset_path="data/embeddings.pkl",
)

summary = summarizer.summarize_video("samples/office_meeting.mp4")
print(summary)
```

## CLI Example

The project CLI now includes a `summarize-video` command.

Transcript-only summary:

```bash
video-intelligence-agent summarize-video --video samples/demo.mp4 --no-person-id
```

Transcript + YOLOv8 Nano scene summary:

```bash
video-intelligence-agent summarize-video --video samples/demo.mp4 --no-person-id --scene-analysis
```

Summary with face identification:

```bash
video-intelligence-agent summarize-video --video samples/demo.mp4 --db data/embeddings.pkl
```

The command above already uses your local Qwen model path by default.

If you want to set it explicitly on Windows PowerShell:

```powershell
video-intelligence-agent summarize-video `
  --video samples/demo.mp4 `
  --llm-backend transformers `
  --llm-model-path "C:\Users\acer\.cache\huggingface\hub\models--Qwen--Qwen3.5-2B-Base\snapshots\982d64a16433b4ece3b33ee53b24a0b416bd979a" `
  --whisper-model base `
  --person-frame-step 20
```

If you prefer Ollama or another OpenAI-compatible server:

```powershell
video-intelligence-agent summarize-video `
  --video samples/demo.mp4 `
  --llm-backend openai `
  --llm-base-url http://localhost:11434/v1 `
  --llm-model qwen2.5:2b
```

## Important Settings

### `llm_base_url`

Where the LLM server is running when `llm_backend="openai"`.

### `llm_model`

Which API model should generate the summary when `llm_backend="openai"`.

### `llm_model_path`

Which local Hugging Face folder should be used when `llm_backend="transformers"`.

The default is your Qwen3.5-2B local snapshot path.

### `whisper_model`

Which Faster-Whisper model to use.

Smaller models are lighter and faster.
Larger models are usually more accurate.

### `whisper_device`

- `auto` to prefer GPU when PyTorch detects CUDA
- `cpu` for normal laptops and simple setups
- `cuda` for explicit NVIDIA GPU acceleration

### `frame_step`

How often the video is checked for faces.

Example:

- `frame_step=30` means analyze every 30th frame
- bigger value = faster but less detailed
- smaller value = slower but more detailed

## What The Output Looks Like

A typical result may look like this:

```text
1. Short Summary
The video shows Alan presenting a small attendance dashboard and explaining how the camera feed is processed.

2. Key Moments
- Around 2s, Alan appears on screen and begins the demo.
- Midway through the clip, he explains how unknown visitors are logged.
- Near the end, he summarizes the final workflow.

3. People Mentioned
- Alan
- Unknown Person
```

## What `module_context` Means

`module_context` is optional extra text you can pass to the summarizer.

Use it when you want the LLM to know about:

- your custom module behavior
- expected input and output format
- exact method names you already use in your project

Example:

```python
summarizer = VideoSummarizer(
    module_context="identify(video_path, dataset_path) returns a list of detections with name, timestamp, and confidence."
)
```

This helps the model stay closer to your real application logic.

## Common Problems

### FFmpeg not found

Problem:

```text
FFmpeg was not found
```

Fix:

- install FFmpeg
- add it to PATH
- restart the terminal

### `openai` package missing

Fix:

```bash
pip install -e .[summary]
```

### `faster-whisper` package missing

Fix:

```bash
pip install -e .[summary]
```

### Face identification fails

Fix:

- make sure your embeddings database exists
- make sure Video Intelligence Agent (VIA) is already working for images
- install the face recognition dependencies with `pip install -e .[full,summary]`

### Summary is too vague

Try:

- using a better LLM
- using a bigger Whisper model
- lowering `frame_step` so more face checks happen
- adding `module_context` if your project needs extra prompt guidance

## In One Sentence

This feature gives you a simple way to turn a video into a readable summary, with optional person names from Video Intelligence Agent (VIA).

