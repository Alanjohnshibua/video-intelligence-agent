from __future__ import annotations

from pathlib import Path

from video_intelligence_agent.video_summarizer import VideoSummarizer


class StubSegment:
    def __init__(self, text: str) -> None:
        self.text = text


class StubWhisper:
    def __init__(self, text: str) -> None:
        self.text = text

    def transcribe(self, audio_path: str, **_kwargs: object) -> tuple[list[StubSegment], None]:
        return ([StubSegment(self.text)], None)


class StubMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class StubChoice:
    def __init__(self, content: str) -> None:
        self.message = StubMessage(content)


class StubResponse:
    def __init__(self, content: str) -> None:
        self.choices = [StubChoice(content)]


class StubCompletions:
    def __init__(self, content: str) -> None:
        self.content = content
        self.last_prompt = ""

    def create(self, **kwargs: object) -> StubResponse:
        messages = kwargs["messages"]
        assert isinstance(messages, list)
        self.last_prompt = str(messages[0]["content"])
        return StubResponse(self.content)


class StubLLMClient:
    def __init__(self, content: str) -> None:
        self.chat = type("StubChat", (), {"completions": StubCompletions(content)})()


class StubPersonIdentifier:
    def identify(
        self,
        video_path: str,
        dataset_path: str | None = None,
    ) -> list[dict[str, object]]:
        return [
            {"name": "Alan", "timestamp": 2.4, "confidence": 0.93},
            {"name": "Unknown", "timestamp": 7.1, "confidence": 0.41},
        ]


class StubSceneAnalyzer:
    def describe(self, video_path: str) -> str:
        return (
            "Visual scene analysis generated from sampled frames with YOLOv8 Nano.\n"
            "Likely activity cues:\n"
            "- sustained human presence across the clip\n"
            "Dominant detected objects: person (4), backpack (2)\n"
            "Sampled timeline:\n"
            "- 1.0s: person, backpack"
        )


class SummarizerForTests(VideoSummarizer):
    def _extract_audio(self, video_path: str, output_path: str) -> None:
        Path(output_path).write_bytes(b"wav")


class StubChatTemplateTokenizer:
    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        assert tokenize is False
        assert add_generation_prompt is True
        return f"CHAT::{messages[0]['content']}"


def test_build_prompt_includes_custom_context_and_person_data() -> None:
    llm_client = StubLLMClient("summary")
    summarizer = VideoSummarizer(
        llm_client=llm_client,
        whisper_instance=StubWhisper("hello world"),
        module_context="identify(video_path, dataset_path) -> List[Dict]",
        unknown_person_label="Visitor",
    )

    prompt = summarizer._build_prompt(
        "Meeting starts now.",
        "- Alan at 2.4s (confidence: 0.93)",
        "- 1.0s: person, backpack",
        120,
    )

    assert "Meeting starts now." in prompt
    assert "Alan at 2.4s" in prompt
    assert "VISUAL SCENE ANALYSIS" in prompt
    assert "person, backpack" in prompt
    assert "identify(video_path, dataset_path)" in prompt
    assert "Visitor" in prompt


def test_summarize_video_runs_full_pipeline_and_cleans_audio(tmp_path: Path) -> None:
    llm_client = StubLLMClient("Short Summary\nKey Moments\nPeople Mentioned")
    video_path = tmp_path / "sample.mp4"
    temp_audio_path = tmp_path / "temp.wav"
    video_path.write_bytes(b"video")

    summarizer = SummarizerForTests(
        llm_client=llm_client,
        whisper_instance=StubWhisper("Alan explains the demo."),
        person_id_module=StubPersonIdentifier(),
        scene_analysis_module=StubSceneAnalyzer(),
        dataset_path="data/embeddings.pkl",
        unknown_person_label="Unknown Person",
    )

    summary = summarizer.summarize_video(
        str(video_path),
        temp_audio_path=str(temp_audio_path),
        max_summary_tokens=100,
    )

    assert "Short Summary" in summary
    assert not temp_audio_path.exists()
    assert "Alan explains the demo." in llm_client.chat.completions.last_prompt
    assert "Unknown Person" in llm_client.chat.completions.last_prompt
    assert "YOLOv8 Nano" in llm_client.chat.completions.last_prompt


def test_empty_transcript_falls_back_to_visual_context(tmp_path: Path) -> None:
    llm_client = StubLLMClient("Visual summary only")
    video_path = tmp_path / "sample.mp4"
    video_path.write_bytes(b"video")

    summarizer = SummarizerForTests(
        llm_client=llm_client,
        whisper_instance=StubWhisper(""),
        scene_analysis_module=StubSceneAnalyzer(),
    )

    summary = summarizer.summarize_video(str(video_path))

    assert "Visual summary only" in summary
    assert "No spoken dialogue was detected" in llm_client.chat.completions.last_prompt
    assert "sustained human presence" in llm_client.chat.completions.last_prompt


def test_prepare_local_prompt_uses_chat_template_when_available() -> None:
    summarizer = VideoSummarizer(
        llm_client=StubLLMClient("summary"),
        whisper_instance=StubWhisper("hello world"),
    )
    summarizer._local_llm_tokenizer = StubChatTemplateTokenizer()

    prepared = summarizer._prepare_local_prompt("Summarize this clip.")

    assert prepared == "CHAT::Summarize this clip."


def test_llm_mode_alias_maps_local_to_transformers() -> None:
    summarizer = VideoSummarizer(
        llm_mode="local",
        llm_client=StubLLMClient("summary"),
        whisper_instance=StubWhisper("hello world"),
    )

    assert summarizer.llm_backend == "transformers"

