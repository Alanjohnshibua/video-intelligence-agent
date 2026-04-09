from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from video_intelligence_agent.cctv_pipeline.core.video_processor import PipelineRunResult


class ReasoningResponderProtocol(Protocol):
    """LLM adapter protocol used only for higher-level reasoning and phrasing."""

    def generate(self, *, prompt: str) -> str: ...


@dataclass(slots=True)
class HybridVideoReasoningAgent:
    """
    Hybrid CV + LLM reasoning layer.

    The computer vision pipeline produces structured detections, tracks, and events.
    This agent turns that evidence into a concise prompt so an external model such as
    Sarvam or another LLM can reason over events instead of raw frames.
    """

    result: PipelineRunResult
    responder: ReasoningResponderProtocol | None = None

    def build_reasoning_prompt(self, question: str) -> str:
        event_lines = []
        for event in self.result.events[:20]:
            event_lines.append(
                (
                    f"- {event.start_time} to {event.end_time}: "
                    f"{event.person_id} {event.action} "
                    f"(track={event.track_id}, duration={event.duration_seconds:.1f}s)"
                )
            )
        if not event_lines:
            event_lines.append("- No events were detected in this run.")

        return "\n".join(
            [
                "You are assisting with CCTV event reasoning.",
                "Use the structured vision evidence below instead of inventing observations.",
                f"Question: {question}",
                "Video evidence:",
                *event_lines,
                "Answer with concise operational reasoning.",
            ]
        )

    def answer(self, question: str) -> str:
        prompt = self.build_reasoning_prompt(question)
        if self.responder is None:
            return prompt
        return self.responder.generate(prompt=prompt)
