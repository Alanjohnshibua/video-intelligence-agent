from __future__ import annotations

from video_intelligence_agent.cctv.config import CCTVAgentConfig
from video_intelligence_agent.cctv.models import PersonObservation


class ActionAnalyzer:
    """Heuristic action labels for a lightweight CCTV demo pipeline."""

    def __init__(self, config: CCTVAgentConfig) -> None:
        self.config = config

    def infer(
        self,
        frame_shape: tuple[int, ...],
        people: list[PersonObservation],
    ) -> list[str]:
        if not people:
            return ["motion detected"]

        labels: list[str] = []
        height, width = frame_shape[:2]
        for person in people:
            name = person.display_name()
            movement_px = float(person.metadata.get("movement_px", 0.0))
            is_new_track = bool(person.metadata.get("is_new_track", False))
            near_border = self._near_border(person, width, height)

            if is_new_track and near_border:
                labels.append(f"{name} entering")
            if (
                not is_new_track
                and near_border
                and movement_px >= self.config.walking_distance_px / 2.0
                and person.tracked_duration_seconds > 0.0
            ):
                labels.append(f"{name} exiting")
            if movement_px >= self.config.walking_distance_px:
                labels.append(f"{name} walking")
            if person.tracked_duration_seconds >= self.config.loitering_threshold_sec:
                labels.append(f"{name} loitering")

        if len(people) >= 2 and self._are_interacting(people):
            labels.append("people interacting")

        return sorted(dict.fromkeys(labels))

    def _near_border(
        self,
        person: PersonObservation,
        frame_width: int,
        frame_height: int,
    ) -> bool:
        bbox = person.bbox
        if bbox is None:
            return False
        margin_x = frame_width * self.config.border_margin_ratio
        margin_y = frame_height * self.config.border_margin_ratio
        return (
            bbox.x <= margin_x
            or bbox.y <= margin_y
            or (bbox.x + bbox.w) >= frame_width - margin_x
            or (bbox.y + bbox.h) >= frame_height - margin_y
        )

    def _are_interacting(self, people: list[PersonObservation]) -> bool:
        for index, person in enumerate(people):
            if person.bbox is None:
                continue
            for other in people[index + 1 :]:
                if other.bbox is None:
                    continue
                if self._distance_between(person, other) <= self.config.interaction_distance_px:
                    return True
        return False

    @staticmethod
    def _distance_between(a: PersonObservation, b: PersonObservation) -> float:
        assert a.bbox is not None
        assert b.bbox is not None
        ax = a.bbox.x + a.bbox.w / 2.0
        ay = a.bbox.y + a.bbox.h / 2.0
        bx = b.bbox.x + b.bbox.w / 2.0
        by = b.bbox.y + b.bbox.h / 2.0
        return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5

