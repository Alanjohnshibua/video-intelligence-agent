from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    try:
        import streamlit as st
    except ImportError as exc:  # pragma: no cover - optional UI
        raise SystemExit(
            "Streamlit is not installed. Install it with `python -m pip install streamlit` "
            "to run the operator demo UI."
        ) from exc

    st.set_page_config(page_title="Video Intelligence Agent", layout="wide")
    st.title("Video Intelligence Agent")
    st.caption("Hybrid CCTV analytics: CV pipeline first, LLM reasoning second.")

    st.markdown(
        """
        ### Pipeline
        `Video Input -> Frame Extraction -> Scene/Person Detection -> Tracking -> Event Extraction -> Agent Reasoning -> Summary`
        """
    )

    outputs_dir = Path("outputs/lobby_demo")
    analysis_path = outputs_dir / "latest_analysis.json"
    events_path = outputs_dir / "events.json"
    summary_path = outputs_dir / "daily_summary.txt"

    left, right = st.columns(2)
    with left:
        st.subheader("Operational Outputs")
        st.write(f"Analysis JSON: `{analysis_path}`")
        st.write(f"Event Log: `{events_path}`")
        st.write(f"Summary Report: `{summary_path}`")
        if analysis_path.exists():
            st.success("Latest analysis artifact found.")
        else:
            st.info("Run `python main.py --config config.yaml` to generate analysis artifacts.")

    with right:
        st.subheader("Design Philosophy")
        st.markdown(
            """
            - Use CV modules for perception and evidence extraction
            - Use the LLM only for reasoning and summarization
            - Keep the system auditable with JSON logs and saved clips
            - Degrade gracefully when some models are unavailable
            """
        )

    st.subheader("Interview Pitch")
    st.write(
        "This system is not a thin Gemini wrapper. It produces structured CV evidence first "
        "and only then uses an LLM for reasoning over events."
    )

    if events_path.exists():
        st.subheader("Recent Events")
        try:
            events = json.loads(events_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            st.warning("Event log exists but could not be parsed.")
        else:
            for event in events[:5]:
                st.code(
                    (
                        f"[{event.get('start_time', '--')}] "
                        f"{event.get('person_id', 'unknown')} "
                        f"{event.get('action', 'event')}"
                    )
                )

    if summary_path.exists():
        st.subheader("Latest Summary")
        st.text(summary_path.read_text(encoding="utf-8"))


if __name__ == "__main__":  # pragma: no cover - UI entrypoint
    main()
