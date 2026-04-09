"""
Streamlit operator UI for Video Intelligence Agent.

Features:
- dashboard for pipeline artifacts and summaries
- event log viewer with filters and downloads
- Sarvam-powered chat over structured CCTV events
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def main() -> None:
    try:
        import streamlit as st
    except ImportError as exc:  # pragma: no cover - optional UI
        raise SystemExit(
            "Streamlit is not installed. Install it with:\n"
            "  python -m pip install streamlit\n"
            "or:\n"
            "  python -m pip install -e \".[ui]\""
        ) from exc

    st.set_page_config(
        page_title="Video Intelligence Agent",
        page_icon="VIA",
        layout="wide",
    )

    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        .stApp {
            background: linear-gradient(135deg, #0d0f14 0%, #111827 60%, #0d1117 100%);
            color: #e2e8f0;
        }

        section[data-testid="stSidebar"] {
            background: rgba(17, 24, 39, 0.95);
            border-right: 1px solid rgba(99,102,241,0.2);
        }

        div[data-testid="metric-container"] {
            background: rgba(30, 41, 59, 0.7);
            border: 1px solid rgba(99,102,241,0.25);
            border-radius: 12px;
            padding: 16px;
        }

        .stButton > button {
            background: linear-gradient(135deg, #6366f1, #8b5cf6);
            color: white;
            border: none;
            border-radius: 8px;
            font-weight: 600;
            transition: opacity 0.2s;
        }
        .stButton > button:hover { opacity: 0.85; }

        div[data-testid="stChatMessage"] {
            background: rgba(30, 41, 59, 0.6);
            border-radius: 12px;
            border: 1px solid rgba(99,102,241,0.15);
            margin: 4px 0;
        }

        pre, code {
            background: rgba(15, 23, 42, 0.8) !important;
            border: 1px solid rgba(99,102,241,0.2);
            border-radius: 8px;
        }

        .stTextInput > div > div > input,
        .stTextArea > div > div > textarea {
            background: rgba(17, 24, 39, 0.9) !important;
            border: 1px solid rgba(99,102,241,0.3) !important;
            border-radius: 8px;
            color: #e2e8f0 !important;
        }

        .status-ok   { color: #34d399; font-weight: 600; }
        .status-warn { color: #fbbf24; font-weight: 600; }
        .status-off  { color: #f87171; font-weight: 600; }

        .gradient-title {
            background: linear-gradient(90deg, #6366f1, #8b5cf6, #a78bfa);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 2.2rem;
            font-weight: 700;
        }

        hr { border-color: rgba(99,102,241,0.2); }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<p class="gradient-title">Video Intelligence Agent</p>', unsafe_allow_html=True)
    st.caption("Hybrid CCTV analytics | CV pipeline first | Sarvam reasoning second")
    st.markdown("---")

    outputs_dir = ROOT / "outputs" / "lobby_demo"
    analysis_path = outputs_dir / "latest_analysis.json"
    events_path = outputs_dir / "events.json"
    summary_path = outputs_dir / "daily_summary.txt"

    with st.sidebar:
        st.markdown("### Agent Configuration")

        api_key_input = st.text_input(
            "Sarvam API Key",
            value=os.environ.get("SARVAM_API_KEY", ""),
            type="password",
            help="Your Sarvam AI subscription key. Leave blank to use SARVAM_API_KEY.",
            key="sarvam_api_key",
        )

        model_name = st.selectbox(
            "Sarvam Model",
            ["sarvam-30b", "sarvam-105b"],
            index=0,
            help="Select the reasoning model used by the agent.",
        )

        st.markdown("---")
        st.markdown("### Event Data")
        custom_events = st.text_input(
            "Events JSON path",
            value=str(events_path),
            help="Path to the events.json produced by the CCTV pipeline.",
        )
        selected_events_path = Path(custom_events)

        if selected_events_path.exists():
            try:
                raw = json.loads(selected_events_path.read_text(encoding="utf-8"))
                event_count = len(raw) if isinstance(raw, list) else 0
                st.markdown(
                    f'<span class="status-ok">OK: {event_count} events loaded</span>',
                    unsafe_allow_html=True,
                )
            except Exception:
                st.markdown(
                    '<span class="status-warn">WARN: file exists but cannot be parsed</span>',
                    unsafe_allow_html=True,
                )
                event_count = 0
        else:
            st.markdown(
                '<span class="status-off">OFF: events.json not found - run the pipeline first</span>',
                unsafe_allow_html=True,
            )
            event_count = 0

        st.markdown("---")
        reload_btn = st.button("Reload Events And Clear Cache")

        st.markdown("---")
        st.markdown("### Example Queries")
        for example in [
            "Show me unknown people yesterday evening",
            "Did anyone loiter between 3 PM and 5 PM?",
            "Who entered today morning?",
            "List all events from 14:00 to 17:00",
            "Summarise today's activity",
        ]:
            st.markdown(f"- *{example}*")

    @st.cache_resource(show_spinner=False)
    def _build_agent_controller(api_key: str, model: str, ev_path: str):
        import importlib
        import video_intelligence_agent.agent.agent_controller
        import video_intelligence_agent.agent.sarvam_client

        importlib.reload(video_intelligence_agent.agent.sarvam_client)
        importlib.reload(video_intelligence_agent.agent.agent_controller)

        from video_intelligence_agent.agent.agent_controller import AgentController
        from video_intelligence_agent.agent.sarvam_client import SarvamClient, SarvamClientError

        sarvam_client = None
        if api_key.strip():
            try:
                sarvam_client = SarvamClient(api_key=api_key.strip(), model_name=model)
            except SarvamClientError as exc:
                return None, str(exc)

        controller = AgentController(events_path=ev_path, sarvam_client=sarvam_client)
        return controller, ""

    effective_key = api_key_input.strip() or os.environ.get("SARVAM_API_KEY", "")
    controller, init_error = _build_agent_controller(effective_key, model_name, str(selected_events_path))

    if reload_btn and controller is not None:
        controller.reload_events()
        st.toast("Events reloaded")

    tab_chat, tab_dashboard, tab_raw = st.tabs(["Ask the Agent", "Dashboard", "Raw Events"])

    with tab_chat:
        st.markdown("#### CCTV Intelligence Chat")

        if init_error:
            st.error(f"Agent initialisation error: {init_error}")

        if not effective_key:
            st.info(
                "No Sarvam API key configured. The agent will return formatted event data "
                "without LLM reasoning. Add your key in the sidebar to enable Sarvam.",
            )

        if event_count == 0:
            st.warning(
                "No events loaded. Run the CCTV pipeline first:\n\n"
                "```bash\npython main.py --config config.yaml\n```"
            )

        if "messages" not in st.session_state:
            st.session_state["messages"] = []

        for msg in st.session_state.get("messages", []):
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("clips"):
                    st.markdown("#### Associated Clips")
                    cols = st.columns(min(3, max(1, len(msg["clips"]))))
                    for i, clip in enumerate(msg["clips"]):
                        with cols[i % 3]:
                            clip_file = Path(clip["path"])
                            if clip_file.exists():
                                st.video(str(clip_file))
                            else:
                                st.warning(f"Clip file missing: {clip_file.name}")
                            st.caption(f"`{clip['action']}` - {clip['person']}")

        user_input = st.chat_input(
            "Ask about your CCTV footage... (for example: 'Show me unknown people yesterday evening')"
        )

        if user_input and controller is not None:
            st.session_state["messages"].append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            with st.chat_message("assistant"):
                with st.spinner("Analysing CCTV events..."):
                    response = controller.ask(user_input)

                st.markdown(response.answer)

                clips_to_show: list[dict[str, str]] = []
                clip_requested = bool(getattr(response.intent, "clip_requested", False))
                summary_requested = bool(getattr(response.intent, "summary_requested", False))
                wants_clips = clip_requested
                auto_show_clips = not summary_requested and len(response.matched_events) <= 3
                if wants_clips or auto_show_clips:
                    for event in response.matched_events:
                        clip_path = event.get("clip_path")
                        if clip_path and clip_path not in [c["path"] for c in clips_to_show]:
                            clips_to_show.append(
                                {
                                    "path": str(clip_path),
                                    "action": str(event.get("action", "unknown")),
                                    "person": str(event.get("person_id", "unknown")),
                                }
                            )

                if clips_to_show:
                    st.markdown("#### Associated Clips")
                    cols = st.columns(min(3, max(1, len(clips_to_show))))
                    for i, clip in enumerate(clips_to_show):
                        with cols[i % 3]:
                            clip_file = Path(clip["path"])
                            if clip_file.exists():
                                st.video(str(clip_file))
                            else:
                                st.warning(f"Clip file missing: {clip_file.name}")
                            st.caption(f"`{clip['action']}` - {clip['person']}")

                with st.expander("Query details", expanded=False):
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Events matched", len(response.matched_events))
                    col2.metric("Sarvam called", "Yes" if response.sarvam_called else "No")
                    col3.metric("From cache", "Yes" if response.from_cache else "No")

                    st.json(
                        {
                            "date": response.intent.date_hint,
                            "time_of_day": response.intent.time_of_day,
                            "start": response.intent.start_time_hint,
                            "end": response.intent.end_time_hint,
                            "person_type": response.intent.person_type_filter,
                            "person_id": response.intent.person_id_filter,
                            "action": response.intent.action_filter,
                            "summary_requested": summary_requested,
                            "clip_requested": clip_requested,
                        }
                    )

                    if response.error:
                        st.error(f"Error: {response.error}")

            st.session_state["messages"].append(
                {"role": "assistant", "content": response.answer, "clips": clips_to_show}
            )

        elif user_input and controller is None:
            st.error("Agent could not be initialised. Check the Sarvam API key.")

        if st.session_state.get("messages") and st.button("Clear chat history"):
            st.session_state["messages"] = []
            st.rerun()

    with tab_dashboard:
        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("#### Pipeline Artifacts")

            def _status_row(label: str, path: Path) -> None:
                icon = "OK" if path.exists() else "MISSING"
                st.markdown(f"{icon} **{label}** - `{path}`")

            _status_row("Events JSON", events_path)
            _status_row("Analysis JSON", analysis_path)
            _status_row("Daily Summary", summary_path)

            if not analysis_path.exists() and not events_path.exists():
                st.info(
                    "Run the pipeline to generate artifacts:\n\n"
                    "```bash\npython main.py --config config.yaml\n```"
                )

            if summary_path.exists():
                st.markdown("---")
                st.markdown("#### Latest Summary")
                st.text(summary_path.read_text(encoding="utf-8"))

        with col_right:
            st.markdown("#### System Pipeline")
            st.markdown(
                """
                ```
                Video Input
                  -> Frame Extraction
                  -> Motion Detection
                  -> Person / Face Detection  (YOLO + DeepFace)
                  -> Multi-Object Tracking
                  -> Event Extraction         (enter / exit / loiter)
                  -> Evidence Logging         (JSON + clips)
                  -> Sarvam Agent Reasoning
                  -> Summary Generation
                ```
                """
            )

            st.markdown("#### Design Philosophy")
            st.markdown(
                """
                - **CV first** - detect, track, and extract events locally
                - **LLM second** - Sarvam reasons over structured evidence, not raw frames
                - **Auditable** - every event has a JSON record and optional clip
                - **Resilient** - graceful degradation when Sarvam is unavailable
                """
            )

    with tab_raw:
        st.markdown("#### Raw Event Log")

        if not selected_events_path.exists():
            st.warning("No events.json found. Run the CCTV pipeline first.")
        else:
            try:
                raw_events = json.loads(selected_events_path.read_text(encoding="utf-8"))
                if isinstance(raw_events, list) and raw_events:
                    st.success(f"Loaded **{len(raw_events)}** events from `{selected_events_path}`")

                    actions = sorted({str(e.get("action", "")) for e in raw_events if e.get("action")})
                    selected_action = st.selectbox(
                        "Filter by action",
                        ["(all)"] + actions,
                        key="raw_action_filter",
                    )
                    show_events = (
                        raw_events
                        if selected_action == "(all)"
                        else [e for e in raw_events if e.get("action") == selected_action]
                    )

                    for event in show_events[:10]:
                        with st.expander(
                            f"[{event.get('start_time', '--')}] "
                            f"{event.get('person_id', 'unknown')} - "
                            f"{event.get('action', 'unknown')}",
                            expanded=False,
                        ):
                            st.json(event)

                    if len(show_events) > 10:
                        st.caption(f"Showing 10 of {len(show_events)} events. Download for full data.")

                    st.download_button(
                        "Download events.json",
                        data=json.dumps(raw_events, indent=2),
                        file_name="events.json",
                        mime="application/json",
                    )
                else:
                    st.info("Event log is empty. Run the pipeline to generate events.")
            except Exception as exc:
                st.error(f"Failed to parse events.json: {exc}")


if __name__ == "__main__":  # pragma: no cover
    main()
