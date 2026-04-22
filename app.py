"""
Streamlit operator UI for Video Intelligence Agent.

Features:
- dashboard for pipeline artifacts and summaries
- event log viewer with filters and downloads
- Sarvam-powered chat over structured CCTV events
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _build_face_identifier(config):
    """Best-effort face identifier construction for Streamlit-triggered runs."""
    try:
        from video_intelligence_agent.config import FaceIdentifierConfig
        from video_intelligence_agent.core import FaceIdentifier
    except Exception as exc:
        return None, (
            "Face recognition dependencies are unavailable. People will be labeled as unknown. "
            f"reason={exc}"
        )

    try:
        identifier = FaceIdentifier(
            FaceIdentifierConfig(
                database_path=config.database_path,
                unknown_dir=config.unknown_dir,
                similarity_threshold=config.similarity_threshold,
                detector_backend=config.detector_backend,
                embedding_model=config.embedding_model,
                prefer_gpu=config.prefer_gpu,
                tensorflow_memory_growth=config.tensorflow_memory_growth,
            )
        )
        return identifier, ""
    except Exception as exc:
        return None, (
            "Face recognition failed to initialize. People will be labeled as unknown. "
            f"reason={exc}"
        )


def _prepare_run_config(config, *, video_path: str, output_dir: Path, debug_enabled: bool):
    """Prepare a per-video config so each upload writes to its own artifact folder."""
    updated = replace(
        config,
        video_path=video_path,
        storage=replace(config.storage, output_dir=output_dir),
    )
    if debug_enabled:
        updated = replace(
            updated,
            debug=replace(updated.debug, enabled=True, save_frames=True, draw_boxes=True),
        )
    return updated


def _run_analysis_for_video(
    *,
    base_config,
    video_path: Path,
    output_dir: Path,
    debug_enabled: bool,
    progress_callback=None,
) -> dict[str, object]:
    """Run the CCTV pipeline for a selected uploaded video and return a compact status payload."""
    from video_intelligence_agent.cctv_pipeline import VideoProcessor

    run_config = _prepare_run_config(
        base_config,
        video_path=str(video_path),
        output_dir=output_dir,
        debug_enabled=debug_enabled,
    )
    face_identifier, face_warning = _build_face_identifier(run_config)
    processor = VideoProcessor(config=run_config, face_identifier=face_identifier)
    result = processor.process_video(progress_callback=progress_callback)
    return {
        "video_path": str(video_path),
        "output_dir": str(output_dir),
        "events_detected": result.stats.events_detected,
        "events_path": str(result.artifacts.events_path),
        "analysis_path": str(result.artifacts.analysis_path) if result.artifacts.analysis_path else "",
        "face_warning": face_warning,
    }


def _sanitize_upload_name(value: str) -> str:
    """Convert a user-provided name into a filesystem-safe label."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-._")
    return cleaned or "item"


def _save_uploaded_video_file(*, uploads_root: Path, uploaded_file) -> Path:
    """Persist a Streamlit-uploaded video into the project upload library."""
    original_name = Path(uploaded_file.name).name
    stem = _sanitize_upload_name(Path(original_name).stem)
    suffix = Path(original_name).suffix.lower() or ".mp4"
    folder = uploads_root / stem
    folder.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    target = folder / f"{stem}-{timestamp}{suffix}"
    target.write_bytes(uploaded_file.getbuffer())
    return target


def _enroll_known_person_from_upload(*, config, person_name: str, uploaded_image) -> dict[str, object]:
    """Save an uploaded image into data and enroll the person into the embeddings store."""
    from video_intelligence_agent.core import FaceIdentifier
    from video_intelligence_agent.image_io import load_image

    safe_name = _sanitize_upload_name(person_name)
    gallery_dir = ROOT / "data" / "known_people" / safe_name
    gallery_dir.mkdir(parents=True, exist_ok=True)

    original_name = Path(uploaded_image.name).name
    suffix = Path(original_name).suffix.lower() or ".jpg"
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    image_path = gallery_dir / f"{safe_name}-{timestamp}{suffix}"
    image_path.write_bytes(uploaded_image.getbuffer())

    identifier, warning = _build_face_identifier(config)
    if identifier is None:
        raise RuntimeError(warning or "Face identifier could not be initialized.")

    result = identifier.add_person(
        person_name,
        load_image(image_path),
        source_image=image_path,
    )
    result["saved_image"] = str(image_path)
    return result


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

    from video_intelligence_agent.cctv_pipeline import load_pipeline_config
    from video_intelligence_agent.video_library import discover_video_records

    config = load_pipeline_config(ROOT / "config.yaml")
    library_dir = config.resolved_video_library_dir()
    library_output_dir = config.resolved_library_output_dir()
    discovered_videos = discover_video_records(library_dir, library_output_dir)
    analyzed_videos = [record for record in discovered_videos if (record.output_dir / "events.json").exists()]

    legacy_output_dir = ROOT / "outputs" / "lobby_demo"
    legacy_events_path = legacy_output_dir / "events.json"
    if legacy_events_path.exists():
        selected_record_labels = ["Legacy sample output"] + [record.display_name for record in discovered_videos]
    else:
        selected_record_labels = [record.display_name for record in discovered_videos]

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
        st.markdown("### Active Video")
        if not selected_record_labels:
            st.markdown(
                '<span class="status-off">OFF: no analyzed videos found yet</span>',
                unsafe_allow_html=True,
            )
            selected_video_label = ""
        else:
            selected_video_label = st.selectbox(
                "Select CCTV video",
                selected_record_labels,
                index=0,
                help="The agent, dashboard, and clip viewer will stay scoped to this video.",
            )

        selected_record = next(
            (record for record in discovered_videos if record.display_name == selected_video_label),
            None,
        )
        if selected_video_label == "Legacy sample output":
            outputs_dir = legacy_output_dir
            active_video_display = "samples/legacy output"
        elif selected_record is not None:
            outputs_dir = selected_record.output_dir
            active_video_display = selected_record.display_name
        else:
            outputs_dir = legacy_output_dir
            active_video_display = "no active video"

        analysis_path = outputs_dir / "latest_analysis.json"
        events_path = outputs_dir / "events.json"
        summary_path = outputs_dir / "daily_summary.txt"

        if selected_record is not None:
            st.caption(f"Active source: `{selected_record.video_path}`")
        elif selected_video_label == "Legacy sample output":
            st.caption("Active source: legacy sample pipeline output")
        else:
            st.caption("Run an analysis first to activate a video context.")

        analyze_debug = st.checkbox(
            "Enable debug artifacts for new analysis",
            value=False,
            help="When enabled, the pipeline saves annotated debug frames for the selected video.",
        )
        analyze_selected_btn = st.button(
            "Analyze selected video",
            disabled=selected_record is None,
            help="Run the CCTV pipeline for the selected uploaded video and refresh the agent context.",
        )

        st.markdown("---")
        st.markdown("### Event Data")
        manual_override = st.checkbox(
            "Use manual events.json override",
            value=False,
            help="Advanced option. Leave this off to keep the agent scoped to the selected video.",
        )
        if manual_override:
            custom_events = st.text_input(
                "Events JSON path",
                value=str(events_path),
                help="Advanced override for the events.json produced by the CCTV pipeline.",
            )
            selected_events_path = Path(custom_events)
            st.markdown(
                '<span class="status-warn">WARN: manual override is active, so the agent may not match the selected video</span>',
                unsafe_allow_html=True,
            )
        else:
            selected_events_path = events_path

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

        if discovered_videos:
            st.markdown("---")
            st.markdown("### Uploaded Video Library")
            for record in discovered_videos:
                status = "analyzed" if (record.output_dir / "events.json").exists() else "not analyzed"
                st.markdown(f"- `{record.display_name}` ({status})")

        st.markdown("---")
        st.markdown("### Add New Video Clip")
        uploaded_video = st.file_uploader(
            "Upload CCTV video",
            type=["mp4", "mov", "avi", "mkv", "webm", "m4v"],
            help="This saves the clip into data/clip_library/uploads so it can be analyzed from the app.",
            key="video_upload",
        )
        upload_video_btn = st.button(
            "Save uploaded video",
            disabled=uploaded_video is None,
            help="Store the uploaded video in the project library.",
        )

        st.markdown("---")
        st.markdown("### Add Known Person")
        known_person_name = st.text_input(
            "Person name",
            value="",
            help="This name will be stored in the embeddings database for face recognition.",
        )
        known_person_image = st.file_uploader(
            "Upload face image",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            help="Use a clear front-facing image for best enrollment quality.",
            key="known_person_upload",
        )
        enroll_person_btn = st.button(
            "Enroll known person",
            disabled=not known_person_name.strip() or known_person_image is None,
            help="Save the source image under data/known_people and add the face embedding to the database.",
        )

    if analyze_selected_btn and selected_record is not None:
        try:
            progress_text = st.empty()
            progress_bar = st.progress(0)

            def _on_analysis_progress(payload: dict[str, object]) -> None:
                percent = int(payload.get("percent", 0) or 0)
                message = str(payload.get("message", "Analyzing video..."))
                progress_bar.progress(min(max(percent, 0), 100))
                progress_text.caption(f"{message} | analyzed={percent}%")

            progress_text.caption(f"Starting analysis for {selected_record.display_name} | analyzed=0%")
            analysis_status = _run_analysis_for_video(
                base_config=config,
                video_path=selected_record.video_path,
                output_dir=selected_record.output_dir,
                debug_enabled=analyze_debug,
                progress_callback=_on_analysis_progress,
            )
            progress_bar.progress(100)
            progress_text.caption(
                f"Analysis complete for {selected_record.display_name} | analyzed=100%"
            )
            st.cache_resource.clear()
            st.session_state["analysis_notice"] = (
                f"Analysis complete for {selected_record.display_name}. "
                f"Events detected: {analysis_status['events_detected']}"
            )
            if analysis_status["face_warning"]:
                st.session_state["analysis_warning"] = str(analysis_status["face_warning"])
            else:
                st.session_state.pop("analysis_warning", None)
            st.rerun()
        except Exception as exc:
            st.error(f"Failed to analyze {selected_record.display_name}: {exc}")

    if upload_video_btn and uploaded_video is not None:
        try:
            saved_video = _save_uploaded_video_file(
                uploads_root=library_dir,
                uploaded_file=uploaded_video,
            )
            st.cache_resource.clear()
            st.session_state["analysis_notice"] = f"Uploaded video saved to {saved_video}"
            st.rerun()
        except Exception as exc:
            st.error(f"Failed to save uploaded video: {exc}")

    if enroll_person_btn and known_person_image is not None and known_person_name.strip():
        try:
            enrollment = _enroll_known_person_from_upload(
                config=config,
                person_name=known_person_name.strip(),
                uploaded_image=known_person_image,
            )
            st.cache_resource.clear()
            st.session_state["analysis_notice"] = (
                f"Enrolled {enrollment['name']} from {enrollment['saved_image']}"
            )
            st.rerun()
        except Exception as exc:
            st.error(f"Failed to enroll known person: {exc}")

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

    if "analysis_notice" in st.session_state:
        st.success(st.session_state.pop("analysis_notice"))
    if "analysis_warning" in st.session_state:
        st.warning(st.session_state.pop("analysis_warning"))

    previous_active_video = st.session_state.get("active_video_display")
    if previous_active_video != active_video_display:
        st.session_state["active_video_display"] = active_video_display
        st.session_state["messages"] = []

    if reload_btn and controller is not None:
        controller.reload_events()
        st.toast("Events reloaded")

    tab_chat, tab_dashboard, tab_raw = st.tabs(["Ask the Agent", "Dashboard", "Raw Events"])

    with tab_chat:
        st.markdown("#### CCTV Intelligence Chat")
        st.caption(f"Current video scope: `{active_video_display}`")

        if init_error:
            st.error(f"Agent initialisation error: {init_error}")

        if not effective_key:
            st.info(
                "No Sarvam API key configured. The agent will return formatted event data "
                "without LLM reasoning. Add your key in the sidebar to enable Sarvam.",
            )

        if event_count == 0:
            st.warning(
                "No events loaded for the selected video yet. Use the sidebar button "
                "`Analyze selected video`, or run:\n\n"
                "```bash\npython main.py --config config.yaml --analyze-all-uploads\n```"
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
                            st.caption(
                                f"`{clip['action']}` - {clip['person']} | source: {clip.get('source_video', 'unknown video')}"
                            )

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
                                    "source_video": str(
                                        ((event.get("metadata", {}) or {}).get("source_video_name", "unknown video"))
                                    ),
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
                            st.caption(
                                f"`{clip['action']}` - {clip['person']} | source: {clip.get('source_video', 'unknown video')}"
                            )

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
            st.caption(f"Current video scope: `{active_video_display}`")

            def _status_row(label: str, path: Path) -> None:
                icon = "OK" if path.exists() else "MISSING"
                st.markdown(f"{icon} **{label}** - `{path}`")

            _status_row("Events JSON", events_path)
            _status_row("Analysis JSON", analysis_path)
            _status_row("Daily Summary", summary_path)

            if not analysis_path.exists() and not events_path.exists():
                st.info(
                    "Use the sidebar button `Analyze selected video`, or run the pipeline manually:\n\n"
                    "```bash\npython main.py --config config.yaml --analyze-all-uploads\n```"
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
        st.caption(f"Current video scope: `{active_video_display}`")

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
                            f"{event.get('action', 'unknown')} "
                            f"({((event.get('metadata', {}) or {}).get('source_video_name', 'unknown video'))})",
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
