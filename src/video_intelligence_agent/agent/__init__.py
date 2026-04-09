"""
Conversational agent layer for querying processed CCTV event data.

Modules:
- reasoning: hybrid CV + LLM protocol layer
- query_parser: extracts dates, times, people, and action filters from user text
- time_filter: resolves human-readable time expressions to concrete windows
- event_retriever: loads and filters the local JSON event log
- sarvam_client: reusable Sarvam API wrapper
- agent_controller: orchestrates the full query-to-response pipeline
"""

from video_intelligence_agent.agent.agent_controller import AgentController
from video_intelligence_agent.agent.reasoning import HybridVideoReasoningAgent
from video_intelligence_agent.agent.sarvam_client import SarvamClient, SarvamClientError

__all__ = ["AgentController", "HybridVideoReasoningAgent", "SarvamClient", "SarvamClientError"]
