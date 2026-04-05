from __future__ import annotations

import logging


class _ColorFormatter(logging.Formatter):
    COLORS = {
        logging.DEBUG: "\033[36m",
        logging.INFO: "\033[32m",
        logging.WARNING: "\033[33m",
        logging.ERROR: "\033[31m",
        logging.CRITICAL: "\033[35m",
    }
    RESET = "\033[0m"

    def __init__(self, use_color: bool = True) -> None:
        super().__init__(datefmt="%Y-%m-%d %H:%M:%S")
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        module_name = record.name.rsplit(".", 1)[-1]
        level_name = record.levelname
        if self.use_color:
            color = self.COLORS.get(record.levelno, "")
            level_name = f"{color}{level_name}{self.RESET}"
        timestamp = self.formatTime(record, self.datefmt)
        return f"[{timestamp}] [{module_name}] [{level_name}] {record.getMessage()}"


def configure_logging(*, debug: bool = False, use_color: bool = True) -> logging.Logger:
    """Configure package logging once and return the package logger."""

    logger = logging.getLogger("video_intelligence_agent.cctv_pipeline")
    level = logging.DEBUG if debug else logging.INFO
    logger.setLevel(level)
    logger.propagate = False

    if logger.handlers:
        for handler in logger.handlers:
            handler.setLevel(level)
            handler.setFormatter(_ColorFormatter(use_color=use_color))
        return logger

    handler = logging.StreamHandler()
    handler.setLevel(level)
    handler.setFormatter(_ColorFormatter(use_color=use_color))
    logger.addHandler(handler)
    return logger


def get_pipeline_logger(module: str) -> logging.Logger:
    """Return a module-scoped logger under the CCTV pipeline namespace."""

    return logging.getLogger(f"video_intelligence_agent.cctv_pipeline.{module}")
