# config/logging_config.py
"""
Единая точка настройки логирования.

Применяется при импорте (setup_logging() в конце файла).
ВЫЗЫВАЙТЕ ПЕРВОЙ СТРОКОЙ в api/app.py и ui/app.py,
до любых импортов core.* — иначе basicConfig из сторонних
модулей (pymilvus и т.п.) перебьёт dictConfig.

Уровень берётся из LOG_LEVEL (env), по умолчанию INFO.
"""
import logging
import logging.config
import os
import sys

# ── Опциональный colorlog с fallback ──────────────────────────────
try:
    from colorlog import ColoredFormatter
    _HAS_COLORLOG = True
except ImportError:
    ColoredFormatter = None
    _HAS_COLORLOG = False


LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG").upper()
LOG_TO_FILE = os.getenv("LOG_TO_FILE", "0") in ("1", "true", "True")
LOG_DIR = os.getenv("LOG_DIR", "logs")

# Библиотеки, которые в DEBUG-режиме пишут слишком много
_NOISY_LOGGERS = {
    "httpx": logging.WARNING,
    "httpcore": logging.WARNING,
    "urllib3": logging.WARNING,
    "asyncio": logging.WARNING,
    "pymilvus": logging.WARNING,
    "pymilvus.client": logging.WARNING,
    "transformers": logging.WARNING,
    "sentence_transformers": logging.WARNING,
    "torch": logging.WARNING,
    "huggingface_hub": logging.WARNING,
    "openai": logging.WARNING,
    "langchain": logging.INFO,
    "langchain_core": logging.INFO,
    "langchain_openai": logging.INFO,
}


# ── Форматтеры ────────────────────────────────────────────────────
def _make_console_formatter():
    if _HAS_COLORLOG:
        return ColoredFormatter(
            fmt=(
                "%(log_color)s%(asctime)s [%(levelname)s]%(reset)s "
                "%(bold_white)s[%(name)s]%(reset)s | %(message)s"
            ),
            datefmt="%Y-%m-%d %H:%M:%S",
            log_colors={
                "DEBUG":    "cyan",
                "INFO":     "green",
                "WARNING":  "yellow",
                "ERROR":    "red",
                "CRITICAL": "red,bg_white",
            },
        )
    # Fallback без цвета — тот же формат, без плейсхолдеров
    return logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] [%(name)s] | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _make_file_formatter():
    return logging.Formatter(
        fmt="[%(asctime)s] %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ── Сборка конфигурации ───────────────────────────────────────────
def _build_config() -> dict:
    handlers = {
        "console": {
            "class": "logging.StreamHandler",
            "level": LOG_LEVEL,
            "formatter": "console",
            "stream": "ext://sys.stdout",
        },
    }

    if LOG_TO_FILE:
        os.makedirs(LOG_DIR, exist_ok=True)
        handlers["file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "INFO",
            "formatter": "file",
            "filename": os.path.join(LOG_DIR, "app.log"),
            "maxBytes": 10 * 1024 * 1024,
            "backupCount": 5,
            "encoding": "utf8",
        }
        handlers["error_file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "ERROR",
            "formatter": "file",
            "filename": os.path.join(LOG_DIR, "error.log"),
            "maxBytes": 10 * 1024 * 1024,
            "backupCount": 5,
            "encoding": "utf8",
        }

    h = list(handlers.keys())

    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "console": {"()": _make_console_formatter},
            "file":    {"()": _make_file_formatter},
        },
        "handlers": handlers,
        "loggers": {
            "": {"level": LOG_LEVEL, "handlers": h},
            "core": {"level": LOG_LEVEL, "handlers": h, "propagate": False},
            "api":  {"level": LOG_LEVEL, "handlers": h, "propagate": False},
            "ui":   {"level": LOG_LEVEL, "handlers": h, "propagate": False},
            "uvicorn":         {"level": LOG_LEVEL, "handlers": h, "propagate": False},
            "uvicorn.access":  {"level": "WARNING", "handlers": h, "propagate": False},
            "uvicorn.error":   {"level": LOG_LEVEL, "handlers": h, "propagate": False},
            "pika":            {"level": "WARNING", "handlers": h, "propagate": False},
            "aiormq":          {"level": "WARNING", "handlers": h, "propagate": False},
        },
    }


def setup_logging() -> None:
    """Применяет конфигурацию. Идемпотентно."""
    logging.config.dictConfig(_build_config())
    for name, lvl in _NOISY_LOGGERS.items():
        logging.getLogger(name).setLevel(lvl)
    logging.getLogger(__name__).info(
        "Logging настроен: level=%s, colorlog=%s, files=%s",
        LOG_LEVEL, _HAS_COLORLOG, LOG_TO_FILE,
    )


# ★ Ключевая строка: применяем конфиг при импорте модуля.
# Без неё `import config.logging_config` в app.py ничего не делает.
setup_logging()