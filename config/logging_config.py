import logging
import logging.config
from colorlog import ColoredFormatter  # Для цветного вывода
# from core.settings import settings
from config.rag_config import RAGConfig

LOG_LEVEL = "DEBUG" if settings.DEBUG_MODE else "INFO"

# Форматтер с цветами для консоли
def get_console_formatter():
    return ColoredFormatter(
        fmt=(
            "%(log_color)s%(asctime)s [%(levelname)s]%(reset)s "
            "%(bold_white)s[%(name)s]%(reset)s | %(message)s"
        ),
        datefmt="%Y-%m-%d %H:%M:%S",
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        }
    )

# Форматтер для файлов (без цветов)
def get_file_formatter():
    return logging.Formatter(
        fmt="[%(asctime)s] %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

# Конфигурация логирования
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "console": {
            "()": get_console_formatter
        },
        "file": {
            "()": get_file_formatter
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "DEBUG",
            "formatter": "console",
            "stream": "ext://sys.stdout"
        },
        # "file": {
        #     "class": "logging.handlers.RotatingFileHandler",
        #     "level": "INFO",
        #     "formatter": "file",
        #     "filename": "logs/app.log",
        #     "maxBytes": 10485760,  # 10 MB
        #     "backupCount": 5,
        #     "encoding": "utf8"
        # },
        # "error_file": {
        #     "class": "logging.handlers.RotatingFileHandler",
        #     "level": "ERROR",
        #     "formatter": "file",
        #     "filename": "logs/error.log",
        #     "maxBytes": 10485760,
        #     "backupCount": 5,
        #     "encoding": "utf8"
        # }
    },
    "loggers": {
        "": {  # root logger
            "level": LOG_LEVEL,
            # "handlers": ["console", "file", "error_file"]
            "handlers": ["console"]
        },
        "uvicorn": {
            "level": LOG_LEVEL,
            "handlers": ["console"],
            "propagate": False
        },
        # "uvicorn.error": {
        #     "level": LOG_LEVEL,
        #     # "handlers": ["console", "error_file"],
        #     "handlers": ["console"],
        #     # "propagate": False
        # },
        "uvicorn.access": {
            "level": LOG_LEVEL,
            "handlers": ["console"],
            "propagate": False
        },
        "aiormq": {  # логи aio-pika
            "level": LOG_LEVEL,
            "handlers": ["console"],
            "propagate": False
        },
        "pika": {
            "level": LOG_LEVEL,
            "handlers": ["console"],
            "propagate": False
        },
        "sqlalchemy.engine": {
            "level": LOG_LEVEL,  # или DEBUG, если хочешь видеть SQL
            "handlers": ["console"],
            "propagate": False
        }
    }
}

# Применяем конфигурацию
logging.config.dictConfig(LOGGING_CONFIG)

# Создаём центральный логгер
logger = logging.getLogger("app")
