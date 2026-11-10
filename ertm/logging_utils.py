"""统一日志初始化。

约定：
- 所有模块通过 ``get_logger(__name__)`` 获取 logger；
- 入口只调用一次 ``setup_logging``，避免重复 handler；
- 控制台输出便于 PyCharm 查看，文件输出统一 UTF-8，便于追溯。
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
ROOT_LOGGER_NAME = "ertm"

_VALID_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


def _coerce_level(level: str | int) -> int:
    if isinstance(level, int):
        return level
    value = str(level).upper()
    if value not in _VALID_LEVELS:
        raise ValueError(f"不支持的日志级别: {level!r}，可选 {sorted(_VALID_LEVELS)}")
    return _VALID_LEVELS[value]


def setup_logging(
    level: str | int = "INFO",
    log_dir: str | Path | None = None,
    log_file: str = "ertm.log",
    *,
    force: bool = False,
) -> logging.Logger:
    """初始化 ``ertm`` 根 logger。

    Parameters
    ----------
    level:
        日志级别，字符串或 logging 常量。
    log_dir:
        若给定，则同时写入 ``log_dir/log_file``。
    log_file:
        日志文件名。
    force:
        为 True 时移除旧 handler，用于入口在读取配置后重新设置级别。
    """
    logger = logging.getLogger(ROOT_LOGGER_NAME)
    logger.setLevel(_coerce_level(level))
    logger.propagate = False

    if force:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            try:
                handler.close()
            except Exception:
                pass

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
               for h in logger.handlers):
        console = logging.StreamHandler(stream=sys.stdout)
        console.setLevel(_coerce_level(level))
        console.setFormatter(formatter)
        logger.addHandler(console)

    if log_dir is not None:
        target = Path(log_dir).expanduser().resolve()
        target.mkdir(parents=True, exist_ok=True)
        log_path = target / log_file
        already = any(
            isinstance(h, logging.FileHandler)
            and Path(getattr(h, "baseFilename", "")).resolve() == log_path.resolve()
            for h in logger.handlers
        )
        if not already:
            file_handler = logging.FileHandler(log_path, encoding="utf-8")
            file_handler.setLevel(_coerce_level(level))
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    # 让第三方库的 warning 也进入统一日志，避免散落在控制台。
    logging.captureWarnings(True)
    return logger


def get_logger(name: str) -> logging.Logger:
    """获取带 ``ertm.`` 前缀的 logger。"""
    if name == ROOT_LOGGER_NAME or name.startswith(f"{ROOT_LOGGER_NAME}."):
        return logging.getLogger(name)
    return logging.getLogger(f"{ROOT_LOGGER_NAME}.{name}")
