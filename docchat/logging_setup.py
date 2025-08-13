import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_logging(log_dir: Path, log_name: str = "docchat.log", *, level: int = logging.INFO, force: bool = False) -> None:
    """Configure root logging.

    Args:
        log_dir: directory for log file
        log_name: file name
        level: base log level (INFO or DEBUG for verbose)
        force: if True, existing handlers are removed and reconfigured
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / log_name

    logger = logging.getLogger()
    if logger.handlers and not force:
        # Already configured and not forcing reconfiguration
        return
    if force:
        for h in list(logger.handlers):
            logger.removeHandler(h)
    logger.setLevel(level)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Rotating file handler
    fh = RotatingFileHandler(log_file, maxBytes=2 * 1024 * 1024, backupCount=3)
    fh.setLevel(level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
