from pathlib import Path
import logging


def configure_logging(
    log_directory: str,
    log_filename: str,
    log_level: str,
) -> None:
    log_dir = Path(log_directory)
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / log_filename

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    root_logger = logging.getLogger()

    numeric_level = getattr(
        logging,
        log_level.upper(),
        logging.INFO,
    )

    root_logger.setLevel(numeric_level)

    if root_logger.handlers:
        return

    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(
        log_file,
        encoding="utf-8",
    )
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(formatter)

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)