from pathlib import Path
import logging
import yaml
import os

logger = logging.getLogger(__name__)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


def load_config():
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        if os.getenv("MLFLOW_TRACKING_URI"):
            config["mlflow"]["tracking_uri"] = os.getenv("MLFLOW_TRACKING_URI")

        logger.info("Configuration loaded successfully.")
        return config

    except FileNotFoundError:
        logger.error(
            "Configuration file not found: %s",
            CONFIG_PATH,
        )
        raise

    except yaml.YAMLError:
        logger.exception(
            "Failed to parse configuration file: %s",
            CONFIG_PATH,
        )
        raise


config = load_config()