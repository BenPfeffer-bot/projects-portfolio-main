import os
import yaml


def load_config(config_path: str = "config/file_config.yaml") -> dict:
    """
    Load configuration from YAML file

    Args:
        config_path: Path to config YAML file

    Returns:
        Dictionary containing configuration
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def setup_logging(log_level: str = "INFO") -> None:
    """
    Configure logging for the project

    Args:
        log_level: Desired logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    import logging
    import sys
    from datetime import datetime

    # Create logs directory if it doesn't exist
    config = load_config()
    log_dir = config["files"]["logs_dir"]
    os.makedirs(log_dir, exist_ok=True)

    # Configure logging format
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(log_format, date_format))
    root_logger.addHandler(console_handler)

    # File handler
    log_file = os.path.join(
        log_dir, f"app_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter(log_format, date_format))
    root_logger.addHandler(file_handler)
