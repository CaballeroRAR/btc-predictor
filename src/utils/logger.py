import logging
import sys
import os
from datetime import datetime

def setup_logger(name: str, level=logging.INFO):
    """
    Standardized logging configuration for the BTC Predictor project.
    Outputs to both stdout and a rolling log file.
    """
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers if setup multiple times
    if logger.hasHandlers():
        return logger

    logger.setLevel(level)

    # Format: 2024-04-11 23:19:20 | INFO | core.loader | Message
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # STDOUT Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler (Optional: Local persistence)
    log_dir = os.path.join(os.getcwd(), 'logs')
    if not os.path.exists(log_dir):
        try:
            os.makedirs(log_dir)
        except Exception:
            pass # Fallback to stdout only if directory creation fails
    
    if os.path.exists(log_dir):
        log_file = os.path.join(log_dir, f"{datetime.now().strftime('%Y-%m-%d')}_system.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
