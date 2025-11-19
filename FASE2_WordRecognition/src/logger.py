"""
Logging utility module for Word Recognition System.

Provides centralized logging configuration.

Author: Senior ML Engineer
Date: 2025
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime

from src.config import LOGGING_CONFIG, OUTPUT_DIR


def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: Optional[str] = None
) -> logging.Logger:
    """
    Configure and return a logger instance.
    
    Args:
        name: Logger name
        log_file: Optional custom log filename
        level: Optional logging level override
    
    Returns:
        Configured logger instance
    """
    log_level = level or LOGGING_CONFIG["level"]
    log_format = LOGGING_CONFIG["format"]
    log_to_file = LOGGING_CONFIG["log_to_file"]
    
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level))
    
    if logger.handlers:
        return logger
    
    formatter = logging.Formatter(log_format)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_to_file:
        if log_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = f"{LOGGING_CONFIG['log_filename'].split('.')[0]}_{timestamp}.log"
        
        file_path = OUTPUT_DIR / log_file
        file_handler = logging.FileHandler(file_path, mode='a', encoding='utf-8')
        file_handler.setLevel(getattr(logging, log_level))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


class LoggerMixin:
    """Mixin class to provide logging capabilities."""
    
    def _setup_logger(self, name: Optional[str] = None) -> None:
        """Initialize logger for the class."""
        logger_name = name or self.__class__.__name__
        self.logger = setup_logger(logger_name)
