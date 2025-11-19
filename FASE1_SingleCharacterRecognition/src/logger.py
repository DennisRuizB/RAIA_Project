"""
Logging utility module for the EMNIST Letter Recognition System.

Provides centralized logging configuration and utilities.

Author: Senior ML Engineer
Date: 2025
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime

from config import LOGGING_CONFIG, LOGS_DIR


def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: Optional[str] = None
) -> logging.Logger:
    """
    Configure and return a logger instance.
    
    Args:
        name: Logger name (typically __name__ of the calling module)
        log_file: Optional custom log filename. If None, uses config default
        level: Optional logging level override. If None, uses config default
    
    Returns:
        Configured logger instance
    
    Example:
        >>> logger = setup_logger(__name__)
        >>> logger.info("Training started")
    """
    # Get configuration
    log_level = level or LOGGING_CONFIG["level"]
    log_format = LOGGING_CONFIG["format"]
    log_to_file = LOGGING_CONFIG["log_to_file"]
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level))
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Create formatter
    formatter = logging.Formatter(log_format)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_to_file:
        if log_file is None:
            # Generate timestamped log file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = f"{LOGGING_CONFIG['log_filename'].split('.')[0]}_{timestamp}.log"
        
        file_path = LOGS_DIR / log_file
        file_handler = logging.FileHandler(file_path, mode='a', encoding='utf-8')
        file_handler.setLevel(getattr(logging, log_level))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"Logging to file: {file_path}")
    
    return logger


class LoggerMixin:
    """
    Mixin class to provide logging capabilities to any class.
    
    Usage:
        class MyClass(LoggerMixin):
            def __init__(self):
                self._setup_logger()
                self.logger.info("MyClass initialized")
    """
    
    def _setup_logger(self, name: Optional[str] = None) -> None:
        """Initialize logger for the class."""
        logger_name = name or self.__class__.__name__
        self.logger = setup_logger(logger_name)
