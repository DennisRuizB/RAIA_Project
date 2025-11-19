"""
Data loading module for EMNIST Letters dataset.

Handles CSV parsing, label mapping, and data validation.

Author: Senior ML Engineer
Date: 2025
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional
from pathlib import Path

from config import (
    TRAIN_DATA_PATH,
    TEST_DATA_PATH,
    MAPPING_PATH,
    N_PIXELS,
    TRAINING_CONFIG
)
from logger import LoggerMixin


class EMNISTDataLoader(LoggerMixin):
    """
    Loads and prepares EMNIST Letters dataset from CSV files.
    
    The EMNIST dataset format:
    - First column: label (1-26)
    - Remaining 784 columns: pixel values (0-255)
    - Images are 28x28 grayscale
    
    Attributes:
        label_mapping: Dictionary mapping numeric labels to letters
    """
    
    def __init__(self):
        """Initialize the data loader."""
        self._setup_logger()
        self.label_mapping: Optional[Dict[int, str]] = None
        self._load_mapping()
    
    def _load_mapping(self) -> None:
        """
        Load the label-to-letter mapping from the mapping file.
        
        The mapping file format:
        label ASCII_uppercase ASCII_lowercase
        Example: 1 65 97 (meaning label 1 = 'A')
        
        Raises:
            FileNotFoundError: If mapping file doesn't exist
            ValueError: If mapping file is malformed
        """
        try:
            if not MAPPING_PATH.exists():
                raise FileNotFoundError(f"Mapping file not found: {MAPPING_PATH}")
            
            self.logger.info(f"Loading label mapping from {MAPPING_PATH}")
            
            mapping_data = pd.read_csv(
                MAPPING_PATH,
                sep=' ',
                header=None,
                names=['label', 'ascii_upper', 'ascii_lower']
            )
            
            # Create mapping: label -> uppercase letter
            self.label_mapping = {
                int(row['label']): chr(int(row['ascii_upper']))
                for _, row in mapping_data.iterrows()
            }
            
            self.logger.info(f"Loaded mapping for {len(self.label_mapping)} classes")
            self.logger.debug(f"Mapping sample: {dict(list(self.label_mapping.items())[:5])}")
            
        except Exception as e:
            self.logger.error(f"Failed to load mapping: {str(e)}")
            raise
    
    def load_train_data(
        self,
        sample_size: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load training data from CSV.
        
        Args:
            sample_size: Optional number of samples to load (for quick testing)
        
        Returns:
            Tuple containing:
                - X_train: Feature array of shape (n_samples, 784)
                - y_train: Label array of shape (n_samples,)
                - letters_train: Letter strings array of shape (n_samples,)
        
        Raises:
            FileNotFoundError: If training data file doesn't exist
            ValueError: If data format is invalid
        """
        return self._load_data(
            TRAIN_DATA_PATH,
            "training",
            sample_size or TRAINING_CONFIG["train_sample_size"]
        )
    
    def load_test_data(
        self,
        sample_size: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load test data from CSV.
        
        Args:
            sample_size: Optional number of samples to load
        
        Returns:
            Tuple containing:
                - X_test: Feature array of shape (n_samples, 784)
                - y_test: Label array of shape (n_samples,)
                - letters_test: Letter strings array of shape (n_samples,)
        
        Raises:
            FileNotFoundError: If test data file doesn't exist
            ValueError: If data format is invalid
        """
        return self._load_data(
            TEST_DATA_PATH,
            "test",
            sample_size or TRAINING_CONFIG["test_sample_size"]
        )
    
    def _load_data(
        self,
        file_path: Path,
        data_type: str,
        sample_size: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Internal method to load data from CSV.
        
        Args:
            file_path: Path to CSV file
            data_type: String descriptor for logging ("training" or "test")
            sample_size: Optional sample size limit
        
        Returns:
            Tuple of (features, labels, letter_strings)
        """
        try:
            if not file_path.exists():
                raise FileNotFoundError(f"{data_type.capitalize()} data not found: {file_path}")
            
            self.logger.info(f"Loading {data_type} data from {file_path}")
            
            # Load CSV (no header, first column is label)
            df = pd.read_csv(file_path, header=None, dtype=np.float32)
            
            self.logger.info(f"Loaded {data_type} dataset shape: {df.shape}")
            
            # Validate data shape
            expected_columns = N_PIXELS + 1  # 784 pixels + 1 label
            if df.shape[1] != expected_columns:
                raise ValueError(
                    f"Expected {expected_columns} columns, got {df.shape[1]}"
                )
            
            # Sample if requested
            if sample_size is not None and sample_size < len(df):
                self.logger.info(f"Sampling {sample_size} examples from {len(df)}")
                df = df.sample(
                    n=sample_size,
                    random_state=TRAINING_CONFIG["random_state"]
                ).reset_index(drop=True)
            
            # Extract labels (first column) and features (remaining columns)
            y = df.iloc[:, 0].values.astype(np.int32)
            X = df.iloc[:, 1:].values.astype(np.float32)
            
            # Convert labels to letters
            letters = np.array([self.label_mapping[label] for label in y])
            
            # Data validation
            self._validate_data(X, y, data_type)
            
            self.logger.info(f"{data_type.capitalize()} data loaded successfully:")
            self.logger.info(f"  - Features shape: {X.shape}")
            self.logger.info(f"  - Labels shape: {y.shape}")
            self.logger.info(f"  - Unique classes: {np.unique(y).shape[0]}")
            self.logger.info(f"  - Label range: [{y.min()}, {y.max()}]")
            
            return X, y, letters
            
        except Exception as e:
            self.logger.error(f"Failed to load {data_type} data: {str(e)}")
            raise
    
    def _validate_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        data_type: str
    ) -> None:
        """
        Validate loaded data for consistency.
        
        Args:
            X: Feature array
            y: Label array
            data_type: Data descriptor for logging
        
        Raises:
            ValueError: If data validation fails
        """
        # Check for NaN values
        if np.isnan(X).any():
            raise ValueError(f"{data_type} features contain NaN values")
        
        if np.isnan(y).any():
            raise ValueError(f"{data_type} labels contain NaN values")
        
        # Check label range (EMNIST letters: 1-26)
        if y.min() < 1 or y.max() > 26:
            raise ValueError(
                f"Invalid label range in {data_type} data: [{y.min()}, {y.max()}]"
            )
        
        # Check feature range (pixels should be 0-255)
        if X.min() < 0 or X.max() > 255:
            self.logger.warning(
                f"Unusual pixel values in {data_type} data: [{X.min()}, {X.max()}]"
            )
        
        self.logger.debug(f"{data_type.capitalize()} data validation passed")
    
    def get_class_distribution(self, y: np.ndarray) -> Dict[str, int]:
        """
        Get the distribution of classes in the dataset.
        
        Args:
            y: Label array
        
        Returns:
            Dictionary mapping letters to their counts
        """
        unique, counts = np.unique(y, return_counts=True)
        distribution = {
            self.label_mapping[label]: count
            for label, count in zip(unique, counts)
        }
        return dict(sorted(distribution.items()))
