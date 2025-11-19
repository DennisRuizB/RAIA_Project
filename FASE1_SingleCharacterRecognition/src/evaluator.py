"""
Model evaluation module for EMNIST letter recognition.

Provides comprehensive evaluation metrics and analysis tools.

Author: Senior ML Engineer
Date: 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from config import EVALUATION_CONFIG, N_CLASSES
from logger import LoggerMixin


class ModelEvaluator(LoggerMixin):
    """
    Evaluates model performance with comprehensive metrics.
    
    Provides:
    - Accuracy, Precision, Recall, F1 scores
    - Confusion matrix analysis
    - Per-class performance breakdown
    - Error analysis and visualization
    
    Attributes:
        label_mapping: Dictionary mapping numeric labels to letters
        results: Dictionary storing evaluation results
    """
    
    def __init__(self, label_mapping: Dict[int, str]):
        """
        Initialize the evaluator.
        
        Args:
            label_mapping: Dictionary mapping numeric labels to letter strings
        """
        self._setup_logger()
        self.label_mapping = label_mapping
        self.results: Dict[str, Any] = {}
        
        self.logger.info("ModelEvaluator initialized")
    
    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        dataset_name: str = "Test"
    ) -> Dict[str, Any]:
        """
        Perform comprehensive evaluation of predictions.
        
        Args:
            y_true: True labels of shape (n_samples,)
            y_pred: Predicted labels of shape (n_samples,)
            dataset_name: Name of dataset being evaluated (for logging)
        
        Returns:
            Dictionary containing all evaluation metrics
        
        Raises:
            ValueError: If y_true and y_pred have different shapes
        """
        if y_true.shape != y_pred.shape:
            raise ValueError(
                f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}"
            )
        
        self.logger.info("=" * 70)
        self.logger.info(f"EVALUATING {dataset_name.upper()} SET")
        self.logger.info("=" * 70)
        
        # Calculate metrics
        results = {}
        
        # Overall accuracy
        accuracy = accuracy_score(y_true, y_pred)
        results["accuracy"] = accuracy
        self.logger.info(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Calculate average metrics
        average = EVALUATION_CONFIG["average"]
        
        if "precision" in EVALUATION_CONFIG["metrics"]:
            precision = precision_score(y_true, y_pred, average=average, zero_division=0)
            results["precision"] = precision
            self.logger.info(f"Precision ({average}): {precision:.4f}")
        
        if "recall" in EVALUATION_CONFIG["metrics"]:
            recall = recall_score(y_true, y_pred, average=average, zero_division=0)
            results["recall"] = recall
            self.logger.info(f"Recall ({average}): {recall:.4f}")
        
        if "f1" in EVALUATION_CONFIG["metrics"]:
            f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
            results["f1"] = f1
            self.logger.info(f"F1-Score ({average}): {f1:.4f}")
        
        # Confusion matrix
        if EVALUATION_CONFIG["generate_confusion_matrix"]:
            cm = confusion_matrix(y_true, y_pred)
            results["confusion_matrix"] = cm
            self.logger.info(f"Confusion Matrix shape: {cm.shape}")
        
        # Classification report
        if EVALUATION_CONFIG["generate_classification_report"]:
            # Get letter labels
            labels = sorted(np.unique(np.concatenate([y_true, y_pred])))
            target_names = [self.label_mapping[label] for label in labels]
            
            report = classification_report(
                y_true,
                y_pred,
                labels=labels,
                target_names=target_names,
                zero_division=0
            )
            results["classification_report"] = report
            self.logger.info("\nClassification Report:\n" + report)
        
        # Error analysis
        error_analysis = self._analyze_errors(y_true, y_pred)
        results["error_analysis"] = error_analysis
        
        self.logger.info("=" * 70)
        
        # Store results
        self.results[dataset_name] = results
        
        return results
    
    def _analyze_errors(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, Any]:
        """
        Analyze prediction errors to identify problematic classes.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
        
        Returns:
            Dictionary with error analysis information
        """
        self.logger.info("\n--- Error Analysis ---")
        
        # Calculate error rate
        n_errors = np.sum(y_true != y_pred)
        error_rate = n_errors / len(y_true)
        
        self.logger.info(f"Total Errors: {n_errors} / {len(y_true)} ({error_rate*100:.2f}%)")
        
        # Per-class error analysis
        class_errors = {}
        labels = sorted(np.unique(y_true))
        
        for label in labels:
            mask = y_true == label
            n_samples = np.sum(mask)
            n_correct = np.sum((y_true == y_pred) & mask)
            class_accuracy = n_correct / n_samples if n_samples > 0 else 0.0
            
            letter = self.label_mapping[label]
            class_errors[letter] = {
                "label": int(label),
                "accuracy": float(class_accuracy),
                "n_samples": int(n_samples),
                "n_correct": int(n_correct),
                "n_errors": int(n_samples - n_correct)
            }
        
        # Sort by accuracy (ascending) to find worst performers
        sorted_errors = sorted(
            class_errors.items(),
            key=lambda x: x[1]["accuracy"]
        )
        
        # Show top-k worst performing classes
        top_k = EVALUATION_CONFIG["top_k_errors"]
        self.logger.info(f"\nTop {top_k} Most Confused Classes:")
        
        for i, (letter, stats) in enumerate(sorted_errors[:top_k], 1):
            self.logger.info(
                f"{i}. '{letter}': {stats['accuracy']*100:.2f}% accuracy "
                f"({stats['n_errors']} errors / {stats['n_samples']} samples)"
            )
        
        return {
            "total_errors": int(n_errors),
            "error_rate": float(error_rate),
            "class_errors": class_errors,
            "worst_classes": sorted_errors[:top_k]
        }
    
    def get_confusion_pairs(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        top_k: int = 10
    ) -> List[Tuple[str, str, int]]:
        """
        Identify the most common confusion pairs (true label, predicted label).
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            top_k: Number of top confusion pairs to return
        
        Returns:
            List of tuples (true_letter, pred_letter, count) sorted by count
        """
        confusion_counts: Dict[Tuple[int, int], int] = {}
        
        # Count confusions (excluding correct predictions)
        for true_label, pred_label in zip(y_true, y_pred):
            if true_label != pred_label:
                pair = (true_label, pred_label)
                confusion_counts[pair] = confusion_counts.get(pair, 0) + 1
        
        # Convert to letter pairs and sort
        confusion_pairs = [
            (
                self.label_mapping[true_label],
                self.label_mapping[pred_label],
                count
            )
            for (true_label, pred_label), count in confusion_counts.items()
        ]
        
        confusion_pairs.sort(key=lambda x: x[2], reverse=True)
        
        # Log top confusions
        self.logger.info(f"\nTop {top_k} Confusion Pairs:")
        for i, (true_letter, pred_letter, count) in enumerate(confusion_pairs[:top_k], 1):
            self.logger.info(
                f"{i}. '{true_letter}' confused with '{pred_letter}': {count} times"
            )
        
        return confusion_pairs[:top_k]
    
    def save_results(self, filepath: str) -> None:
        """
        Save evaluation results to a text file.
        
        Args:
            filepath: Path where to save the results
        """
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write("=" * 70 + "\n")
                f.write("MODEL EVALUATION RESULTS\n")
                f.write("=" * 70 + "\n\n")
                
                for dataset_name, results in self.results.items():
                    f.write(f"\n{dataset_name} Set Results:\n")
                    f.write("-" * 50 + "\n")
                    
                    # Write metrics
                    if "accuracy" in results:
                        f.write(f"Accuracy: {results['accuracy']:.4f}\n")
                    if "precision" in results:
                        f.write(f"Precision: {results['precision']:.4f}\n")
                    if "recall" in results:
                        f.write(f"Recall: {results['recall']:.4f}\n")
                    if "f1" in results:
                        f.write(f"F1-Score: {results['f1']:.4f}\n")
                    
                    # Write classification report
                    if "classification_report" in results:
                        f.write("\n" + results["classification_report"] + "\n")
                    
                    # Write error analysis
                    if "error_analysis" in results:
                        ea = results["error_analysis"]
                        f.write(f"\nTotal Errors: {ea['total_errors']}\n")
                        f.write(f"Error Rate: {ea['error_rate']*100:.2f}%\n")
                    
                    f.write("\n")
            
            self.logger.info(f"Results saved to: {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to save results: {str(e)}")
            raise
    
    def get_per_class_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> pd.DataFrame:
        """
        Get detailed per-class metrics as a DataFrame.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
        
        Returns:
            DataFrame with per-class metrics (precision, recall, f1)
        """
        labels = sorted(np.unique(y_true))
        target_names = [self.label_mapping[label] for label in labels]
        
        # Calculate per-class metrics
        precision = precision_score(
            y_true, y_pred, labels=labels, average=None, zero_division=0
        )
        recall = recall_score(
            y_true, y_pred, labels=labels, average=None, zero_division=0
        )
        f1 = f1_score(
            y_true, y_pred, labels=labels, average=None, zero_division=0
        )
        
        # Create DataFrame
        df = pd.DataFrame({
            "Letter": target_names,
            "Label": labels,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1
        })
        
        return df.sort_values("F1-Score")
    
    def visualize_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        normalize: bool = False
    ) -> None:
        """
        Visualize confusion matrix using matplotlib.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            normalize: Whether to normalize the confusion matrix
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
        except ImportError:
            self.logger.warning("Matplotlib not available for visualization")
            return
        
        # Get unique labels from both y_true and y_pred
        labels = sorted(np.unique(np.concatenate([y_true, y_pred])))
        target_names = [self.label_mapping[label] for label in labels]
        
        # Generate confusion matrix with explicit labels
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        fig, ax = plt.subplots(figsize=(12, 10))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        
        # Set ticks - now matching the actual number of labels
        ax.set(
            xticks=np.arange(len(target_names)),
            yticks=np.arange(len(target_names)),
            xticklabels=target_names,
            yticklabels=target_names,
            title="Confusion Matrix",
            ylabel="True Label",
            xlabel="Predicted Label"
        )
        
        # Rotate x labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add text annotations (only for small matrices)
        if len(target_names) <= 15:
            fmt = '.2f' if normalize else 'd'
            thresh = cm.max() / 2. if cm.max() > 0 else 0.5
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, format(cm[i, j], fmt),
                           ha="center", va="center",
                           color="white" if cm[i, j] > thresh else "black",
                           fontsize=8)
        
        fig.tight_layout()
        
        # Save figure
        save_path = "confusion_matrix.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        self.logger.info(f"Confusion matrix saved to: {save_path}")
        plt.close()
