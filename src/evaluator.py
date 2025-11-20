"""Evaluation framework for comparing workflows"""

import re
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd


class WorkflowEvaluator:
    """Evaluate and compare different LangGraph workflows"""
    
    def __init__(self):
        self.results = []
    
    def extract_prediction(self, llm_output: str) -> str:
        """Extract HIGH/MEDIUM/LOW from LLM output"""
        output_upper = llm_output.upper()
        
        if "HIGH" in output_upper:
            return "HIGH"
        elif "MEDIUM" in output_upper:
            return "MEDIUM"
        elif "LOW" in output_upper:
            return "LOW"
        else:
            # Try to find prediction in common formats
            patterns = [
                r"prediction[:\s]+(HIGH|MEDIUM|LOW)",
                r"(HIGH|MEDIUM|LOW)[\s]+likelihood",
                r"likelihood[:\s]+(HIGH|MEDIUM|LOW)"
            ]
            for pattern in patterns:
                match = re.search(pattern, output_upper)
                if match:
                    return match.group(1)
            return "UNKNOWN"
    
    def convert_to_binary(self, prediction: str, threshold: str = "HIGH") -> int:
        """Convert prediction to binary (1 for promoted, 0 for not promoted)"""
        if threshold == "HIGH":
            return 1 if prediction == "HIGH" else 0
        elif threshold == "MEDIUM":
            return 1 if prediction in ["HIGH", "MEDIUM"] else 0
        else:
            return 1 if prediction != "LOW" else 0
    
    def evaluate_workflow(
        self,
        workflow_name: str,
        predictions: List[str],
        ground_truth: List[bool],
        threshold: str = "HIGH"
    ) -> Dict:
        """Evaluate a single workflow"""
        # Extract predictions
        extracted = [self.extract_prediction(p) for p in predictions]
        
        # Convert to binary
        binary_pred = [self.convert_to_binary(p, threshold) for p in extracted]
        binary_truth = [1 if gt else 0 for gt in ground_truth]
        
        # Calculate metrics
        accuracy = accuracy_score(binary_truth, binary_pred)
        precision = precision_score(binary_truth, binary_pred, zero_division=0)
        recall = recall_score(binary_truth, binary_pred, zero_division=0)
        f1 = f1_score(binary_truth, binary_pred, zero_division=0)
        
        cm = confusion_matrix(binary_truth, binary_pred)
        
        return {
            "workflow": workflow_name,
            "threshold": threshold,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "confusion_matrix": cm.tolist(),
            "predictions": extracted,
            "binary_predictions": binary_pred,
            "ground_truth": binary_truth
        }
    
    def compare_workflows(
        self,
        workflow1_name: str,
        workflow1_results: Dict,
        workflow2_name: str,
        workflow2_results: Dict
    ) -> pd.DataFrame:
        """Compare two workflows side by side"""
        comparison = {
            "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
            workflow1_name: [
                workflow1_results["accuracy"],
                workflow1_results["precision"],
                workflow1_results["recall"],
                workflow1_results["f1_score"]
            ],
            workflow2_name: [
                workflow2_results["accuracy"],
                workflow2_results["precision"],
                workflow2_results["recall"],
                workflow2_results["f1_score"]
            ]
        }
        
        df = pd.DataFrame(comparison)
        df["Difference"] = df[workflow1_name] - df[workflow2_name]
        df["Winner"] = df.apply(
            lambda row: workflow1_name if row["Difference"] > 0 else workflow2_name,
            axis=1
        )
        
        return df
    
    def generate_report(
        self,
        experiment_name: str,
        results: List[Dict]
    ) -> str:
        """Generate a text report of experiment results"""
        report = f"\n{'='*60}\n"
        report += f"Experiment: {experiment_name}\n"
        report += f"{'='*60}\n\n"
        
        for result in results:
            report += f"Workflow: {result['workflow']}\n"
            report += f"  Accuracy:  {result['accuracy']:.4f}\n"
            report += f"  Precision: {result['precision']:.4f}\n"
            report += f"  Recall:    {result['recall']:.4f}\n"
            report += f"  F1 Score:  {result['f1_score']:.4f}\n"
            report += f"  Confusion Matrix:\n"
            report += f"    {result['confusion_matrix']}\n\n"
        
        return report

