"""
Evaluation script for Crocs RTB relevance system.

Trains the RelevanceModel, evaluates on test set, and generates predictions
for the pages dataset. Supports train-only, eval-only, or full pipeline modes.
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.progress import track
from rich.panel import Panel
from rich.text import Text
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, f1_score

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))
from model import RelevanceModel
from baseline import BaselineCosine


class ModelEvaluator:
    """Handles training, evaluation, and result generation for the relevance model."""
    
    def __init__(self):
        self.console = Console()
        self.model = None
        self.project_root = Path(__file__).parent.parent
        self.data_dir = self.project_root / "data"
        self.artifacts_dir = self.project_root / "artifacts"
        
        # Ensure artifacts directory exists
        self.artifacts_dir.mkdir(exist_ok=True)
    
    def _check_required_files(self) -> None:
        """Check that all required data files exist."""
        required_files = [
            self.data_dir / "labeled_examples.csv",
            self.data_dir / "test_set_1.csv", 
            self.data_dir / "pages.csv",
            self.data_dir / "brief.txt"
        ]
        
        missing_files = [f for f in required_files if not f.exists()]
        if missing_files:
            self.console.print(f"[bold red]Error:[/bold red] Required files missing:")
            for f in missing_files:
                self.console.print(f"  - {f}")
            sys.exit(1)

    def load_brief(self) -> str:
        """Load the Crocs campaign brief text."""
        brief_path = self.data_dir / "brief.txt"
        if not brief_path.exists():
            raise FileNotFoundError(f"Brief file not found: {brief_path}")
        
        with open(brief_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    
    def clean_artifacts(self) -> None:
        """Clean and recreate artifacts directory."""
        if self.artifacts_dir.exists():
            self.console.print(f"🗑️  Cleaning existing artifacts directory: {self.artifacts_dir}")
            shutil.rmtree(self.artifacts_dir)
        
        self.artifacts_dir.mkdir(exist_ok=True)
        self.console.print(f"📁 Created fresh artifacts directory: {self.artifacts_dir}")

    def train_model(self) -> None:
        """Train the relevance model and save to artifacts directory."""
        self.console.print(Panel("[bold blue]Training Relevance Model[/bold blue]"))
        
        # Load brief
        brief_text = self.load_brief()
        self.console.print(f"📋 Loaded brief: {len(brief_text)} characters")
        
        # Load training data
        train_path = self.data_dir / "labeled_examples.csv"
        if not train_path.exists():
            raise FileNotFoundError(f"Training data not found: {train_path}")
        
        df_train = pd.read_csv(train_path)
        self.console.print(f"📊 Loaded training data: {len(df_train)} examples")
        self.console.print(f"   - Positive examples: {df_train['label'].sum()}")
        self.console.print(f"   - Negative examples: {len(df_train) - df_train['label'].sum()}")
        
        # Initialize and train model
        self.model = RelevanceModel()
        
        with self.console.status("[bold green]Training model...") as status:
            self.model.fit(str(train_path), brief_text)
        
        # Save model
        model_save_path = str(self.artifacts_dir)
        self.model.save(model_save_path)
        
        self.console.print(f"✅ Model trained and saved to {model_save_path}")
        self.console.print(f"   - Decision threshold: {self.model.threshold:.3f}")
        self.console.print()
    
    def evaluate_model(self) -> Dict[str, float]:
        """Evaluate the model on test set and return metrics."""
        self.console.print(Panel("[bold blue]Evaluating Model Performance vs Baseline[/bold blue]"))
        
        # Load model if not already loaded
        if self.model is None:
            self.model = RelevanceModel()
            model_load_path = str(self.artifacts_dir)
            self.model.load(model_load_path)
            self.console.print(f"📂 Loaded model from {model_load_path}")
        
        # Load test data
        test_path = self.data_dir / "test_set_1.csv"
        if not test_path.exists():
            raise FileNotFoundError(f"Test data not found: {test_path}")
        
        df_test = pd.read_csv(test_path)
        self.console.print(f"📊 Loaded test data: {len(df_test)} examples")
        
        # Train and evaluate baseline
        self.console.print("🔄 Training baseline (cosine similarity)...")
        baseline = BaselineCosine(encoder_name=self.model.encoder_name)
        brief_text = self.load_brief()
        train_path = self.data_dir / "labeled_examples.csv"
        baseline.fit(str(train_path), brief_text)
        
        # Generate predictions for both models
        y_true = df_test['label'].values
        
        # Learned model predictions
        learned_scores = []
        learned_pred = []
        
        # Baseline predictions  
        baseline_scores = []
        baseline_pred = []
        
        for snippet in track(df_test['snippet'], description="Evaluating both models..."):
            # Learned model
            result = self.model.predict(snippet)
            learned_scores.append(result['score'])
            learned_pred.append(result['bid'])
            
            # Baseline model
            score = baseline.predict(snippet)
            baseline_scores.append(score)
            baseline_pred.append(baseline.predict_binary(snippet))
        
        learned_scores = np.array(learned_scores)
        learned_pred = np.array(learned_pred)
        baseline_scores = np.array(baseline_scores)
        baseline_pred = np.array(baseline_pred)
        
        # Calculate metrics for both models
        learned_metrics = {
            'accuracy': accuracy_score(y_true, learned_pred) * 100,
            'roc_auc': roc_auc_score(y_true, learned_scores) * 100,
            'pr_auc': average_precision_score(y_true, learned_scores) * 100,
            'f1_score': f1_score(y_true, learned_pred) * 100
        }
        
        baseline_metrics = {
            'accuracy': accuracy_score(y_true, baseline_pred) * 100,
            'roc_auc': roc_auc_score(y_true, baseline_scores) * 100,
            'pr_auc': average_precision_score(y_true, baseline_scores) * 100,
            'f1_score': f1_score(y_true, baseline_pred) * 100
        }
        
        # Display comparison results
        self._display_comparison_results(baseline_metrics, learned_metrics, y_true, baseline_pred, learned_pred)
        
        return learned_metrics
    
    def _display_evaluation_results(self, metrics: Dict[str, float], y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """Display evaluation results in a formatted table."""
        
        # Create metrics table
        table = Table(title="Model Performance Metrics", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan", width=12)
        table.add_column("Value", style="green", width=10)
        table.add_column("Description", style="white", width=40)
        
        table.add_row("Accuracy", f"{metrics['accuracy']:.1f}%", "Overall prediction accuracy")
        table.add_row("ROC AUC", f"{metrics['roc_auc']:.1f}%", "Area under ROC curve")
        table.add_row("PR AUC", f"{metrics['pr_auc']:.1f}%", "Area under Precision-Recall curve")
        table.add_row("F1 Score", f"{metrics['f1_score']:.1f}%", "Harmonic mean of precision and recall")
        
        self.console.print(table)
        
        # Additional statistics
        total_examples = len(y_true)
        true_positives = np.sum((y_true == 1) & (y_pred == 1))
        false_positives = np.sum((y_true == 0) & (y_pred == 1))
        true_negatives = np.sum((y_true == 0) & (y_pred == 0))
        false_negatives = np.sum((y_true == 1) & (y_pred == 0))
        
        stats_text = Text()
        stats_text.append(f"Test Set Statistics:\n", style="bold")
        stats_text.append(f"  Total examples: {total_examples}\n")
        stats_text.append(f"  True Positives: {true_positives}\n", style="green")
        stats_text.append(f"  False Positives: {false_positives}\n", style="red")
        stats_text.append(f"  True Negatives: {true_negatives}\n", style="green")
        stats_text.append(f"  False Negatives: {false_negatives}\n", style="red")
        
        self.console.print(Panel(stats_text, title="Confusion Matrix Breakdown"))
        self.console.print()
    
    def _display_comparison_results(self, baseline_metrics: Dict[str, float], learned_metrics: Dict[str, float], 
                                  y_true: np.ndarray, baseline_pred: np.ndarray, learned_pred: np.ndarray) -> None:
        """Display baseline vs learned model comparison in a formatted table."""
        
        # Create comparison table
        table = Table(title="Baseline vs Learned Model Comparison", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan", width=12)
        table.add_column("Baseline", style="yellow", width=12, justify="right")
        table.add_column("Learned", style="green", width=12, justify="right")
        table.add_column("Improvement", style="blue", width=12, justify="right")
        
        metrics_order = ['accuracy', 'roc_auc', 'pr_auc', 'f1_score']
        metrics_names = ['Accuracy', 'ROC AUC', 'PR AUC', 'F1 Score']
        
        for metric, name in zip(metrics_order, metrics_names):
            baseline_val = baseline_metrics[metric]
            learned_val = learned_metrics[metric]
            improvement = learned_val - baseline_val
            improvement_str = f"+{improvement:.1f}%" if improvement > 0 else f"{improvement:.1f}%"
            
            table.add_row(
                name,
                f"{baseline_val:.1f}%",
                f"{learned_val:.1f}%", 
                improvement_str
            )
        
        self.console.print(table)
        
        # Additional statistics
        baseline_tp = np.sum((y_true == 1) & (baseline_pred == 1))
        baseline_fp = np.sum((y_true == 0) & (baseline_pred == 1))
        learned_tp = np.sum((y_true == 1) & (learned_pred == 1))
        learned_fp = np.sum((y_true == 0) & (learned_pred == 1))
        
        stats_text = Text()
        stats_text.append(f"Baseline: {baseline_tp} TP, {baseline_fp} FP | ", style="yellow")
        stats_text.append(f"Learned: {learned_tp} TP, {learned_fp} FP", style="green")
        
        self.console.print(Panel(stats_text, title="True/False Positives"))
        self.console.print()
    
    def generate_results(self) -> None:
        """Generate predictions for pages.csv and save to results.json."""
        self.console.print(Panel("[bold blue]Generating Results for Pages Dataset[/bold blue]"))
        
        # Load model if not already loaded
        if self.model is None:
            self.model = RelevanceModel()
            model_load_path = str(self.artifacts_dir)
            self.model.load(model_load_path)
            self.console.print(f"📂 Loaded model from {model_load_path}")
        
        # Load pages data
        pages_path = self.data_dir / "pages.csv"
        if not pages_path.exists():
            raise FileNotFoundError(f"Pages data not found: {pages_path}")
        
        df_pages = pd.read_csv(pages_path)
        self.console.print(f"📊 Loaded pages data: {len(df_pages)} pages")
        
        # Generate predictions
        results = []
        blocked_count = 0
        bid_count = 0
        
        for _, row in track(df_pages.iterrows(), total=len(df_pages), description="Processing pages..."):
            url = row['url']
            snippet = row['snippet']
            
            # Get prediction
            prediction = self.model.predict(snippet)
            
            # Track statistics
            if prediction['bid'] == 0 and prediction['score'] == 0.0:
                blocked_count += 1
            elif prediction['bid'] == 1:
                bid_count += 1
            
            # Add to results
            results.append({
                'url': url,
                'bid': prediction['bid'],
                'price': prediction['price'],
                'score': prediction['score']
            })
        
        # Save results to JSON
        results_path = self.project_root / "results.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Display summary
        no_bid_count = len(results) - bid_count - blocked_count
        avg_price = np.mean([r['price'] for r in results if r['bid'] == 1]) if bid_count > 0 else 0
        avg_score = np.mean([r['score'] for r in results])
        
        summary_table = Table(title="Results Summary", show_header=True, header_style="bold magenta")
        summary_table.add_column("Metric", style="cyan", width=20)
        summary_table.add_column("Count", style="green", width=10)
        summary_table.add_column("Percentage", style="yellow", width=12)
        
        summary_table.add_row("Total Pages", str(len(results)), "100.0%")
        summary_table.add_row("Bid (Win)", str(bid_count), f"{bid_count/len(results)*100:.1f}%")
        summary_table.add_row("No Bid (Low Score)", str(no_bid_count), f"{no_bid_count/len(results)*100:.1f}%")
        summary_table.add_row("Blocked (Brand Safety)", str(blocked_count), f"{blocked_count/len(results)*100:.1f}%")
        
        self.console.print(summary_table)
        
        self.console.print(f"\n📈 Average CPM for winning bids: ${avg_price:.3f}")
        self.console.print(f"📊 Average relevance score: {avg_score:.3f}")
        self.console.print(f"💾 Results saved to: {results_path}")
        self.console.print()


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate Crocs RTB relevance model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python eval.py                 # Full pipeline: train + eval + results
  python eval.py --train-only    # Only train the model
  python eval.py --eval-only     # Only evaluate (requires trained model)
  python eval.py --clean         # Clean artifacts + full pipeline
        """
    )
    
    parser.add_argument(
        '--train-only',
        action='store_true',
        help='Only train the model (skip evaluation and results generation)'
    )
    
    parser.add_argument(
        '--eval-only',
        action='store_true',
        help='Only evaluate the model (requires pre-trained model)'
    )
    
    parser.add_argument(
        '--clean',
        action='store_true',
        help='Delete and recreate artifacts directory before training'
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Run full pipeline (same as default)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.train_only and args.eval_only:
        parser.error("Cannot specify both --train-only and --eval-only")
    
    if args.eval_only and args.clean:
        parser.error("Cannot specify both --eval-only and --clean")
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    try:
        # Check required files first
        evaluator._check_required_files()
        
        # Clean artifacts if requested
        if args.clean:
            evaluator.clean_artifacts()
        
        if args.train_only:
            # Train only
            evaluator.train_model()
            
        elif args.eval_only:
            # Evaluate only
            metrics = evaluator.evaluate_model()
            
            # Print concise summary
            evaluator.console.print(Panel(
                f"Test metrics: Accuracy {metrics['accuracy']:.1f}%, "
                f"ROC AUC {metrics['roc_auc']:.1f}%, "
                f"PR AUC {metrics['pr_auc']:.1f}%, "
                f"F1 {metrics['f1_score']:.1f}%",
                title="📊 Test Results"
            ))
            
        else:
            # Full pipeline (default or --all)
            evaluator.console.print(Panel("[bold green]Crocs RTB Relevance System - Full Pipeline[/bold green]"))
            
            # Step 1: Train
            evaluator.train_model()
            
            # Step 2: Evaluate
            metrics = evaluator.evaluate_model()
            
            # Step 3: Generate results
            evaluator.generate_results()
            
            # Load results for summary
            results_path = evaluator.project_root / "results.json"
            with open(results_path, 'r') as f:
                results = json.load(f)
            
            total_rows = len(results)
            bid_count = sum(1 for r in results if r['bid'] == 1)
            blocked_count = sum(1 for r in results if r['bid'] == 0 and r['score'] == 0.0)
            avg_cpm = np.mean([r['price'] for r in results if r['bid'] == 1]) if bid_count > 0 else 0
            avg_score = np.mean([r['score'] for r in results])
            
            bid_rate = bid_count / total_rows * 100
            blocked_rate = blocked_count / total_rows * 100
            
            # Final concise summary
            evaluator.console.print(Panel(
                f"Test metrics: Accuracy {metrics['accuracy']:.1f}%, "
                f"ROC AUC {metrics['roc_auc']:.1f}%, "
                f"PR AUC {metrics['pr_auc']:.1f}%, "
                f"F1 {metrics['f1_score']:.1f}%\n"
                f"Results summary: {total_rows} rows, "
                f"{bid_rate:.1f}% bid rate, "
                f"{blocked_rate:.1f}% blocked rate, "
                f"${avg_cpm:.2f} avg CPM, "
                f"{avg_score:.3f} avg score",
                title="✅ Pipeline Complete"
            ))
    
    except FileNotFoundError as e:
        evaluator.console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)
    except Exception as e:
        evaluator.console.print(f"[bold red]Unexpected error:[/bold red] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
