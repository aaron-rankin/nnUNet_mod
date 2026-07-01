#!/usr/bin/env python3
"""
Evaluation and Comparison Script for nnUNet Models
Compares new model performance against baseline and generates detailed reports.
"""

import os
import json
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np


def run_predictions(dataset_id: int, fold: int, trainer: str, planner: str, 
                   input_dir: str, output_dir: str, device: str = 'cuda') -> bool:
    """
    Run predictions for a specific model configuration.
    
    Args:
        dataset_id: Dataset ID (e.g., 502)
        fold: Fold number
        trainer: Trainer class name
        planner: Planner name
        input_dir: Input images directory
        output_dir: Output predictions directory
        device: Device to use (cuda/cpu)
    
    Returns:
        True if successful, False otherwise
    """
    cmd = [
        'nnUNetv2_predict',
        '-d', str(dataset_id),
        '-i', input_dir,
        '-o', output_dir,
        '-f', str(fold),
        '-tr', trainer,
        '-p', planner,
        '-device', device
    ]
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"Prediction completed successfully for fold {fold}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running prediction for fold {fold}:")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False


def evaluate_predictions(dataset_id: int, input_dir: str, output_dir: str,
                        gt_dir: str) -> Optional[Dict]:
    """
    Run nnUNet evaluation on predictions.
    
    Returns:
        Dictionary with evaluation metrics or None if failed
    """
    cmd = [
        'nnUNetv2_evaluate_predictions',
        '-d', str(dataset_id),
        '-i', input_dir,
        '-o', output_dir,
        '-gt', gt_dir
    ]
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("Evaluation completed successfully")
        
        # Parse metrics from output
        metrics = parse_evaluation_output(result.stdout)
        return metrics
    except subprocess.CalledProcessError as e:
        print(f"Error running evaluation:")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return None


def parse_evaluation_output(output: str) -> Dict:
    """Parse nnUNet evaluation output to extract metrics."""
    metrics = {
        'dice': {},
        'mrae': {},
        'mean_dice': 0.0,
        'mean_mrae': 0.0
    }
    
    # Parse Dice scores
    # Expected format: "Class X: Dice = Y"
    for line in output.split('\n'):
        if 'Dice' in line and ':' in line:
            parts = line.split(':')
            if len(parts) >= 2:
                class_name = parts[0].strip()
                dice_str = parts[1].split('=')[-1].strip()
                try:
                    dice_val = float(dice_str)
                    metrics['dice'][class_name] = dice_val
                except:
                    pass
        
        if 'Mean Dice' in line:
            parts = line.split(':')
            if len(parts) >= 2:
                try:
                    metrics['mean_dice'] = float(parts[1].strip())
                except:
                    pass
    
    return metrics


def aggregate_metrics_across_folds(metrics_list: List[Dict]) -> Dict:
    """Aggregate metrics across multiple folds."""
    if not metrics_list:
        return {}
    
    aggregated = {
        'dice': {},
        'mrae': {},
        'mean_dice_mean': 0.0,
        'mean_dice_std': 0.0
    }
    
    # Aggregate per-class Dice
    all_classes = set()
    for metrics in metrics_list:
        all_classes.update(metrics.get('dice', {}).keys())
    
    for cls in all_classes:
        values = [m['dice'].get(cls, 0) for m in metrics_list if cls in m.get('dice', {})]
        if values:
            aggregated['dice'][cls] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
    
    # Aggregate mean Dice
    mean_dices = [m.get('mean_dice', 0) for m in metrics_list]
    aggregated['mean_dice_mean'] = np.mean(mean_dices)
    aggregated['mean_dice_std'] = np.std(mean_dices)
    
    return aggregated


def compare_models(baseline_metrics: Dict, new_metrics: Dict) -> Dict:
    """Compare new model against baseline."""
    comparison = {
        'overall_improvement': new_metrics.get('mean_dice_mean', 0) - baseline_metrics.get('mean_dice_mean', 0),
        'per_class': {},
        'winner': 'tie'
    }
    
    # Compare per-class
    for cls in baseline_metrics.get('dice', {}):
        if cls in new_metrics.get('dice', {}):
            baseline_mean = baseline_metrics['dice'][cls].get('mean', 0)
            new_mean = new_metrics['dice'][cls].get('mean', 0)
            
            comparison['per_class'][cls] = {
                'baseline': baseline_mean,
                'new': new_mean,
                'delta': new_mean - baseline_mean,
                'percent_change': ((new_mean - baseline_mean) / baseline_mean * 100) if baseline_mean > 0 else 0
            }
    
    # Determine winner
    if comparison['overall_improvement'] > 0.005:  # 0.5% threshold
        comparison['winner'] = 'new'
    elif comparison['overall_improvement'] < -0.005:
        comparison['winner'] = 'baseline'
    else:
        comparison['winner'] = 'tie'
    
    return comparison


def print_comparison_report(baseline_trainer: str, new_trainer: str, 
                           baseline_metrics: Dict, new_metrics: Dict,
                           comparison: Dict):
    """Print a formatted comparison report."""
    print("\n" + "="*80)
    print("MODEL COMPARISON REPORT")
    print("="*80)
    print(f"Baseline: {baseline_trainer}")
    print(f"New Model: {new_trainer}")
    print("="*80)
    
    # Overall metrics
    print(f"\nOverall Mean Dice:")
    print(f"  Baseline: {baseline_metrics.get('mean_dice_mean', 0):.4f} ± {baseline_metrics.get('mean_dice_std', 0):.4f}")
    print(f"  New:      {new_metrics.get('mean_dice_mean', 0):.4f} ± {new_metrics.get('mean_dice_std', 0):.4f}")
    print(f"  Delta:    {comparison['overall_improvement']:+.4f} ({comparison['overall_improvement']/baseline_metrics.get('mean_dice_mean', 1)*100:+.2f}%)")
    
    # Winner announcement
    winner = comparison['winner']
    if winner == 'new':
        print(f"\n🏆 WINNER: {new_trainer} (improvement: {comparison['overall_improvement']:.4f})")
    elif winner == 'baseline':
        print(f"\n🏆 WINNER: {baseline_trainer} (baseline is better by {abs(comparison['overall_improvement']):.4f})")
    else:
        print(f"\n🤝 TIE: Models perform similarly (difference: {abs(comparison['overall_improvement']):.4f})")
    
    # Per-class comparison
    print(f"\nPer-Class Dice Comparison:")
    print("-"*80)
    print(f"{'Class':<15} {'Baseline':<12} {'New':<12} {'Delta':<12} {'Change %':<12}")
    print("-"*80)
    
    for cls, values in sorted(comparison['per_class'].items()):
        print(f"{cls:<15} {values['baseline']:.4f}       {values['new']:.4f}       "
              f"{values['delta']:+.4f}      {values['percent_change']:+.2f}%")
    
    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate and compare nnUNet models against baseline'
    )
    
    parser.add_argument('--dataset', type=int, default=502,
                       help='Dataset ID (default: 502)')
    parser.add_argument('--baseline-trainer', type=str, default='nnUNetTrainer',
                       help='Baseline trainer name')
    parser.add_argument('--new-trainer', type=str, required=True,
                       help='New trainer name to evaluate')
    parser.add_argument('--planner', type=str, default='nnUNetResEncUNetMPlans',
                       help='Planner name')
    parser.add_argument('--folds', type=str, default='0,1,2,3,4',
                       help='Comma-separated fold numbers (default: 0,1,2,3,4)')
    parser.add_argument('--input-dir', type=str, required=True,
                       help='Directory with input images')
    parser.add_argument('--gt-dir', type=str, required=True,
                       help='Directory with ground truth labels')
    parser.add_argument('--output-dir', type=str, default='./evaluation_results',
                       help='Directory for output results')
    parser.add_argument('--skip-prediction', action='store_true',
                       help='Skip prediction step (use existing predictions)')
    parser.add_argument('--report-only', action='store_true',
                       help='Only generate comparison report (requires previous evaluation)')
    
    args = parser.parse_args()
    
    # Parse folds
    folds = [int(f) for f in args.folds.split(',')]
    
    # Create output directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    baseline_metrics_list = []
    new_metrics_list = []
    
    if not args.report_only:
        print(f"\nEvaluating {len(folds)} folds...")
        print("="*80)
        
        for fold in folds:
            print(f"\n--- Fold {fold} ---")
            
            # Run predictions and evaluation for baseline
            if not args.skip_prediction:
                baseline_pred_dir = output_dir / f'baseline_fold{fold}'
                baseline_pred_dir.mkdir(exist_ok=True)
                
                success = run_predictions(
                    args.dataset, fold, args.baseline_trainer, args.planner,
                    args.input_dir, str(baseline_pred_dir)
                )
                
                if success:
                    metrics = evaluate_predictions(
                        args.dataset, str(baseline_pred_dir), 
                        str(output_dir / f'baseline_eval_fold{fold}'),
                        args.gt_dir
                    )
                    if metrics:
                        baseline_metrics_list.append(metrics)
            
            # Run predictions and evaluation for new model
            if not args.skip_prediction:
                new_pred_dir = output_dir / f'new_fold{fold}'
                new_pred_dir.mkdir(exist_ok=True)
                
                success = run_predictions(
                    args.dataset, fold, args.new_trainer, args.planner,
                    args.input_dir, str(new_pred_dir)
                )
                
                if success:
                    metrics = evaluate_predictions(
                        args.dataset, str(new_pred_dir),
                        str(output_dir / f'new_eval_fold{fold}'),
                        args.gt_dir
                    )
                    if metrics:
                        new_metrics_list.append(metrics)
        
        # Save raw metrics
        with open(output_dir / 'baseline_metrics.json', 'w') as f:
            json.dump(baseline_metrics_list, f, indent=2)
        with open(output_dir / 'new_metrics.json', 'w') as f:
            json.dump(new_metrics_list, f, indent=2)
    
    else:
        # Load existing metrics
        try:
            with open(output_dir / 'baseline_metrics.json', 'r') as f:
                baseline_metrics_list = json.load(f)
            with open(output_dir / 'new_metrics.json', 'r') as f:
                new_metrics_list = json.load(f)
            print(f"Loaded metrics for {len(baseline_metrics_list)} baseline folds, "
                  f"{len(new_metrics_list)} new model folds")
        except FileNotFoundError:
            print("Error: Could not find existing metrics files. Run without --report-only first.")
            return
    
    # Aggregate metrics
    baseline_metrics = aggregate_metrics_across_folds(baseline_metrics_list)
    new_metrics = aggregate_metrics_across_folds(new_metrics_list)
    
    # Compare
    comparison = compare_models(baseline_metrics, new_metrics)
    
    # Print report
    print_comparison_report(
        args.baseline_trainer, args.new_trainer,
        baseline_metrics, new_metrics,
        comparison
    )
    
    # Save comparison
    report = {
        'baseline_trainer': args.baseline_trainer,
        'new_trainer': args.new_trainer,
        'baseline_metrics': baseline_metrics,
        'new_metrics': new_metrics,
        'comparison': comparison,
        'folds_evaluated': len(baseline_metrics_list)
    }
    
    with open(output_dir / 'comparison_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nReport saved to: {output_dir / 'comparison_report.json'}")


if __name__ == '__main__':
    main()
