import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score
)
from sklearn.preprocessing import label_binarize
import yaml
import argparse
from pathlib import Path
import json
from tqdm import tqdm

from data_loader import get_data_loaders, get_demo_data_loader
from model import create_model
from utils import load_checkpoint, get_device, create_directory


class ModelEvaluator:
    """Comprehensive model evaluation"""
    
    def __init__(self, model, device, class_names):
        self.model = model
        self.device = device
        self.class_names = class_names
        self.num_classes = len(class_names)
        
    def predict(self, data_loader):
        """Get predictions and true labels"""
        self.model.eval()
        all_predictions = []
        all_probabilities = []
        all_targets = []
        
        with torch.no_grad():
            for images, targets in tqdm(data_loader, desc='Predicting'):
                images = images.to(self.device)
                
                outputs = self.model(images)
                probabilities = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(probabilities, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                all_targets.extend(targets.numpy())
        
        return np.array(all_predictions), np.array(all_probabilities), np.array(all_targets)
    
    def compute_metrics(self, y_true, y_pred, y_prob):
        """Compute comprehensive metrics"""
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average='macro')
        f1_weighted = f1_score(y_true, y_pred, average='weighted')
        f1_per_class = f1_score(y_true, y_pred, average=None)
        
        # Per-class metrics
        report = classification_report(y_true, y_pred, target_names=self.class_names, 
                                     output_dict=True, zero_division=0)
        
        # ROC-AUC (one-vs-rest)
        if self.num_classes > 2:
            y_true_bin = label_binarize(y_true, classes=range(self.num_classes))
            roc_auc = roc_auc_score(y_true_bin, y_prob, average='macro', multi_class='ovr')
            roc_auc_per_class = roc_auc_score(y_true_bin, y_prob, average=None, multi_class='ovr')
        else:
            roc_auc = roc_auc_score(y_true, y_prob[:, 1])
            roc_auc_per_class = [roc_auc]
        
        # Average Precision (one-vs-rest)
        if self.num_classes > 2:
            avg_precision = average_precision_score(y_true_bin, y_prob, average='macro')
            avg_precision_per_class = average_precision_score(y_true_bin, y_prob, average=None)
        else:
            avg_precision = average_precision_score(y_true, y_prob[:, 1])
            avg_precision_per_class = [avg_precision]
        
        metrics = {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'f1_per_class': dict(zip(self.class_names, f1_per_class)),
            'roc_auc_macro': roc_auc,
            'roc_auc_per_class': dict(zip(self.class_names, roc_auc_per_class)),
            'avg_precision_macro': avg_precision,
            'avg_precision_per_class': dict(zip(self.class_names, avg_precision_per_class)),
            'classification_report': report
        }
        
        return metrics
    
    def plot_confusion_matrix(self, y_true, y_pred, save_path=None):
        """Plot confusion matrix"""
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return cm
    
    def plot_roc_curves(self, y_true, y_prob, save_path=None):
        """Plot ROC curves for each class"""
        
        if self.num_classes > 2:
            y_true_bin = label_binarize(y_true, classes=range(self.num_classes))
            
            plt.figure(figsize=(12, 8))
            
            for i, class_name in enumerate(self.class_names):
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
                roc_auc = roc_auc_score(y_true_bin[:, i], y_prob[:, i])
                
                plt.plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc:.3f})')
            
            plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curves (One-vs-Rest)')
            plt.legend(loc='lower right')
            plt.grid(True, alpha=0.3)
        else:
            # Binary classification
            fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
            roc_auc = roc_auc_score(y_true, y_prob[:, 1])
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.3f})')
            plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend(loc='lower right')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_precision_recall_curves(self, y_true, y_prob, save_path=None):
        """Plot Precision-Recall curves"""
        
        if self.num_classes > 2:
            y_true_bin = label_binarize(y_true, classes=range(self.num_classes))
            
            plt.figure(figsize=(12, 8))
            
            for i, class_name in enumerate(self.class_names):
                precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_prob[:, i])
                avg_prec = average_precision_score(y_true_bin[:, i], y_prob[:, i])
                
                plt.plot(recall, precision, label=f'{class_name} (AP = {avg_prec:.3f})')
            
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curves (One-vs-Rest)')
            plt.legend(loc='lower left')
            plt.grid(True, alpha=0.3)
        else:
            # Binary classification
            precision, recall, _ = precision_recall_curve(y_true, y_prob[:, 1])
            avg_prec = average_precision_score(y_true, y_prob[:, 1])
            
            plt.figure(figsize=(8, 6))
            plt.plot(recall, precision, label=f'PR Curve (AP = {avg_prec:.3f})')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.legend(loc='lower left')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_class_distribution(self, y_true, y_pred, save_path=None):
        """Plot class distribution (true vs predicted)"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # True distribution
        unique_true, counts_true = np.unique(y_true, return_counts=True)
        ax1.bar(range(len(self.class_names)), 
                [counts_true[np.where(unique_true == i)[0][0]] if i in unique_true else 0 
                 for i in range(len(self.class_names))])
        ax1.set_title('True Class Distribution')
        ax1.set_xlabel('Class')
        ax1.set_ylabel('Count')
        ax1.set_xticks(range(len(self.class_names)))
        ax1.set_xticklabels(self.class_names, rotation=45)
        
        # Predicted distribution
        unique_pred, counts_pred = np.unique(y_pred, return_counts=True)
        ax2.bar(range(len(self.class_names)), 
                [counts_pred[np.where(unique_pred == i)[0][0]] if i in unique_pred else 0 
                 for i in range(len(self.class_names))])
        ax2.set_title('Predicted Class Distribution')
        ax2.set_xlabel('Class')
        ax2.set_ylabel('Count')
        ax2.set_xticks(range(len(self.class_names)))
        ax2.set_xticklabels(self.class_names, rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_evaluation_report(self, data_loader, output_dir):
        """Generate comprehensive evaluation report"""
        
        create_directory(output_dir)
        
        # Get predictions
        y_pred, y_prob, y_true = self.predict(data_loader)
        
        # Compute metrics
        metrics = self.compute_metrics(y_true, y_pred, y_prob)
        
        # Save metrics as JSON
        metrics_path = os.path.join(output_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        
        # Print summary
        print("="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"F1 Score (Macro): {metrics['f1_macro']:.4f}")
        print(f"F1 Score (Weighted): {metrics['f1_weighted']:.4f}")
        print(f"ROC-AUC (Macro): {metrics['roc_auc_macro']:.4f}")
        print(f"Average Precision (Macro): {metrics['avg_precision_macro']:.4f}")
        print("\nPer-Class Metrics:")
        for class_name in self.class_names:
            print(f"  {class_name}:")
            print(f"    F1: {metrics['f1_per_class'][class_name]:.4f}")
            print(f"    ROC-AUC: {metrics['roc_auc_per_class'][class_name]:.4f}")
            print(f"    Avg Precision: {metrics['avg_precision_per_class'][class_name]:.4f}")
        
        # Generate plots
        print("\nGenerating visualizations...")
        
        # Confusion Matrix
        cm_path = os.path.join(output_dir, 'confusion_matrix.png')
        self.plot_confusion_matrix(y_true, y_pred, cm_path)
        
        # ROC Curves
        roc_path = os.path.join(output_dir, 'roc_curves.png')
        self.plot_roc_curves(y_true, y_prob, roc_path)
        
        # Precision-Recall Curves
        pr_path = os.path.join(output_dir, 'precision_recall_curves.png')
        self.plot_precision_recall_curves(y_true, y_prob, pr_path)
        
        # Class Distribution
        dist_path = os.path.join(output_dir, 'class_distribution.png')
        self.plot_class_distribution(y_true, y_pred, dist_path)
        
        # Save detailed classification report
        report_path = os.path.join(output_dir, 'classification_report.txt')
        with open(report_path, 'w') as f:
            f.write("DETAILED CLASSIFICATION REPORT\n")
            f.write("="*50 + "\n\n")
            for class_name in self.class_names:
                class_metrics = metrics['classification_report'][class_name]
                f.write(f"Class: {class_name}\n")
                f.write(f"  Precision: {class_metrics['precision']:.4f}\n")
                f.write(f"  Recall: {class_metrics['recall']:.4f}\n")
                f.write(f"  F1-Score: {class_metrics['f1-score']:.4f}\n")
                f.write(f"  Support: {class_metrics['support']}\n\n")
        
        print(f"Evaluation complete! Results saved to: {output_dir}")
        
        return metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate bone fracture classification model')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str, choices=['train', 'val', 'test'], 
                       default='test', help='Dataset to evaluate')
    parser.add_argument('--output', type=str, default='results', help='Output directory')
    parser.add_argument('--demo', action='store_true', help='Run in demo mode')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override demo mode if specified
    if args.demo:
        config['app']['demo_mode'] = True
    
    # Setup device
    device = get_device()
    
    # Load data
    try:
        train_loader, val_loader, test_loader, class_names, _ = get_data_loaders(config)
        if args.dataset == 'train':
            data_loader = train_loader
        elif args.dataset == 'val':
            data_loader = val_loader
        else:
            data_loader = test_loader
    except Exception as e:
        print(f"Failed to load real dataset: {e}")
        print("Using demo dataset...")
        train_loader, class_names, _ = get_demo_data_loader(config)
        data_loader = train_loader
    
    # Create model
    model_type = config['model']['type']
    num_classes = config['model']['num_classes']
    pretrained = False  # Don't use pretrained weights when loading checkpoint
    
    model = create_model(model_type, num_classes, pretrained)
    model = model.to(device)
    
    # Load checkpoint
    if os.path.exists(args.checkpoint):
        load_checkpoint(args.checkpoint, model, None, None, device)
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    
    # Create evaluator
    evaluator = ModelEvaluator(model, device, class_names)
    
    # Generate evaluation report
    metrics = evaluator.generate_evaluation_report(data_loader, args.output)
    
    print("\nEvaluation completed successfully!")


if __name__ == "__main__":
    main()
