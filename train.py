import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
import yaml
import numpy as np
from tqdm import tqdm
import logging
from datetime import datetime
import json
from pathlib import Path
import argparse

from data_loader import get_data_loaders, get_demo_data_loader
from model import create_model, count_parameters
from utils import setup_logging, save_checkpoint, load_checkpoint, EarlyStopping


class WeightedCrossEntropyLoss(nn.Module):
    """Weighted cross entropy loss for class imbalance"""
    
    def __init__(self, class_weights=None):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.class_weights = class_weights
        
    def forward(self, logits, targets):
        if self.class_weights is not None:
            device = logits.device
            class_weights = torch.tensor(list(self.class_weights.values()), 
                                       dtype=torch.float32).to(device)
            return nn.functional.cross_entropy(logits, targets, weight=class_weights)
        return nn.functional.cross_entropy(logits, targets)


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance"""
    
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, logits, targets):
        ce_loss = nn.functional.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def create_loss_function(loss_config, class_weights=None):
    """Create loss function based on configuration"""
    
    loss_type = loss_config['type']
    
    if loss_type == 'cross_entropy':
        return nn.CrossEntropyLoss()
    elif loss_type == 'weighted_cross_entropy':
        return WeightedCrossEntropyLoss(class_weights)
    elif loss_type == 'focal':
        return FocalLoss()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def train_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch, log_interval=10):
    """Train for one epoch"""
    
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}')
    
    for batch_idx, (images, targets) in enumerate(pbar):
        images, targets = images.to(device), targets.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass with mixed precision
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, targets)
        
        # Backward pass
        scaler.scale(loss).backward()
        
        # Gradient clipping
        if hasattr(scaler, 'get_scale'):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Optimizer step
        scaler.step(optimizer)
        scaler.update()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # Update progress bar
        current_acc = 100. * correct / total
        pbar.set_postfix({
            'Loss': f'{running_loss/(batch_idx+1):.4f}',
            'Acc': f'{current_acc:.2f}%'
        })
        
        # Log detailed progress
        if batch_idx % log_interval == 0:
            logging.info(f'Epoch {epoch+1}, Batch {batch_idx}/{len(train_loader)}, '
                        f'Loss: {loss.item():.4f}, Acc: {current_acc:.2f}%')
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def validate_epoch(model, val_loader, criterion, device):
    """Validate for one epoch"""
    
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc='Validation'):
            images, targets = images.to(device), targets.to(device)
            
            # Forward pass
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, targets)
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Store predictions and targets for metrics
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc, np.array(all_predictions), np.array(all_targets)


def main():
    parser = argparse.ArgumentParser(description='Train bone fracture classification model')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--demo', action='store_true', help='Run in demo mode with dummy data')
    parser.add_argument('--gpu', type=int, help='GPU ID to use')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override demo mode if specified
    if args.demo:
        config['app']['demo_mode'] = True
    
    # Setup logging
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'training_{timestamp}.log'
    
    setup_logging(log_file)
    logging.info(f"Starting training with config: {config}")
    
    # Setup device
    if args.gpu is not None:
        device = torch.device(f'cuda:{args.gpu}')
    elif config['training']['device'] == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(config['training']['device'])
    
    logging.info(f"Using device: {device}")
    
    # Create data loaders
    try:
        train_loader, val_loader, test_loader, class_names, class_weights = get_data_loaders(config)
        logging.info(f"Loaded dataset with classes: {class_names}")
    except Exception as e:
        logging.warning(f"Failed to load real dataset: {e}")
        logging.info("Using demo dataset...")
        train_loader, class_names, class_weights = get_demo_data_loader(config)
        val_loader = train_loader  # Use same loader for demo
        test_loader = train_loader
    
    # Create model
    model_type = config['model']['type']
    num_classes = config['model']['num_classes']
    pretrained = config['model']['pretrained']
    
    model = create_model(model_type, num_classes, pretrained)
    model = model.to(device)
    
    total_params, trainable_params = count_parameters(model)
    logging.info(f"Model: {model_type}")
    logging.info(f"Total parameters: {total_params:,}")
    logging.info(f"Trainable parameters: {trainable_params:,}")
    
    # Create loss function
    criterion = create_loss_function(config['loss'], class_weights)
    
    # Create optimizer
    learning_rate = config['training']['learning_rate']
    weight_decay = config['training']['weight_decay']
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Create learning rate scheduler
    epochs = config['training']['epochs']
    warmup_epochs = config['training']['warmup_epochs']
    
    # Warmup scheduler
    warmup_scheduler = LinearLR(
        optimizer, 
        start_factor=0.1, 
        end_factor=1.0, 
        total_iters=warmup_epochs
    )
    
    # Main scheduler
    main_scheduler = CosineAnnealingLR(
        optimizer, 
        T_max=epochs - warmup_epochs,
        eta_min=1e-6
    )
    
    # Mixed precision scaler
    scaler = GradScaler() if config['training']['mixed_precision'] else None
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=config['training']['early_stopping_patience'],
        min_delta=1e-4,
        restore_best_weights=True
    )
    
    # Create checkpoint directory
    checkpoint_dir = Path(config['checkpoint']['save_dir'])
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Training loop
    best_val_loss = float('inf')
    train_history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    start_epoch = 0
    
    # Resume from checkpoint if specified
    if args.resume:
        if os.path.exists(args.resume):
            start_epoch, best_val_loss = load_checkpoint(
                args.resume, model, optimizer, scaler, device
            )
            logging.info(f"Resumed training from epoch {start_epoch}")
        else:
            logging.warning(f"Checkpoint not found: {args.resume}")
    
    for epoch in range(start_epoch, epochs):
        logging.info(f"\nEpoch {epoch+1}/{epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch
        )
        
        # Validate
        val_loss, val_acc, val_preds, val_targets = validate_epoch(
            model, val_loader, criterion, device
        )
        
        # Update learning rate
        if epoch < warmup_epochs:
            warmup_scheduler.step()
        else:
            main_scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log results
        logging.info(f"Epoch {epoch+1} - "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, "
                    f"LR: {current_lr:.6f}")
        
        # Save history
        train_history['train_loss'].append(train_loss)
        train_history['train_acc'].append(train_acc)
        train_history['val_loss'].append(val_loss)
        train_history['val_acc'].append(val_acc)
        
        # Save checkpoint
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
        
        checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pth'
        save_checkpoint(
            epoch+1, model, optimizer, scaler, val_loss, 
            is_best, checkpoint_path, train_history
        )
        
        # Early stopping check
        if early_stopping(val_loss, model):
            logging.info(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    # Save final training history
    history_path = checkpoint_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(train_history, f, indent=2)
    
    logging.info("Training completed!")
    logging.info(f"Best validation loss: {best_val_loss:.4f}")
    
    # Load best model for final evaluation
    best_checkpoint = checkpoint_dir / 'best_model.pth'
    if best_checkpoint.exists():
        start_epoch, _ = load_checkpoint(best_checkpoint, model, optimizer, scaler, device)
        logging.info("Loaded best model for final evaluation")
        
        # Final test evaluation
        test_loss, test_acc, test_preds, test_targets = validate_epoch(
            model, test_loader, criterion, device
        )
        
        logging.info(f"Final Test Results:")
        logging.info(f"Test Loss: {test_loss:.4f}")
        logging.info(f"Test Accuracy: {test_acc:.2f}%")


if __name__ == "__main__":
    main()
