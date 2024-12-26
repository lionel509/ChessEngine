import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional
from pathlib import Path
import logging
from metrics import ChessMetrics

class ChessTrainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        checkpoint_dir: Optional[str] = None
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else Path('./checkpoints')
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.metrics = ChessMetrics()
        self.current_epoch = 0
        
    def train_epoch(self, train_loader: DataLoader) -> Dict:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, (positions, moves) in enumerate(train_loader):
            positions = positions.to(self.device)
            moves = moves.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(positions)
            loss = self.criterion(outputs, moves)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return {'loss': total_loss / len(train_loader)}
    
    def validate(self, val_loader: DataLoader) -> Dict:
        """Validate model performance"""
        self.model.eval()
        total_loss = 0.0
        correct_moves = 0
        total_moves = 0
        
        with torch.no_grad():
            for positions, moves in val_loader:
                positions = positions.to(self.device)
                moves = moves.to(self.device)
                
                outputs = self.model(positions)
                loss = self.criterion(outputs, moves)
                
                total_loss += loss.item()
                predicted = outputs.argmax(dim=1)
                correct_moves += (predicted == moves).sum().item()
                total_moves += moves.size(0)
                
        return {
            'val_loss': total_loss / len(val_loader),
            'accuracy': correct_moves / total_moves
        }
    
    def save_checkpoint(self, metrics: Dict):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics
        }
        path = self.checkpoint_dir / f'checkpoint_epoch_{self.current_epoch}.pt'
        torch.save(checkpoint, path)
        
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.current_epoch = checkpoint['epoch']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint.get('metrics', {})
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        save_frequency: int = 1
    ):
        """Full training loop"""
        best_val_loss = float('inf')
        
        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            
            # Training phase
            train_metrics = self.train_epoch(train_loader)
            
            # Validation phase
            val_metrics = self.validate(val_loader)
            
            # Combine metrics
            metrics = {**train_metrics, **val_metrics}
            
            # Log progress
            logging.info(f'Epoch {epoch+1}/{num_epochs}')
            for key, value in metrics.items():
                logging.info(f'{key}: {value:.4f}')
            
            # Save checkpoint
            if (epoch + 1) % save_frequency == 0:
                self.save_checkpoint(metrics)
                
            # Save best model
            if val_metrics['val_loss'] < best_val_loss:
                best_val_loss = val_metrics['val_loss']
                self.save_checkpoint({'best_epoch': epoch, **metrics})