#!/usr/bin/env python3
"""Training script for disinformation analyzer."""

import hydra
from omegaconf import DictConfig
import torch
from pathlib import Path

from src.models import DisinformationAnalyzer
from src.data import create_dataloaders
from src.training import Trainer
from src.utils import setup_logging

@hydra.main(config_path="../configs", config_name="analyzer_config")
def main(cfg: DictConfig) -> None:
    """Main training function."""
    setup_logging()
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_dataloaders(cfg.data)
    
    # Initialize model
    model = DisinformationAnalyzer(cfg.model)
    
    # Initialize trainer
    trainer = Trainer(model, cfg)
    
    # Train model
    trainer.fit(train_loader, val_loader)
    
    # Evaluate on test set
    test_metrics = trainer.evaluate(test_loader)
    print(f"Test metrics: {test_metrics}")
    
    # Save model
    save_path = Path("models") / f"{cfg.model.name}_final.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    main()
