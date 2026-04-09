#!/usr/bin/env python3
"""Training script for manifold diffusion model."""

import hydra
from omegaconf import DictConfig
import torch
from pathlib import Path

from src.models import ManifoldDiffusion
from src.data import create_diffusion_dataloaders
from src.training import DiffusionTrainer
from src.utils import setup_logging

@hydra.main(config_path="../configs", config_name="diffusion_config")
def main(cfg: DictConfig) -> None:
    """Main training function."""
    setup_logging()
    
    # Create data loaders
    train_loader, val_loader = create_diffusion_dataloaders(cfg.data)
    
    # Initialize model
    model = ManifoldDiffusion(cfg.model)
    
    # Initialize trainer
    trainer = DiffusionTrainer(model, cfg)
    
    # Train model
    trainer.fit(train_loader, val_loader)
    
    # Generate samples
    samples = model.sample(num_samples=16)
    save_path = Path("results") / "generated_samples.png"
    trainer.save_samples(samples, save_path)
    print(f"Generated samples saved to {save_path}")
    
    # Save model
    model_save_path = Path("models") / f"{cfg.model.name}_final.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

if __name__ == "__main__":
    main()
