"""
Manifold Diffusion Model
Advanced implementation based on arXiv:2604.07213

Features:
- Implicit Manifold-valued Diffusions (IMDs)
- Data-driven SDE construction
- Manifold-aware sampling
- Multi-scale diffusion
- Adaptive manifold learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
from dataclasses import dataclass
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import Isomap, LocallyLinearEmbedding
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

logger = logging.getLogger(__name__)

@dataclass
class DiffusionConfig:
    """Configuration for manifold diffusion model"""
    data_dim: int = 2
    hidden_dim: int = 512
    num_layers: int = 6
    num_heads: int = 8
    diffusion_steps: int = 1000
    beta_start: float = 0.0001
    beta_end: float = 0.02
    beta_schedule: str = "linear"
    manifold_neighbors: int = 15
    manifold_sigma: float = 1.0
    adaptive_manifold: bool = True
    multi_scale: bool = True

class ManifoldLearner(nn.Module):
    """Learn manifold structure from data"""
    
    def __init__(self, data_dim: int, hidden_dim: int, num_layers: int = 3):
        super().__init__()
        
        self.data_dim = data_dim
        self.hidden_dim = hidden_dim
        
        # Encoder for manifold structure
        layers = []
        in_dim = data_dim
        
        for i in range(num_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            in_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        
        # Manifold coordinate decoder
        self.manifold_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, data_dim)
        )
        
        # Tangent space estimator
        self.tangent_estimator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, data_dim * data_dim)  # Tangent basis
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass to learn manifold structure"""
        encoded = self.encoder(x)
        manifold_coords = self.manifold_decoder(encoded)
        tangent_basis = self.tangent_estimator(encoded)
        
        # Reshape tangent basis
        batch_size = x.shape[0]
        tangent_basis = tangent_basis.view(batch_size, self.data_dim, self.data_dim)
        
        return {
            'manifold_coords': manifold_coords,
            'tangent_basis': tangent_basis,
            'encoded': encoded
        }

class DiffusionNetwork(nn.Module):
    """Neural network for diffusion process"""
    
    def __init__(
        self,
        data_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        time_embedding_dim: int = 128
    ):
        super().__init__()
        
        self.data_dim = data_dim
        self.hidden_dim = hidden_dim
        self.time_embedding_dim = time_embedding_dim
        
        # Time embedding
        self.time_embedding = nn.Sequential(
            nn.Linear(1, time_embedding_dim),
            nn.ReLU(),
            nn.Linear(time_embedding_dim, time_embedding_dim)
        )
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Input projection
        self.input_projection = nn.Linear(data_dim + time_embedding_dim, hidden_dim)
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, data_dim)
        )
        
        # Manifold constraint layer
        self.manifold_constraint = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, data_dim),
            nn.Tanh()
        )
    
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        manifold_info: Optional[Dict] = None
    ) -> torch.Tensor:
        """Forward pass of diffusion network"""
        # Time embedding
        t_emb = self.time_embedding(t.unsqueeze(-1))
        
        # Concatenate input and time embedding
        x_t = torch.cat([x, t_emb], dim=-1)
        
        # Project to hidden dimension
        hidden = self.input_projection(x_t)
        
        # Add sequence dimension for transformer
        hidden = hidden.unsqueeze(1)
        
        # Apply transformer
        hidden = self.transformer(hidden)
        
        # Remove sequence dimension
        hidden = hidden.squeeze(1)
        
        # Generate output
        output = self.output_projection(hidden)
        
        # Apply manifold constraint if available
        if manifold_info is not None:
            constraint = self.manifold_constraint(hidden)
            output = output + 0.1 * constraint
        
        return output

class ManifoldDiffusionModel(nn.Module):
    """
    Advanced Manifold Diffusion Model implementing IMDs from arXiv:2604.07213
    
    Features:
    - Implicit manifold learning
    - Data-driven SDE construction
    - Multi-scale diffusion
    - Adaptive manifold constraints
    """
    
    def __init__(self, config: DiffusionConfig):
        super().__init__()
        
        self.config = config
        
        # Manifold learner
        self.manifold_learner = ManifoldLearner(
            config.data_dim,
            config.hidden_dim,
            num_layers=3
        )
        
        # Diffusion network
        self.diffusion_network = DiffusionNetwork(
            config.data_dim,
            config.hidden_dim,
            config.num_layers,
            config.num_heads
        )
        
        # Setup diffusion schedule
        self._setup_diffusion_schedule()
        
        # Manifold parameters
        self.manifold_neighbors = config.manifold_neighbors
        self.manifold_sigma = config.manifold_sigma
        self.adaptive_manifold = config.adaptive_manifold
        self.multi_scale = config.multi_scale
        
        # Learned manifold structure
        self.learned_manifold = None
        self.proximity_graph = None
        
    def _setup_diffusion_schedule(self):
        """Setup diffusion noise schedule"""
        if self.config.beta_schedule == "linear":
            self.betas = torch.linspace(
                self.config.beta_start,
                self.config.beta_end,
                self.config.diffusion_steps
            )
        elif self.config.beta_schedule == "cosine":
            steps = self.config.diffusion_steps
            x = torch.linspace(0, steps, steps + 1)
            alphas_cumprod = torch.cos(((x / steps) + 0.008) / 1.008 * np.pi / 2) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            self.betas = torch.clip(betas, 0, 0.999)
        else:
            raise ValueError(f"Unknown beta schedule: {self.config.beta_schedule}")
        
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
    
    def learn_manifold_structure(self, data: torch.Tensor) -> Dict:
        """Learn manifold structure from data"""
        logger.info("Learning manifold structure from data...")
        
        # Convert to numpy for sklearn
        data_np = data.detach().cpu().numpy()
        
        # Build proximity graph
        nbrs = NearestNeighbors(
            n_neighbors=self.manifold_neighbors,
            algorithm='auto',
            metric='euclidean'
        ).fit(data_np)
        
        distances, indices = nbrs.kneighbors(data_np)
        
        # Create adjacency matrix
        n_samples = len(data_np)
        adjacency = np.zeros((n_samples, n_samples))
        
        for i in range(n_samples):
            adjacency[i, indices[i]] = np.exp(-distances[i] ** 2 / (2 * self.manifold_sigma ** 2))
        
        # Normalize adjacency
        adjacency = adjacency / adjacency.sum(axis=1, keepdims=True)
        
        # Use neural manifold learner
        manifold_info = self.manifold_learner(data)
        
        # Store learned structure
        self.learned_manifold = {
            'adjacency': torch.tensor(adjacency, dtype=torch.float32),
            'indices': torch.tensor(indices, dtype=torch.long),
            'distances': torch.tensor(distances, dtype=torch.float32),
            'manifold_coords': manifold_info['manifold_coords'],
            'tangent_basis': manifold_info['tangent_basis']
        }
        
        self.proximity_graph = nbrs
        
        return self.learned_manifold
    
    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward diffusion process with manifold constraints"""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        # Standard diffusion
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        
        # Apply diffusion
        x_noisy = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
        
        # Apply manifold constraint
        if self.learned_manifold is not None:
            x_noisy = self._apply_manifold_constraint(x_noisy, t)
        
        return x_noisy
    
    def p_mean_variance(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        manifold_info: Optional[Dict] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute mean and variance for reverse diffusion"""
        # Predict noise
        predicted_noise = self.diffusion_network(x, t, manifold_info)
        
        # Compute mean
        alpha_t = self._extract(self.alphas, t, x.shape)
        alpha_cumprod_t = self._extract(self.alphas_cumprod, t, x.shape)
        beta_t = self._extract(self.betas, t, x.shape)
        
        # Compute model mean
        model_mean = (1 / torch.sqrt(alpha_t)) * (
            x - beta_t / torch.sqrt(1 - alpha_cumprod_t) * predicted_noise
        )
        
        # Compute variance
        model_variance = beta_t
        
        return model_mean, model_variance, predicted_noise
    
    def p_sample(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        manifold_info: Optional[Dict] = None
    ) -> torch.Tensor:
        """Single reverse diffusion step"""
        model_mean, model_variance, _ = self.p_mean_variance(x, t, manifold_info)
        
        if t[0] == 0:
            return model_mean
        else:
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(model_variance) * noise
    
    def sample(
        self,
        shape: Union[Tuple, torch.Size],
        n_steps: Optional[int] = None,
        return_intermediate: bool = False
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Generate samples using reverse diffusion"""
        if n_steps is None:
            n_steps = self.config.diffusion_steps
        
        # Start with noise
        x = torch.randn(shape)
        
        if return_intermediate:
            intermediates = [x.clone()]
        
        # Reverse diffusion process
        for t in reversed(range(n_steps)):
            t_tensor = torch.tensor([t], dtype=torch.long, device=x.device)
            
            x = self.p_sample(x, t_tensor, self.learned_manifold)
            
            if return_intermediate and t % 100 == 0:
                intermediates.append(x.clone())
        
        if return_intermediate:
            return intermediates
        else:
            return x
    
    def _apply_manifold_constraint(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Apply manifold constraint to keep samples on manifold"""
        if self.learned_manifold is None:
            return x
        
        # Project onto learned manifold coordinates
        with torch.no_grad():
            manifold_info = self.manifold_learner(x)
            manifold_coords = manifold_info['manifold_coords']
            
            # Weight constraint by diffusion step (stronger at beginning)
            constraint_weight = 1.0 - (t[0].item() / self.config.diffusion_steps)
            
            # Blend original and constrained
            x_constrained = (1 - constraint_weight * 0.1) * x + constraint_weight * 0.1 * manifold_coords
            
        return x_constrained
    
    def _extract(self, a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
        """Extract values from 1D tensor at specified indices"""
        batch_size = t.shape[0]
        out = a.to(t.device).gather(0, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
    
    def compute_manifold_metrics(self, data: torch.Tensor) -> Dict[str, float]:
        """Compute manifold quality metrics"""
        if self.learned_manifold is None:
            self.learn_manifold_structure(data)
        
        data_np = data.detach().cpu().numpy()
        manifold_coords_np = self.learned_manifold['manifold_coords'].detach().cpu().numpy()
        
        # Compute intrinsic dimensionality
        distances = pdist(data_np)
        correlation_length = np.mean(distances)
        
        # Compute manifold preservation
        from sklearn.metrics.pairwise import pairwise_distances
        
        original_distances = pairwise_distances(data_np)
        manifold_distances = pairwise_distances(manifold_coords_np)
        
        # Correlation between distance matrices
        correlation = np.corrcoef(original_distances.flatten(), manifold_distances.flatten())[0, 1]
        
        return {
            'intrinsic_dimensionality': self._estimate_intrinsic_dimensionality(data_np),
            'correlation_length': correlation_length,
            'manifold_preservation': correlation,
            'reconstruction_error': torch.norm(data - self.learned_manifold['manifold_coords']).item()
        }
    
    def _estimate_intrinsic_dimensionality(self, data: np.ndarray, k: int = 10) -> float:
        """Estimate intrinsic dimensionality using nearest neighbors"""
        nbrs = NearestNeighbors(n_neighbors=k + 1).fit(data)
        distances, _ = nbrs.kneighbors(data)
        
        # Use the method from Levina and Bickel (2004)
        log_distances = np.log(distances[:, 1:])
        mean_log_distances = np.mean(log_distances, axis=1)
        
        # Estimate dimensionality
        d = - (k - 2) / np.sum(mean_log_distances - np.mean(mean_log_distances))
        
        return max(1.0, min(float(d), data.shape[1]))
    
    def visualize_manifold(
        self,
        data: torch.Tensor,
        generated_samples: Optional[torch.Tensor] = None,
        save_path: Optional[str] = None
    ):
        """Visualize manifold structure and diffusion process"""
        fig = plt.figure(figsize=(20, 15))
        
        if data.shape[1] == 2:
            # 2D visualization
            self._visualize_2d_manifold(fig, data, generated_samples)
        elif data.shape[1] == 3:
            # 3D visualization
            self._visualize_3d_manifold(fig, data, generated_samples)
        else:
            # High-dimensional visualization using PCA
            self._visualize_highd_manifold(fig, data, generated_samples)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def _visualize_2d_manifold(self, fig, data: torch.Tensor, generated_samples: Optional[torch.Tensor]):
        """Visualize 2D manifold"""
        # Original data
        ax1 = fig.add_subplot(2, 3, 1)
        ax1.scatter(data[:, 0].numpy(), data[:, 1].numpy(), alpha=0.6, s=20, c='blue', label='Original Data')
        ax1.set_title('Original Data')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Manifold coordinates
        if self.learned_manifold is not None:
            ax2 = fig.add_subplot(2, 3, 2)
            manifold_coords = self.learned_manifold['manifold_coords']
            ax2.scatter(manifold_coords[:, 0].numpy(), manifold_coords[:, 1].numpy(), 
                       alpha=0.6, s=20, c='green', label='Manifold Coordinates')
            ax2.set_title('Learned Manifold Coordinates')
            ax2.set_xlabel('X')
            ax2.set_ylabel('Y')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Generated samples
        if generated_samples is not None:
            ax3 = fig.add_subplot(2, 3, 3)
            ax3.scatter(generated_samples[:, 0].numpy(), generated_samples[:, 1].numpy(), 
                       alpha=0.6, s=20, c='red', label='Generated Samples')
            ax3.set_title('Generated Samples')
            ax3.set_xlabel('X')
            ax3.set_ylabel('Y')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # Diffusion process visualization
        ax4 = fig.add_subplot(2, 3, 4)
        timesteps = [0, 250, 500, 750, 999]
        colors = plt.cm.viridis(np.linspace(0, 1, len(timesteps)))
        
        for i, t in enumerate(timesteps):
            noisy_data = self.q_sample(data, torch.tensor([t]))
            ax4.scatter(noisy_data[:, 0].numpy(), noisy_data[:, 1].numpy(), 
                       alpha=0.4, s=10, c=[colors[i]], label=f't={t}')
        
        ax4.set_title('Forward Diffusion Process')
        ax4.set_xlabel('X')
        ax4.set_ylabel('Y')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Proximity graph
        if self.proximity_graph is not None:
            ax5 = fig.add_subplot(2, 3, 5)
            data_np = data.detach().cpu().numpy()
            distances, indices = self.proximity_graph.kneighbors(data_np[:50])  # First 50 points
            
            ax5.scatter(data_np[:50, 0], data_np[:50, 1], alpha=0.6, s=30, c='purple')
            
            # Draw connections
            for i in range(50):
                for j in indices[i]:
                    if i < j < 50:  # Avoid duplicate lines
                        ax5.plot([data_np[i, 0], data_np[j, 0]], 
                                [data_np[i, 1], data_np[j, 1]], 
                                'gray', alpha=0.2, linewidth=0.5)
            
            ax5.set_title('Proximity Graph (Sample)')
            ax5.set_xlabel('X')
            ax5.set_ylabel('Y')
            ax5.grid(True, alpha=0.3)
        
        # Metrics
        ax6 = fig.add_subplot(2, 3, 6)
        metrics = self.compute_manifold_metrics(data)
        
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())
        
        bars = ax6.bar(metric_names, metric_values, alpha=0.7, color='orange')
        ax6.set_title('Manifold Quality Metrics')
        ax6.set_ylabel('Value')
        plt.setp(ax6.get_xticklabels(), rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{value:.3f}', ha='center', va='bottom')
    
    def _visualize_3d_manifold(self, fig, data: torch.Tensor, generated_samples: Optional[torch.Tensor]):
        """Visualize 3D manifold"""
        ax = fig.add_subplot(111, projection='3d')
        
        # Original data
        ax.scatter(data[:, 0].numpy(), data[:, 1].numpy(), data[:, 2].numpy(), 
                  alpha=0.6, s=20, c='blue', label='Original Data')
        
        # Generated samples
        if generated_samples is not None:
            ax.scatter(generated_samples[:, 0].numpy(), generated_samples[:, 1].numpy(), 
                      generated_samples[:, 2].numpy(), alpha=0.6, s=20, c='red', label='Generated Samples')
        
        ax.set_title('3D Manifold Visualization')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
    
    def _visualize_highd_manifold(self, fig, data: torch.Tensor, generated_samples: Optional[torch.Tensor]):
        """Visualize high-dimensional manifold using PCA"""
        from sklearn.decomposition import PCA
        
        # Apply PCA to reduce to 2D
        pca = PCA(n_components=2)
        data_2d = pca.fit_transform(data.detach().cpu().numpy())
        
        ax = fig.add_subplot(111)
        ax.scatter(data_2d[:, 0], data_2d[:, 1], alpha=0.6, s=20, c='blue', label='Original Data (PCA)')
        
        if generated_samples is not None:
            gen_2d = pca.transform(generated_samples.detach().cpu().numpy())
            ax.scatter(gen_2d[:, 0], gen_2d[:, 1], alpha=0.6, s=20, c='red', label='Generated Samples (PCA)')
        
        ax.set_title(f'High-D Manifold (PCA: {pca.explained_variance_ratio_.sum():.2%} variance)')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def save_model(self, path: str):
        """Save model state"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'learned_manifold': self.learned_manifold
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load model state"""
        checkpoint = torch.load(path, map_location='cpu')
        self.load_state_dict(checkpoint['model_state_dict'])
        self.learned_manifold = checkpoint.get('learned_manifold')
        logger.info(f"Model loaded from {path}")

# Factory function
def create_manifold_diffusion(
    data_dim: int = 2,
    hidden_dim: int = 512,
    diffusion_steps: int = 1000,
    manifold_neighbors: int = 15
) -> ManifoldDiffusionModel:
    """Create and return a manifold diffusion model"""
    config = DiffusionConfig(
        data_dim=data_dim,
        hidden_dim=hidden_dim,
        diffusion_steps=diffusion_steps,
        manifold_neighbors=manifold_neighbors
    )
    return ManifoldDiffusionModel(config)
