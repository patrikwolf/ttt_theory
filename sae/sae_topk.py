import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple, Dict


class TopKActivation(nn.Module):
    
    def __init__(self, k: int):
        super().__init__()
        self.k = k
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        topk_values, topk_indices = torch.topk(x, k=self.k, dim=-1)
 
        activated = torch.zeros_like(x)
        activated.scatter_(-1, topk_indices, topk_values)
        
        return activated, topk_indices


class TopKSAE(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        k: int,
        use_ghost_grads: bool = True,
        ghost_threshold: float = 0.0,
        ghost_weight: float = 0.01,
        normalize_decoder: bool = True,
        decoder_bias: bool = True,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.k = k
        self.use_ghost_grads = use_ghost_grads
        self.ghost_threshold = ghost_threshold
        self.ghost_weight = ghost_weight
        self.normalize_decoder = normalize_decoder

        self.W_enc = nn.Parameter(torch.empty(input_dim, hidden_dim))
        self.b_enc = nn.Parameter(torch.zeros(hidden_dim))
        
        self.W_dec = nn.Parameter(torch.empty(hidden_dim, input_dim))

        if decoder_bias:
            self.b_dec = nn.Parameter(torch.zeros(input_dim))
        else:
            self.register_parameter('b_dec', None)

        self.drop = nn.Dropout(p=0.5)
        
        self.topk = TopKActivation(k)
        self.register_buffer('feature_counts', torch.zeros(hidden_dim))
        self.register_buffer('total_samples', torch.tensor(0))
        
        self._init_weights()
        
    def _init_weights(self):
        nn.init.xavier_uniform_(self.W_enc)
        
        with torch.no_grad():
            self.W_dec.data = self.W_enc.data.T.contiguous()
            
        if self.normalize_decoder:
            with torch.no_grad():
                self.W_dec.data = F.normalize(self.W_dec.data, dim=1)
    
    def update_k(self, new_k: int):
        self.k = new_k
        self.topk.k = new_k
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = x

        pre_activation = features @ self.W_enc + self.b_enc
        pre_activation = self.drop(pre_activation)
        activated, topk_indices = self.topk(pre_activation)

        with torch.no_grad():
            active_mask = activated > 0
            batch_feature_counts = active_mask.float().sum(dim=0)
            self.feature_counts += batch_feature_counts
            self.total_samples += x.shape[0]
        
        return pre_activation, activated
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        if self.normalize_decoder:
            W_dec_normalized = F.normalize(self.W_dec, dim=1)
            reconstructed = z @ W_dec_normalized
        else:
            reconstructed = z @ self.W_dec
            
        if self.b_dec is not None:
            reconstructed = reconstructed + self.b_dec
            
        return reconstructed
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        pre_activation, activated = self.encode(x)
        reconstruction = self.decode(activated)
        
        active_mask = activated > 0
        
        output = {
            'reconstruction': reconstruction,
            'activated': activated,
            'pre_activation': pre_activation,
            'active_mask': active_mask,
            'ghost_loss': torch.tensor(0.0, device=x.device)
        }
        
        # Ghost gradients
        if self.training and self.use_ghost_grads and self.ghost_weight > 0:
            # Compute reconstruction error
            with torch.no_grad():
                error = x - reconstruction
            
            # Find dead (or almost) features
            dead_features = (self.feature_counts / max(self.total_samples.item(), 1)) < 1e-4
            
            if dead_features.any():
                # Use dead features for reconstruction
                dead_mask = dead_features.unsqueeze(0).expand_as(pre_activation)
                ghost_pre_act = pre_activation * dead_mask.float()
                ghost_reconstruction = self.decode(ghost_pre_act)
                
                # Ghost loss: dead features improve reconstruction
                ghost_loss = F.mse_loss(ghost_reconstruction, error) * self.ghost_weight
                output['ghost_loss'] = ghost_loss
        
        return output
    
    def get_feature_stats(self) -> Dict[str, float]:
        if self.total_samples == 0:
            return {
                'dead_features': self.hidden_dim,
                'dead_feature_pct': 100.0,
            }
        
        dead_features = (self.feature_counts == 0).sum().item()
        dead_feature_pct = (dead_features / self.hidden_dim) * 100
            
        return {
            'dead_features': dead_features,
            'dead_feature_pct': dead_feature_pct,
        }
    
    def reset_statistics(self):
        """Reset feature statistics tracking."""
        self.feature_counts.zero_()
        self.total_samples.zero_()


def create_topk_sae(
    input_dim: int = 512,
    expansion_factor: int = 8,
    k: int = 32,
    use_ghost_grads: bool = True,
    ghost_threshold: float = 0.0,
    ghost_weight: float = 0.01,
    **kwargs
) -> TopKSAE:
    """Create a TopK SAE"""
    hidden_dim = input_dim * expansion_factor
    
    return TopKSAE(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        k=k,
        use_ghost_grads=use_ghost_grads,
        ghost_threshold=ghost_threshold,
        ghost_weight=ghost_weight,
        **kwargs
    )