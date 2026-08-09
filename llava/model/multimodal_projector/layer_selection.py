import torch
import torch.nn as nn
import torch.nn.functional as F


def get_num_vision_layers(config, default=24):
    return getattr(config, 'mm_vision_num_layers', getattr(config, 'num_hidden_layers', default))


class LayerSelector(nn.Module):
    """
    Dynamically reweights and combines image features from different layers
    based on a given sentence embedding, matching the LayerRouter logic.
    """
    def __init__(self, config):
        super().__init__()
        self.num_layers = get_num_vision_layers(config)
        self.feat_dim = config.hidden_size
        self.hidden_size = config.mm_hidden_size
        
        # An MLP to project the sentence embedding to a set of layer weights.
        intermediate_dim = 512
        self.layer_weights_projection = nn.Sequential(
            nn.Linear(self.feat_dim, intermediate_dim),
            nn.ReLU(),
            nn.Linear(intermediate_dim, self.num_layers)
        )

        # LayerNorm to stabilize the final fused features
        self.norm = nn.LayerNorm(self.hidden_size)

    def forward(self, all_layers_features: torch.Tensor, text_features: torch.Tensor):
        # 1. Compute layer weights from the sentence embedding
        # The router's output should match the total number of layers (L)
        layer_logits = self.layer_weights_projection(text_features) # Shape: [B, L]
        layer_weights = F.softmax(layer_logits, dim=-1)

        # 2. Calculate the dynamic, routed features
        # Reshape weights for broadcasting: [B, L] -> [B, L, 1, 1]
        reshaped_weights = layer_weights.unsqueeze(-1).unsqueeze(-1)
        
        # Perform the weighted sum across the layer dimension
        dynamic_features = torch.sum(all_layers_features * reshaped_weights, dim=1) # Shape: [B, P, F]

        # 3. Add a stable baseline (the penultimate layer) and normalize
        baseline_features = all_layers_features[:, -2, :, :] # Shape: [B, P, F]
        
        final_features = baseline_features + dynamic_features
        final_features = self.norm(final_features)

        # Return the final features and the weights (for potential balancing loss)
        return final_features, layer_weights
