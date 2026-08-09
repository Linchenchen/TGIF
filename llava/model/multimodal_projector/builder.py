import torch.nn as nn
import re

from .layer_selection import LayerSelector


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)


def build_mlp_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')

    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')

class LayerSelectorWithMLPProjector(nn.Module):
    def __init__(self, config, delay_load=False, **kwargs):
        super().__init__()
        self.layer_selector = LayerSelector(config)
        
        mlp_config = type(config)(**vars(config))
        mlp_config.mm_projector_type = 'mlp2x_gelu'
        self.mlp = build_mlp_projector(mlp_config, delay_load=delay_load, **kwargs)
    
    def forward(self, image_features, all_layers_features, text_embeddings):
        selected_image_features, layer_weights = self.layer_selector(all_layers_features, text_embeddings)
        return self.mlp(selected_image_features), layer_weights

def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')

    if projector_type == 'layer_selector':
        return LayerSelectorWithMLPProjector(config, **kwargs)

    return build_mlp_projector(config, delay_load=delay_load)
