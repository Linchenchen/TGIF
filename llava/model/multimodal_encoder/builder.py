import os
from .clip_encoder import CLIPVisionTower, CLIPVisionTowerS2
from .siglip_encoder import SigLIPVisionTower


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    is_absolute_path_exists = os.path.exists(vision_tower)
    use_s2 = getattr(vision_tower_cfg, 's2', False)
    vision_tower_lower = vision_tower.lower()

    if 'siglip' in vision_tower_lower:
        if use_s2:
            raise ValueError('S2 multiscale mode is not implemented for SigLIP vision towers.')
        return SigLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    if is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("laion") or "ShareGPT4V" in vision_tower:
        if use_s2:
            return CLIPVisionTowerS2(vision_tower, args=vision_tower_cfg, **kwargs)
        else:
            return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f'Unknown vision tower: {vision_tower}')
