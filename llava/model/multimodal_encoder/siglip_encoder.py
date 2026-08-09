import torch
import torch.nn as nn

try:
    from transformers import SiglipVisionModel, SiglipImageProcessor, SiglipVisionConfig
except ImportError as exc:
    SiglipVisionModel = None
    SiglipImageProcessor = None
    SiglipVisionConfig = None
    _SIGLIP_IMPORT_ERROR = exc
else:
    _SIGLIP_IMPORT_ERROR = None


class SigLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        if SiglipVisionModel is None:
            raise ImportError(
                'SigLIP vision towers require transformers with SiglipVisionModel, '
                'SiglipImageProcessor, and SiglipVisionConfig support.'
            ) from _SIGLIP_IMPORT_ERROR

        self.is_loaded = False
        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
            self.cfg_only = SiglipVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.image_processor = SiglipImageProcessor.from_pretrained(self.vision_tower_name)
        siglip_device_map = None if device_map == 'auto' else device_map
        self.vision_tower = SiglipVisionModel.from_pretrained(self.vision_tower_name, device_map=siglip_device_map)
        self.vision_tower.requires_grad_(False)
        self.is_loaded = True

    def _has_cls_token(self, hidden_state):
        return hidden_state.shape[1] == self.num_patches + 1

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            if self._has_cls_token(image_features):
                image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    def attention_select(self, image_forward_outs):
        hidden_state = image_forward_outs.hidden_states[self.select_layer]
        batch_size = hidden_state.shape[0]
        patch_count = hidden_state.shape[1] - 1 if self._has_cls_token(hidden_state) else hidden_state.shape[1]
        attentions = getattr(image_forward_outs, 'attentions', None)

        if attentions is None or len(attentions) == 0:
            return torch.zeros(batch_size, 1, patch_count, device=hidden_state.device, dtype=hidden_state.dtype)

        attention_idx = self.select_layer
        if attention_idx < 0:
            attention_idx = len(attentions) + attention_idx
        if attention_idx < 0 or attention_idx >= len(attentions) or attentions[attention_idx] is None:
            return torch.zeros(batch_size, 1, patch_count, device=hidden_state.device, dtype=hidden_state.dtype)

        layer_attention = attentions[attention_idx]
        if self._has_cls_token(hidden_state):
            return layer_attention[:, :, 0, 1:]

        return layer_attention.mean(dim=2)

    def get_all_layers_features(self, image_forward_outs, image):
        all_layers_features = []
        for i in range(1, len(image_forward_outs.hidden_states)):
            image_features = image_forward_outs.hidden_states[i]
            if self.select_feature == 'patch':
                if self._has_cls_token(image_features):
                    image_features = image_features[:, 1:]
            elif self.select_feature == 'cls_patch':
                image_features = image_features
            else:
                raise ValueError(f'Unexpected select feature: {self.select_feature}')
            all_layers_features.append(image_features.to(image.dtype))
        return torch.stack(all_layers_features, dim=1)

    def get_all_cls_tokens(self, image_forward_outs, image):
        all_cls_tokens = []
        for i in range(1, len(image_forward_outs.hidden_states)):
            layer_hidden_state = image_forward_outs.hidden_states[i]
            if self._has_cls_token(layer_hidden_state):
                cls_token = layer_hidden_state[:, 0]
            else:
                cls_token = layer_hidden_state.mean(dim=1)
            all_cls_tokens.append(cls_token.to(image.dtype))
        return torch.stack(all_cls_tokens, dim=1)

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            cls_attentions = []
            all_layers_features = []
            all_layers_cls_tokens = []

            for image in images:
                image_forward_out = self.vision_tower(
                    image.to(device=self.device, dtype=self.dtype).unsqueeze(0),
                    output_hidden_states=True,
                    output_attentions=False,
                )
                image_features.append(self.feature_select(image_forward_out).to(image.dtype))
                cls_attentions.append(self.attention_select(image_forward_out).to(image.dtype))
                all_layers_features.append(self.get_all_layers_features(image_forward_out, image))
                all_layers_cls_tokens.append(self.get_all_cls_tokens(image_forward_out, image))
        else:
            image_forward_outs = self.vision_tower(
                images.to(device=self.device, dtype=self.dtype),
                output_hidden_states=True,
                output_attentions=False,
            )
            image_features = self.feature_select(image_forward_outs).to(images.dtype)
            cls_attentions = self.attention_select(image_forward_outs).to(images.dtype)
            all_layers_features = self.get_all_layers_features(image_forward_outs, images)
            all_layers_cls_tokens = self.get_all_cls_tokens(image_forward_outs, images)

        return image_features, cls_attentions, all_layers_features, all_layers_cls_tokens

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2

    @property
    def num_layers(self):
        return self.config.num_hidden_layers
