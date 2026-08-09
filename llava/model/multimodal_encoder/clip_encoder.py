import torch
import torch.nn as nn

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig


class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features
    
    def attention_select(self, image_forward_outs):
        # tuple of attention for each layer shape (batch_size, num_heads, sequence_length, sequence_length)
        cls_attention = image_forward_outs.attentions[self.select_layer][:, :, 0, 1:]
        return cls_attention

    def get_all_layers_features(self, image_forward_outs, image):
        all_layers_features = None
        for i in range(1, len(image_forward_outs.hidden_states)):
            image_features = image_forward_outs.hidden_states[i]
            if self.select_feature == 'patch':
                image_features = image_features[:, 1:]
            elif self.select_feature == 'cls_patch':
                image_features = image_features
            else:
                raise ValueError(f'Unexpected select feature: {self.select_feature}')
            image_features = image_features[None, :].to(image.dtype)
            if all_layers_features is None:
                all_layers_features = image_features
            else:
                all_layers_features = torch.cat((all_layers_features, image_features), dim=0)
        all_layers_features = all_layers_features.permute(1, 0, 2, 3) # (batch_size, num_layers, num_patches, hidden_size)
        return all_layers_features

    def get_all_cls_tokens(self, image_forward_outs, image):
        """
        Extracts the [CLS] token's representation from every layer of the ViT.
        """
        all_cls_tokens = []
        # Start from 1 to skip the initial embedding layer
        for i in range(1, len(image_forward_outs.hidden_states)):
            # Get the hidden state for the current layer
            layer_hidden_state = image_forward_outs.hidden_states[i]
            # The CLS token is the first token in the sequence (index 0)
            cls_token = layer_hidden_state[:, 0]
            all_cls_tokens.append(cls_token.to(image.dtype))
        
        # Stack the list of CLS tokens into a single tensor
        # Final shape: (batch_size, num_layers, hidden_size)
        stacked_cls_tokens = torch.stack(all_cls_tokens, dim=1)
        return stacked_cls_tokens

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            cls_attentions = []
            all_layers_features = []
            all_layers_cls_tokens = []

            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True, output_attentions=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                cls_attention = self.attention_select(image_forward_out).to(image.dtype)
                
                all_layers_feature = self.get_all_layers_features(image_forward_out, image)
                all_cls_token = self.get_all_cls_tokens(image_forward_out, image)

                image_features.append(image_feature)
                cls_attentions.append(cls_attention)
                all_layers_features.append(all_layers_feature)
                all_layers_cls_tokens.append(all_cls_token)
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True, output_attentions=True)
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
        else:
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



class CLIPVisionTowerS2(CLIPVisionTower):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__(vision_tower, args, delay_load)

        self.s2_scales = getattr(args, 's2_scales', '336,672,1008')
        self.s2_scales = list(map(int, self.s2_scales.split(',')))
        self.s2_scales.sort()
        self.s2_split_size = self.s2_scales[0]
        self.s2_image_size = self.s2_scales[-1]

        try:
            from s2wrapper import forward as multiscale_forward
        except ImportError:
            raise ImportError('Package s2wrapper not found! Please install by running: \npip install git+https://github.com/bfshi/scaling_on_scales.git')
        self.multiscale_forward = multiscale_forward

        # change resize/crop size in preprocessing to the largest image size in s2_scale
        if not delay_load or getattr(args, 'unfreeze_mm_vision_tower', False):
            self.image_processor.size['shortest_edge'] = self.s2_image_size
            self.image_processor.crop_size['height'] = self.image_processor.crop_size['width'] = self.s2_image_size

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower.requires_grad_(False)

        self.image_processor.size['shortest_edge'] = self.s2_image_size
        self.image_processor.crop_size['height'] = self.image_processor.crop_size['width'] = self.s2_image_size

        self.is_loaded = True

    @torch.no_grad()
    def forward_feature(self, images):
        image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
        image_features = self.feature_select(image_forward_outs).to(images.dtype)
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_feature = self.multiscale_forward(self.forward_feature, image.unsqueeze(0), img_sizes=self.s2_scales, max_split_size=self.s2_split_size)
                image_features.append(image_feature)
        else:
            image_features = self.multiscale_forward(self.forward_feature, images, img_sizes=self.s2_scales, max_split_size=self.s2_split_size)

        return image_features

    @property
    def hidden_size(self):
        return self.config.hidden_size * len(self.s2_scales)
