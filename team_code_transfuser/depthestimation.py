import torch
import torch.nn as nn
import torch.nn.functional as F


from DeformableAtt import DeformableAtt
from transformers import AutoModel, AutoImageProcessor


class DeformableDino(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.deformable_att = DeformableAtt(self.config)
        self.dino = AutoModel.from_pretrained(config.dino_model_name_or_path)
        self.image_processor = AutoImageProcessor.from_pretrained(config.dino_model_name_or_path)

        # freezing DINO v2 parameter
        for param in self.dino.parameters():
            param.require_grads = False

        
        # layer for predicting depth
        dino_output_dim = self.dino.conig.hidden_size

        self.depth_estimator = nn.Sequential(
            nn.Linear(dino_output_dim, 256),
            nn.ReLU(),
            nn.Linear(256,1)
        )

    def forward(self,x):
        
        features, pos, reference = self.deformable_att(x)
        B, C, H, W = features.shape

        features = features.permute(0,2,3,1).contiguous()

        # dino v2 output
        dino_outputs = self.dino(pixel_values=features, output_hidden_states=True)
        dino_features = dino_outputs.last_hiddnes_state

        depth = self.depth_estimator(dino_features[:,0])

        depth  = depth.view(B,1,H,W)

        return depth , pos, features

