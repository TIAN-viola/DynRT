import torch.nn as nn
from model.TRAR.layer_norm import LayerNorm
import torch
import torch.nn.functional as F


class cls_layer_img(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(cls_layer_img, self).__init__()
        self.proj_norm = LayerNorm(input_dim)
        self.proj = nn.Linear(input_dim, output_dim)
        
    def forward(self, lang_feat, img_feat):
        proj_feat = self.proj_norm(img_feat)
        proj_feat = self.proj(proj_feat)

        return proj_feat

class cls_layer_txt(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(cls_layer_txt, self).__init__()
        self.proj_norm = LayerNorm(input_dim)
        self.proj = nn.Linear(input_dim, output_dim)
        
    def forward(self, lang_feat, img_feat):
        proj_feat = self.proj_norm(lang_feat)
        proj_feat = self.proj(proj_feat)

        return proj_feat

class cls_layer_both(nn.Module):
    def __init__(self,  input_dim, output_dim):
        super(cls_layer_both, self).__init__()
        self.proj_norm = LayerNorm(input_dim)
        self.proj = nn.Linear(input_dim, output_dim)
        
    def forward(self, lang_feat, img_feat):
        proj_feat = lang_feat + img_feat
        proj_feat = self.proj_norm(proj_feat)
        proj_feat = self.proj(proj_feat)

        return proj_feat