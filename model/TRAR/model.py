import torch.nn as nn
import torch
from model.TRAR.trar import DynRT_ED
from model.TRAR.cls_layer import cls_layer_both, cls_layer_img, cls_layer_txt

class DynRT(nn.Module):
    def __init__(self, opt):
        super(DynRT, self).__init__()


        self.backbone = DynRT_ED(opt)


        if opt["classifier"] == 'both':
            self.cls_layer = cls_layer_both(opt["hidden_size"],opt["output_size"])
        elif opt["classifier"] == 'text':
            self.cls_layer = cls_layer_txt(opt["hidden_size"],opt["output_size"])
        elif opt["classifier"] == 'img':
            self.cls_layer = cls_layer_img(opt["hidden_size"],opt["output_size"])


    def forward(self, img_feat, lang_feat, lang_feat_mask):
        img_feat_mask = torch.zeros([img_feat.shape[0],1,1,img_feat.shape[1]],dtype=torch.bool,device=img_feat.device)
        # (bs, 1, 1, grid_num)
        lang_feat, img_feat = self.backbone(
            lang_feat,
            img_feat,
            lang_feat_mask,
            img_feat_mask
        )

        lang_feat = torch.mean(lang_feat, dim=1)
        img_feat = torch.mean(img_feat, dim=1)

        proj_feat = self.cls_layer(lang_feat, img_feat)

        return proj_feat, lang_feat, img_feat
    