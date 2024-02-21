import torch
import torch.nn as nn
from core.submodule import *
from core.utils.utils import disp_warp


class RefineNet(nn.Module):
    def __init__(self, args):
        super(RefineNet, self).__init__()
        self.args = args

        self.conv_disp = nn.Sequential(
            nn.Conv2d(1, 32, 7, 1, 3),
            nn.ReLU(inplace=True),
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(64 + args.hidden_dims[0], 64, 3, 1, 1),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 3, dilation=3),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 7, dilation=7),
            nn.ReLU(inplace=True),
        )
        self.conv4 = nn.Conv2d(64, 1, 3, 1, 1)
    
    def forward(self, disp, error_map, refined_feats):
        uncertainty_map = 1 - error_map
        x = self.conv_disp(disp)
        error_map_feat = refined_feats[0]
        features = refined_feats[1]
        x1 = self.conv1(torch.cat((x, features, error_map_feat), dim=1))
        x = self.conv2(x1)
        x = self.conv3(x)
        x = self.conv4(x)
        disp = disp + x * uncertainty_map
        return disp


class ErrorNet(nn.Module):
    def __init__(self, args, in_channels):
        super(ErrorNet, self).__init__()
        self.args = args

        self.featureRefine = nn.Sequential(
            BasicConv_IN(in_channels, args.hidden_dims[0], kernel_size=3, padding=1, stride=1),
            BasicConv_IN(args.hidden_dims[0], args.hidden_dims[0], kernel_size=1, padding=0, stride=1),
        )

        self.small_hourglass = nn.Sequential(
            BasicConv_IN(args.hidden_dims[0], args.hidden_dims[0], kernel_size=3, padding=1, stride=2),
            BasicConv_IN(args.hidden_dims[0], args.hidden_dims[0], kernel_size=3, padding=1, stride=1),
            BasicConv_IN(args.hidden_dims[0], args.hidden_dims[0], deconv=True, kernel_size=4, padding=1, stride=2),
            nn.Conv2d(args.hidden_dims[0], args.hidden_dims[0], kernel_size=1, padding=0, stride=1),
        )
    
    def cos_similarity(self, feat1, feat2):
        normed_feat1 = feat1 / (feat1.norm(dim=1, keepdim=True) + 1e-5)
        normed_feat2 = feat2 / (feat2.norm(dim=1, keepdim=True) + 1e-5)
        return (normed_feat1 * normed_feat2).sum(dim=1, keepdim=True)
    
    def forward(self, disp, left_feat, right_feat):
        b, c, h, w = left_feat.shape

        # cos-sim with group corr
        left_refined_feats = self.small_hourglass(self.featureRefine(left_feat))
        right_refined_feats = self.small_hourglass(self.featureRefine(right_feat))

        # warp the right feature to left
        zeros = torch.zeros_like(disp)  # [B, 1, H, W]
        displace = torch.cat((-disp, zeros), dim=1)  # [B, 2, H, W]
        right_refined_warped_feats = disp_warp(displace, right_refined_feats)

        similarity = self.cos_similarity(left_refined_feats, right_refined_warped_feats)
        error_prob = 1 - similarity

        return error_prob, left_refined_feats
