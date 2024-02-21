import numpy as np
import torch
import torch.nn.functional as F
from core.utils.utils import disp_warp
from core.utils.utils import bilinear_sampler


class Combined_Geo_Encoding_Volume:
    def __init__(self, init_fmap1, init_fmap2, geo_volume, num_levels=2, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.geo_volume_pyramid = []
        self.init_corr_pyramid = []

        # all pairs correlation
        init_corr = Combined_Geo_Encoding_Volume.corr(init_fmap1, init_fmap2)

        b, h, w, _, w2 = init_corr.shape
        b, c, d, h, w = geo_volume.shape
        geo_volume = geo_volume.permute(0, 3, 4, 1, 2).reshape(b*h*w, c, 1, d)

        init_corr = init_corr.reshape(b*h*w, 1, 1, w2)
        self.geo_volume_pyramid.append(geo_volume)
        self.init_corr_pyramid.append(init_corr)
        for i in range(self.num_levels-1):
            geo_volume = F.avg_pool2d(geo_volume, [1,2], stride=[1,2])
            self.geo_volume_pyramid.append(geo_volume)

        for i in range(self.num_levels-1):
            init_corr = F.avg_pool2d(init_corr, [1,2], stride=[1,2])
            self.init_corr_pyramid.append(init_corr)

    def __call__(self, disp, coords):
        r = self.radius
        b, _, h, w = disp.shape
        out_pyramid = []
        for i in range(self.num_levels):
            geo_volume = self.geo_volume_pyramid[i]
            dx = torch.linspace(-r, r, 2*r+1)
            dx = dx.view(1, 1, 2*r+1, 1).to(disp.device)
            x0 = dx + disp.reshape(b*h*w, 1, 1, 1) / 2**i
            y0 = torch.zeros_like(x0)

            disp_lvl = torch.cat([x0,y0], dim=-1)
            geo_volume = bilinear_sampler(geo_volume, disp_lvl)
            geo_volume = geo_volume.view(b, h, w, -1)

            init_corr = self.init_corr_pyramid[i]
            init_x0 = coords.reshape(b*h*w, 1, 1, 1)/2**i - disp.reshape(b*h*w, 1, 1, 1) / 2**i + dx
            init_coords_lvl = torch.cat([init_x0,y0], dim=-1)
            init_corr = bilinear_sampler(init_corr, init_coords_lvl)
            init_corr = init_corr.view(b, h, w, -1)

            out_pyramid.append(geo_volume)
            out_pyramid.append(init_corr)
        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()

    @staticmethod
    def corr(fmap1, fmap2):
        B, D, H, W1 = fmap1.shape
        _, _, _, W2 = fmap2.shape
        fmap1 = fmap1.view(B, D, H, W1)
        fmap2 = fmap2.view(B, D, H, W2)
        corr = torch.einsum('aijk,aijh->ajkh', fmap1, fmap2)
        corr = corr.reshape(B, H, W1, 1, W2).contiguous()
        return corr


class Adaptive_Sampled_Error_Aware_Combined_Cost_Volume:
    def __init__(self, image1_dw, image2_dw, init_fmap1, init_fmap2, geo_volume, num_levels=2, radius=4, error_aware_corr_windows=1, error_aware_corr_planes=9, error_aware_corr_groups=4):
        self.num_levels = num_levels
        self.radius = radius
        self.geo_volume_pyramid = []
        self.init_corr_pyramid = []
        self.image1_dw = image1_dw
        self.image2_dw = image2_dw
        self.init_fmap1 = init_fmap1
        self.init_fmap2 = init_fmap2
        self.error_aware_corr_windows = error_aware_corr_windows
        self.error_aware_corr_planes = error_aware_corr_planes
        self.error_aware_corr_groups = error_aware_corr_groups

        # all pairs correlation
        init_corr = Adaptive_Sampled_Error_Aware_Combined_Cost_Volume.corr(init_fmap1, init_fmap2)

        b, h, w, _, w2 = init_corr.shape
        b, c, d, h, w = geo_volume.shape
        geo_volume = geo_volume.permute(0, 3, 4, 1, 2).reshape(b*h*w, c, 1, d)

        init_corr = init_corr.reshape(b*h*w, 1, 1, w2)
        self.geo_volume_pyramid.append(geo_volume)
        self.init_corr_pyramid.append(init_corr)
        for i in range(self.num_levels-1):
            geo_volume = F.avg_pool2d(geo_volume, [1,2], stride=[1,2])
            self.geo_volume_pyramid.append(geo_volume)

        for i in range(self.num_levels-1):
            init_corr = F.avg_pool2d(init_corr, [1,2], stride=[1,2])
            self.init_corr_pyramid.append(init_corr)

    def __call__(self, disp, coords, sigma):
        # Compute GEV and RAFT Correlation
        r = self.radius
        b, _, h, w = disp.shape
        sigma = sigma.permute(0, 2, 3, 1).contiguous().reshape(b * h * w, 1, 1, 1)
        out_pyramid = []
        for i in range(self.num_levels):
            geo_volume = self.geo_volume_pyramid[i]
            dx = torch.linspace(-r, r, 2*r+1)
            dx = dx.view(1, 1, 2*r+1, 1).to(disp.device) * sigma
            # x0 = dx * sigma + disp.reshape(b*h*w, 1, 1, 1) / 2 ** i
            x0 = dx + disp.reshape(b*h*w, 1, 1, 1) / 2**i
            y0 = torch.zeros_like(x0)

            disp_lvl = torch.cat([x0,y0], dim=-1)
            geo_volume = bilinear_sampler(geo_volume, disp_lvl)
            geo_volume = geo_volume.view(b, h, w, -1)

            init_corr = self.init_corr_pyramid[i]
            init_x0 = coords.reshape(b*h*w, 1, 1, 1)/2**i - disp.reshape(b*h*w, 1, 1, 1) / 2**i + dx
            init_coords_lvl = torch.cat([init_x0,y0], dim=-1)
            init_corr = bilinear_sampler(init_corr, init_coords_lvl)
            init_corr = init_corr.view(b, h, w, -1)

            out_pyramid.append(geo_volume)
            out_pyramid.append(init_corr)
        
        # Compute Error Aware Correlation
        error_aware_corr = []
        # Get number of planes
        plane_offset = self.error_aware_corr_planes // 2
        dxs = torch.linspace(-plane_offset, plane_offset, self.error_aware_corr_planes)
        for dx in dxs:
            ddx = (dx * sigma).reshape(b, h, w, 1).permute(0, 3, 1, 2).contiguous()
            x0 = ddx + disp
            warped_corr = self.compute_error_aware_corr(self.image1_dw, self.image2_dw, self.init_fmap1, self.init_fmap2, x0, self.error_aware_corr_windows, self.error_aware_corr_groups)
            warped_corr = warped_corr.permute(0, 2, 3, 1).contiguous()
            error_aware_corr.append(warped_corr)
        error_aware_corr = torch.cat(error_aware_corr, dim=-1)
        gev_raft_corr = torch.cat(out_pyramid, dim=-1)

        return gev_raft_corr.permute(0, 3, 1, 2).contiguous().float(), error_aware_corr.permute(0, 3, 1, 2).contiguous().float()
    
    def manual_pad(self, x, pady, padx):
        pad = (padx, padx, pady, pady)
        return F.pad(x.clone().detach(), pad, "replicate")

    def window_correlation(self, fmap1, fmap2, win_h, win_w, psize=(3, 3), dilate=(1, 1)):
        N, C, H, W = fmap1.shape

        di_y, di_x = dilate[0], dilate[1]
        pady, padx = psize[0] // 2 * di_y, psize[1] // 2 * di_x

        fmap2_pad = self.manual_pad(fmap2, pady, padx)

        corr_list = []
        for h in range(0, win_h, 1):
            for w in range(0, win_w, 1):
                fmap2_crop = fmap2_pad[:, :, h : h + H, w : w + W]
                assert fmap2_crop.shape == fmap1.shape
                corr = torch.mean(fmap1 * fmap2_crop, dim=1, keepdims=True)
                corr_list.append(corr)

        corr_final = torch.cat(corr_list, dim=1)

        return corr_final
    
    def compute_error_aware_corr(self, image1_dw, image2_dw, fmap1, fmap2, disp, windows, groups):
        # warp the right feature to left
        zeros = torch.zeros_like(disp)  # [B, 1, H, W]
        displace = torch.cat((-disp, zeros), dim=1)  # [B, 2, H, W]
        warped_fmap2 = disp_warp(displace, fmap2)
        warped_image2_dw = disp_warp(displace, image2_dw) # [B, 3, H, W]
        error_aware_mask = (
            torch.mean(
                torch.abs(image1_dw - warped_image2_dw), dim=1, keepdims=True
            ) < 0.05
        ).float()
        warped_fmap2 = warped_fmap2 * error_aware_mask

        # split the features into groups
        lefts = torch.split(fmap1, fmap1.shape[1] // groups, dim=1)
        rights = torch.split(warped_fmap2, warped_fmap2.shape[1] // groups, dim=1)

        warped_error_aware_corrs = []
        if windows > 1:
            win_h = int(np.floor(np.sqrt(windows)))
            win_w = int(np.ceil(np.sqrt(windows)))
            if win_h != win_w:
                win_h = 1
                win_w = windows
            psize_list = [(win_h, win_w) for _ in range(groups)]
            dilate_list = [(1, 1) for _ in range(groups)]

            for i in range(groups):
                corr = self.window_correlation(
                    lefts[i], rights[i], win_h, win_w, psize_list[i], dilate_list[i]
                )
                warped_error_aware_corrs.append(corr)
        else:
            for i in range(groups):
                corr = torch.mean(lefts[i] * rights[i], dim=1, keepdims=True)
                warped_error_aware_corrs.append(corr)
            
        final_corr = torch.cat(warped_error_aware_corrs, dim=1)

        return final_corr

    @staticmethod
    def corr(fmap1, fmap2):
        B, D, H, W1 = fmap1.shape
        _, _, _, W2 = fmap2.shape
        fmap1 = fmap1.view(B, D, H, W1)
        fmap2 = fmap2.view(B, D, H, W2)
        corr = torch.einsum('aijk,aijh->ajkh', fmap1, fmap2)
        corr = corr.reshape(B, H, W1, 1, W2).contiguous()
        return corr
