import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from .UNet2 import UNet2 as UNet
from .DCNv2.dcn_v2 import DCNv2
from .module_util import make_layer, backwarp, RCAB
from .RAFT.raft import RAFT

from collections import OrderedDict
from easydict import EasyDict as edict
import warnings

warnings.filterwarnings("ignore")


class TSAFusion(nn.Module):
    """Temporal Spatial Attention (TSA) fusion module.

    Temporal: Calculate the correlation between center frame and
        neighboring frames;
    Spatial: It has 3 pyramid levels, the attention is similar to SFT.
        (SFT: Recovering realistic texture in image super-resolution by deep
            spatial feature transform.)

    Args:
        num_feat (int): Channel number of middle features. Default: 64.
        num_frame (int): Number of frames. Default: 5.
        center_frame_idx (int): The index of center frame. Default: 2.
    """
    def __init__(self, num_feat=64, num_frame=5, center_frame_idx=2):
        super(TSAFusion, self).__init__()
        self.center_frame_idx = center_frame_idx
        # temporal attention (before fusion conv)
        self.temporal_attn1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.temporal_attn2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.feat_fusion = nn.Conv2d(num_frame * num_feat, num_feat, 1, 1)

        # spatial attention (after fusion conv)
        self.max_pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.avg_pool = nn.AvgPool2d(3, stride=2, padding=1)
        self.spatial_attn1 = nn.Conv2d(num_frame * num_feat, num_feat, 1)
        self.spatial_attn2 = nn.Conv2d(num_feat * 2, num_feat, 1)
        self.spatial_attn3 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.spatial_attn4 = nn.Conv2d(num_feat, num_feat, 1)
        self.spatial_attn5 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.spatial_attn_l1 = nn.Conv2d(num_feat, num_feat, 1)
        self.spatial_attn_l2 = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)
        self.spatial_attn_l3 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.spatial_attn_add1 = nn.Conv2d(num_feat, num_feat, 1)
        self.spatial_attn_add2 = nn.Conv2d(num_feat, num_feat, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.upsample = nn.Upsample(scale_factor=2,
                                    mode='bilinear',
                                    align_corners=False)

    def forward(self, aligned_feat):
        """
        Args:
            aligned_feat (Tensor): Aligned features with shape (b, t, c, h, w).

        Returns:
            Tensor: Features after TSA with the shape (b, c, h, w).
        """
        b, t, c, h, w = aligned_feat.size()
        # temporal attention
        embedding_ref = self.temporal_attn1(
            aligned_feat[:, self.center_frame_idx, :, :, :].clone())
        embedding = self.temporal_attn2(aligned_feat.view(-1, c, h, w))
        embedding = embedding.view(b, t, -1, h, w)  # (b, t, c, h, w)

        corr_l = []  # correlation list
        for i in range(t):
            emb_neighbor = embedding[:, i, :, :, :]
            corr = torch.sum(emb_neighbor * embedding_ref, 1)  # (b, h, w)
            corr_l.append(corr.unsqueeze(1))  # (b, 1, h, w)
        corr_prob = torch.sigmoid(torch.cat(corr_l, dim=1))  # (b, t, h, w)
        corr_prob = corr_prob.unsqueeze(2).expand(b, t, c, h, w)
        corr_prob = corr_prob.contiguous().view(b, -1, h, w)  # (b, t*c, h, w)
        aligned_feat = aligned_feat.view(b, -1, h, w) * corr_prob

        # fusion
        feat = self.lrelu(self.feat_fusion(aligned_feat))

        # spatial attention
        attn = self.lrelu(self.spatial_attn1(aligned_feat))
        attn_max = self.max_pool(attn)
        attn_avg = self.avg_pool(attn)
        attn = self.lrelu(
            self.spatial_attn2(torch.cat([attn_max, attn_avg], dim=1)))
        # pyramid levels
        attn_level = self.lrelu(self.spatial_attn_l1(attn))
        attn_max = self.max_pool(attn_level)
        attn_avg = self.avg_pool(attn_level)
        attn_level = self.lrelu(
            self.spatial_attn_l2(torch.cat([attn_max, attn_avg], dim=1)))
        attn_level = self.lrelu(self.spatial_attn_l3(attn_level))
        attn_level = self.upsample(attn_level)

        attn = self.lrelu(self.spatial_attn3(attn)) + attn_level
        attn = self.lrelu(self.spatial_attn4(attn))
        attn = self.upsample(attn)
        attn = self.spatial_attn5(attn)
        attn_add = self.spatial_attn_add2(
            self.lrelu(self.spatial_attn_add1(attn)))
        attn = torch.sigmoid(attn)

        # after initialization, * 2 makes (attn * 2) to be close to 1.
        feat = feat * attn * 2 + attn_add
        return feat


class FGDA(nn.Module):
    """flow-guided deformable alignment (FGDA) module

    Args:
        num_feat (int): Channel number of middle features. Default: 64.
        deformable_groups (int): Deformable groups. Defaults: 8.
    """
    def __init__(self, num_feat=64, deformable_groups=16):
        super(FGDA, self).__init__()

        self.dcn = DCNv2(num_feat,
                         num_feat,
                         kernel_size=3,
                         stride=1,
                         padding=1,
                         deformable_groups=deformable_groups)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv_offset_mask = nn.Sequential(
            nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1), self.lrelu,
            nn.Conv2d(num_feat, num_feat, 3, 1, 1), self.lrelu,
            nn.Conv2d(num_feat, num_feat, 3, 1, 1), self.lrelu,
            nn.Conv2d(num_feat, deformable_groups * 3 * 3 * 3, 3, 1, 1))

    def forward(self, nbr_feat, ref_feat, flow):
        warped_feat = backwarp(nbr_feat, flow)

        out = self.conv_offset_mask(torch.cat([warped_feat, ref_feat], dim=1))
        o1, o2, mask = torch.chunk(out, 3, dim=1)

        residual_offset = torch.cat((o1, o2), dim=1)  # [N, g*3*3*2, H, W]
        N, _, H, W = flow.size()

        offset = torch.flip(flow, dims=[1]).view(
            N, 1, 2, H, W) + residual_offset.view(N, -1, 2, H, W)

        offset = offset.view(N, -1, H, W)
        mask = torch.sigmoid(mask)

        feat = self.lrelu(self.dcn(nbr_feat, offset, mask))
        return feat


class ICFuion(nn.Module):
    def __init__(self, num_feat=64):
        super(ICFuion, self).__init__()
        self.conv_first = nn.Conv2d(3, num_feat, 3, 1, 1)
        self.fusionI = nn.Conv2d(3 * num_feat, num_feat, 1, 1)
        self.fusionC = nn.Conv2d(3 * 64, 64, 1, 1)
        self.fusionIC = nn.Conv2d(num_feat + 64, num_feat, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, I1t, It, I2t, C1t, Ct, C2t):
        B, _, H, W = It.size()

        I_aligned = torch.stack([I1t, It, I2t], dim=1)  # (B, 3, 3, H, W)
        I_feat = self.lrelu(self.conv_first(I_aligned.view(-1, 3, H, W))).view(
            B, -1, H, W)
        I_feat = self.fusionI(I_feat)

        C_aligned = torch.cat([C1t, Ct, C2t], dim=1)  # (B, 3*C, H, W)
        C_feat = self.fusionC(C_aligned)

        feat = self.fusionIC(torch.cat([I_feat, C_feat], dim=1))
        return feat


class Refinement(nn.Module):
    def __init__(self, num_feat=64, num_block=5):
        super(Refinement, self).__init__()
        self.conv_first = nn.Conv2d(num_feat * 2 + 64, num_feat, 1)
        self.conv_last = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.body = make_layer(RCAB, num_block, num_feat=num_feat)

    def forward(self, feat_t, feat_2, Ft2, Ct):
        feat = torch.cat([feat_t, backwarp(feat_2, Ft2), Ct], dim=1)
        feat = self.lrelu(self.conv_first(feat))
        feat = self.body(feat)
        out = self.conv_last(feat)
        return out + feat_t


class FCA(nn.Module):
    def __init__(self,
                 scale=4,
                 num_feat=64,
                 num_reconstruct_block=10,
                 num_refine_block=5):
        super(FCA, self).__init__()
        self.scale = scale

        # extract the contextual information
        resnet18 = torchvision.models.resnet18(pretrained=True)
        self.feat_ext = list(resnet18.children())[0]
        self.feat_ext.stride = (1, 1)
        self.feat_ext.requires_grad = False

        # RAFT
        args = edict({
            'model': 'checkpoints/raft/raft-sintel.pth-no-zip',
            'small': False,
            'alternate_corr': False,
            'mixed_precision': False
        })
        self.raft = RAFT(args)
        load_net = torch.load(args.model)
        load_net_clean = OrderedDict()
        for k, v in load_net.items():
            if k.startswith('module.'):
                load_net_clean[k[7:]] = v
            else:
                load_net_clean[k] = v
        self.raft.load_state_dict(load_net_clean, strict=True)

        # Flow and Mask Prediction
        self.flowMaskPred = UNet(24, 5)

        # SR
        self.conv_first = nn.Conv2d(3, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.FGDAlign = FGDA(num_feat)
        self.fusion_t = ICFuion(num_feat)
        self.fusion_2 = TSAFusion(num_feat, num_frame=4, center_frame_idx=2)
        self.reconstruction = make_layer(RCAB,
                                         num_reconstruct_block,
                                         num_feat=num_feat)
        self.refinement = Refinement(num_feat, num_refine_block)
        self.upconv = nn.Conv2d(num_feat, 3 * scale * scale, 3, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(scale)

    def forward(self, x, F10=None, F12=None):
        B, _, _, H, W = x.size()
        I0, I1, I2, I3 = torch.split(x, 1, dim=1)
        I0, I1, I2, I3 = I0.squeeze(1), I1.squeeze(1), I2.squeeze(
            1), I3.squeeze(1)

        # Optical FLow
        with torch.no_grad():
            if F10 is None or F12 is None:
                _, Fout = self.raft(torch.cat([I1, I1, I1, I2, I2, I2], dim=0),
                                    torch.cat([I0, I2, I3, I3, I1, I0], dim=0),
                                    iters=5,
                                    test_mode=True)
                F10, F12, F13, F23, F21, F20 = torch.split(Fout, B, dim=0)
            else:
                _, Fout = self.raft(torch.cat([I1, I2, I2, I2], dim=0),
                                    torch.cat([I3, I3, I1, I0], dim=0),
                                    iters=5,
                                    test_mode=True)
                F13, F23, F21, F20 = torch.split(Fout, B, dim=0)

        # Flow and Mask Prediction
        output, _ = self.flowMaskPred(
            torch.cat([F10, F12, F13, F23, F21, F20, I0, I1, I2, I3], dim=1))
        Ft1, Ft2, M = output[:, :2], output[:, 2:4], output[:, 4:5]
        M = torch.sigmoid(M)

        # Context Extraction
        with torch.no_grad():
            C1 = self.feat_ext(I1)
            C2 = self.feat_ext(I2)

        # Blending
        I1t = backwarp(I1, Ft1)
        I2t = backwarp(I2, Ft2)
        C1t = backwarp(C1, Ft1)
        C2t = backwarp(C2, Ft2)
        It = M * I1t + (1 - M) * I2t
        Ct = M * C1t + (1 - M) * C2t

        # ICF
        feat_t = self.fusion_t(I1t, It, I2t, C1t, Ct, C2t)

        # FGDA
        feat_2 = torch.stack([I0, I1, I2, I3], dim=1)  # (B, 4, C, H, W)
        feat_2 = self.lrelu(self.conv_first(feat_2.view(-1, 3, H, W))).view(
            B, 4, -1, H, W)
        aligned = []
        ref_feat = feat_2[:, 2, :, :, :].clone()
        nbr_feat = feat_2[:, 0, :, :, :].clone()
        aligned.append(self.FGDAlign(nbr_feat, ref_feat, F20))
        nbr_feat = feat_2[:, 1, :, :, :].clone()
        aligned.append(self.FGDAlign(nbr_feat, ref_feat, F21))
        aligned.append(ref_feat)
        nbr_feat = feat_2[:, 3, :, :, :].clone()
        aligned.append(self.FGDAlign(nbr_feat, ref_feat, F23))
        aligned = torch.stack(aligned, dim=1)  # (B, 4, C, H, W)

        # TSA
        feat_2 = self.fusion_2(aligned)

        # Reconstruction
        feat = torch.cat([feat_t, feat_2], dim=0)
        feat = self.reconstruction(feat)

        # Refinement
        feat_t, feat_2 = torch.split(feat, B, dim=0)
        feat_t = self.refinement(feat_t, feat_2, Ft2, Ct)
        feat = torch.cat([feat_t, feat_2], dim=0)

        # upsample
        out = self.pixel_shuffle(self.upconv(feat))
        res_t, res_2 = torch.split(out, B, dim=0)
        It_SR = F.interpolate(It, scale_factor=self.scale,
                              mode='bilinear') + res_t
        I2_SR = F.interpolate(I2, scale_factor=self.scale,
                              mode='bilinear') + res_2

        outs = torch.stack([It_SR, I2_SR], dim=1)  # [B, 2, C, H, W]

        if self.training:
            return outs
        else:
            return outs, F21, F23
