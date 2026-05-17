import torch
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F
from einops import rearrange
import math


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


#########################################
# Helper Modules
#########################################


class DepthWiseConv(nn.Module):
    def __init__(self, dim, kernel_size=3):
        super(DepthWiseConv, self).__init__()
        self.conv = nn.Conv2d(
            dim,
            dim,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=dim,
            bias=True,
        )

    def forward(self, x):
        # Expects (B, C, H, W)
        return self.conv(x)


class LayerNorm(nn.Module):
    # Channel-first LayerNorm to match dimensions in conv blocks usually
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        # x: (B, H, W, C) -> (B, L, C)
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight * x + self.bias


#########################################
# 1. ReLinA: Rank-Enhanced Linear Attention
# Figure 3 (d) & Section 3.4
#########################################
class ReLinA(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        dwc_kernel_size=5,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.dwc = DepthWiseConv(dim, kernel_size=dwc_kernel_size)
        self.eps = 1e-6  # Avoid division by zero.

    def forward(self, x):
        B, H, W, C = x.shape
        N = H * W

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 1. Feature Map Formulation (1 + ELU)
        q = 1 + F.elu(q)
        k = 1 + F.elu(k)

        # 2. Linear attention with normalization.
        # Numerator: Q * (K^T * V)
        kv = k.transpose(-2, -1) @ v  # (B, Heads, HeadDim, HeadDim)
        x_attn = q @ kv  # (B, Heads, N, HeadDim)

        # Denominator: Q * (K^T * 1) = Q * K_sum
        k_sum = k.sum(dim=-2)  # (B, Heads, HeadDim) - sum over N
        z = 1 / (q @ k_sum.unsqueeze(-1) + self.eps)  # (B, Heads, N, 1)

        # Normalize the attention output.
        x_attn = x_attn * z

        x_attn = x_attn.transpose(1, 2).reshape(B, H, W, C)

        # 3. Rank Enhancement
        v_spatial = rearrange(v, "b heads (h w) d -> b (heads d) h w", h=H, w=W)
        v_enhanced = self.dwc(v_spatial)
        v_enhanced = v_enhanced.permute(0, 2, 3, 1)

        # Fusion
        x = x_attn + v_enhanced

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CAB(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj1 = nn.Conv2d(dim, dim, 1)
        self.dwconv = DepthWiseConv(dim, 3)
        self.act = nn.GELU()

        # Channel Attention 
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 8, 1, padding=0, bias=True),  # Reduction ratio often 8 or 16
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 8, dim, 1, padding=0, bias=True),
            nn.Sigmoid(),
        )

        # 1x1 Conv
        self.proj2 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        # Expects (B, H, W, C) input
        shortcut = x
        x = x.permute(0, 3, 1, 2)  # (B, C, H, W)

        x = self.proj1(x)
        x = self.dwconv(x)
        x = self.act(x)

        # Channel Attention
        chn_attn = self.ca(x)
        x = x * chn_attn

        x = self.proj2(x)

        x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        return x


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class DESM(nn.Module):
    """
    Wrapper-adapted DESM:
    Input/Output: [B, H, W, C]  (NHWC)
    Internally converts to [B, HW, C] to reuse original DESM logic.
    """

    def __init__(self, dim=32, hidden_dim=128, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        self.linear1 = nn.Sequential(nn.Linear(dim, hidden_dim), act_layer())

        self.dwconv = nn.Sequential(
            nn.Conv2d(
                hidden_dim // 2, hidden_dim // 2, kernel_size=3, padding=1, groups=hidden_dim // 2
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                hidden_dim // 2,
                hidden_dim // 2,
                kernel_size=3,
                padding=1,
                dilation=1,
                groups=hidden_dim // 2,
            ),
        )

        self.conv = nn.Conv2d(
            in_channels=hidden_dim // 4,
            out_channels=hidden_dim // 2,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True,
            dilation=1,
        )

        self.linear2 = nn.Sequential(nn.Linear(hidden_dim, dim))

        self.dim = dim
        self.hidden_dim = hidden_dim

        self.sc = SimpleGate()  

        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(
                in_channels=hidden_dim // 4,
                out_channels=hidden_dim // 4,
                kernel_size=1,
                padding=0,
                stride=1,
                groups=1,
                bias=True,
                dilation=1,
            ),
        )

        # optional dropout (token-wise). 
        self.drop = nn.Dropout(drop) if drop and drop > 0 else nn.Identity()

    def forward(self, x):
        """
        x: [B,H,W,C]
        return: [B,H,W,C]
        """
        B, H, W, C = x.shape
        assert C == self.dim, f"Channel mismatch: got C={C}, expected dim={self.dim}"

        # NHWC -> [B, HW, C]
        x = x.view(B, H * W, C)

        # original DESM token pipeline
        x = self.linear1(x)
        x = self.drop(x)

        # restore spatial: [B, HW, hidden] -> [B, hidden, H, W]
        x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W)

        x1, x2 = x.chunk(2, dim=1)  # each: [B, hidden/2, H, W]

        x2 = self.dwconv(x2)

        x1 = self.sc(x1)  # expect it reduces channels by half: [B, hidden/4, H, W]
        x1 = self.conv(self.sca(x1) * x1)

        x = torch.cat((x1, x2), dim=1)  # [B, hidden, H, W]

        # flatten back: [B, hidden, H, W] -> [B, HW, hidden]
        x = rearrange(x, "b c h w -> b (h w) c", h=H, w=W)

        x = self.linear2(x)
        x = self.drop(x)

        # [B, HW, C] -> NHWC
        x = x.view(B, H, W, C)
        return x


class Mlp(nn.Module):
    def __init__(
        self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x



class PSCBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)

        self.rela = ReLinA(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.cab = CAB(dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = DESM(dim, mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        # x: (B, H, W, C)

        shortcut = x
        x_norm = self.norm1(x)

        x_relina = self.rela(x_norm)
        x_cab = self.cab(x_norm)

        x = shortcut + self.drop_path(x_relina + x_cab)
        

        # FFN
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        # print("input: ",x.shape)
        return x


class BasicRaLiFormerLayer(nn.Module):
    def __init__(
        self,
        dim,
        output_dim,
        depth,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=LayerNorm,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth

        self.blocks = nn.ModuleList(
            [
                PSCBlock(
                    dim=dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

    def forward(self, x):
        # x: (B, L, C) from Uformer context, but we need (B, H, W, C) for convs
        B, L, C = x.shape
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        x = x.view(B, H, W, C)

        for blk in self.blocks:
            x = blk(x)

        x = x.view(B, L, C)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Downsample, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x):
        B, L, C = x.shape
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        out = self.conv(x).flatten(2).transpose(1, 2).contiguous()
        return out


class Upsample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Upsample, self).__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel, kernel_size=2, stride=2),
        )

    def forward(self, x):
        B, L, C = x.shape
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        out = self.deconv(x).flatten(2).transpose(1, 2).contiguous()
        return out


class InputProj(nn.Module):
    def __init__(
        self, in_channel=3, out_channel=64, kernel_size=3, stride=1, act_layer=nn.LeakyReLU
    ):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(
                in_channel, out_channel, kernel_size=3, stride=stride, padding=kernel_size // 2
            ),
            act_layer(inplace=True),
        )

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2).contiguous()


class OutputProj(nn.Module):
    def __init__(self, in_channel=64, out_channel=3, kernel_size=3, stride=1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(
                in_channel, out_channel, kernel_size=3, stride=stride, padding=kernel_size // 2
            ),
        )

    def forward(self, x):
        B, L, C = x.shape
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        x = x.transpose(1, 2).view(B, C, H, W)
        return self.proj(x)


#########################################
# Main Architecture: RaLiFormer
#########################################
class RaLiFormer(nn.Module):
    def __init__(
        self,
        img_size=128,
        in_chans=3,
        out_chans=6,
        embed_dim=32,
        depths=[2, 2, 2, 2, 2, 2, 2, 2, 2],
        num_heads=[1, 2, 4, 8, 16, 16, 8, 4, 2],
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.25,
        norm_layer=LayerNorm,
        input_downscale=True,
        **kwargs,
    ):
        super().__init__()

        self.num_enc_layers = len(depths) // 2
        self.num_dec_layers = len(depths) // 2
        self.embed_dim = embed_dim
        self.mlp_ratio = mlp_ratio

        # Stochastic depth decay rule
        enc_dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths[: self.num_enc_layers]))
        ]
        conv_dpr = [drop_path_rate] * depths[4]
        dec_dpr = enc_dpr[::-1]

        stride = 2 if input_downscale else 1
        self.input_proj = InputProj(
            in_channel=in_chans,
            out_channel=embed_dim,
            kernel_size=3,
            stride=stride,
            act_layer=nn.LeakyReLU,
        )
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.input_downscale = input_downscale
        # 2. Encoder Stages
        # Stage 0 has the largest memory footprint.
        self.encoderlayer_0 = BasicRaLiFormerLayer(
            dim=embed_dim,
            output_dim=embed_dim,
            depth=depths[0],
            num_heads=num_heads[0],
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=enc_dpr[sum(depths[:0]) : sum(depths[:1])],
            norm_layer=norm_layer,
        )
        self.downsample_0 = Downsample(embed_dim, embed_dim * 2)

        self.encoderlayer_1 = BasicRaLiFormerLayer(
            dim=embed_dim * 2,
            output_dim=embed_dim * 2,
            depth=depths[1],
            num_heads=num_heads[1],
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=enc_dpr[sum(depths[:1]) : sum(depths[:2])],
            norm_layer=norm_layer,
        )
        self.downsample_1 = Downsample(embed_dim * 2, embed_dim * 4)

        self.encoderlayer_2 = BasicRaLiFormerLayer(
            dim=embed_dim * 4,
            output_dim=embed_dim * 4,
            depth=depths[2],
            num_heads=num_heads[2],
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=enc_dpr[sum(depths[:2]) : sum(depths[:3])],
            norm_layer=norm_layer,
        )
        self.downsample_2 = Downsample(embed_dim * 4, embed_dim * 8)

        self.encoderlayer_3 = BasicRaLiFormerLayer(
            dim=embed_dim * 8,
            output_dim=embed_dim * 8,
            depth=depths[3],
            num_heads=num_heads[3],
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=enc_dpr[sum(depths[:3]) : sum(depths[:4])],
            norm_layer=norm_layer,
        )
        self.downsample_3 = Downsample(embed_dim * 8, embed_dim * 16)

        # 3. Bottleneck at low resolution.
        self.conv = BasicRaLiFormerLayer(
            dim=embed_dim * 16,
            output_dim=embed_dim * 16,
            depth=depths[4],
            num_heads=num_heads[4],
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=conv_dpr,
            norm_layer=norm_layer,
        )

        # 4. Decoder Stages
        self.upsample_0 = Upsample(embed_dim * 16, embed_dim * 8)
        self.decoderlayer_0 = BasicRaLiFormerLayer(
            dim=embed_dim * 16,
            output_dim=embed_dim * 16,
            depth=depths[5],
            num_heads=num_heads[5],
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=dec_dpr[: depths[5]],
            norm_layer=norm_layer,
        )

        self.upsample_1 = Upsample(embed_dim * 16, embed_dim * 4)
        self.decoderlayer_1 = BasicRaLiFormerLayer(
            dim=embed_dim * 8,
            output_dim=embed_dim * 8,
            depth=depths[6],
            num_heads=num_heads[6],
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=dec_dpr[sum(depths[5:6]) : sum(depths[5:7])],
            norm_layer=norm_layer,
        )

        self.upsample_2 = Upsample(embed_dim * 8, embed_dim * 2)
        self.decoderlayer_2 = BasicRaLiFormerLayer(
            dim=embed_dim * 4,
            output_dim=embed_dim * 4,
            depth=depths[7],
            num_heads=num_heads[7],
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=dec_dpr[sum(depths[5:7]) : sum(depths[5:8])],
            norm_layer=norm_layer,
        )

        self.upsample_3 = Upsample(embed_dim * 4, embed_dim)
        self.decoderlayer_3 = BasicRaLiFormerLayer(
            dim=embed_dim * 2,
            output_dim=embed_dim * 2,
            depth=depths[8],
            num_heads=num_heads[8],
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=dec_dpr[sum(depths[5:8]) : sum(depths[5:9])],
            norm_layer=norm_layer,
        )

        # 5. Output Projection
        # Upsample back when the input was downscaled at the start.
        if self.input_downscale:
            self.final_up = nn.Sequential(
                nn.ConvTranspose2d(
                    embed_dim * 2, embed_dim, kernel_size=2, stride=2
                ),  # 2x upsample
                nn.LeakyReLU(inplace=True),
            )
            self.output_proj = OutputProj(
                in_channel=embed_dim, out_channel=out_chans, kernel_size=3, stride=1
            )
        else:
            self.final_up = nn.Identity()
            self.output_proj = OutputProj(
                in_channel=embed_dim * 2, out_channel=out_chans, kernel_size=3, stride=1
            )

        self.activation = nn.Sequential(nn.Sigmoid())
        self.apply(self._init_weights)
        self._init_last_layer(self.output_proj.proj[0])

    def _init_last_layer(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.constant_(m.weight, 0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        # Input projection, optionally with downsampling.
        y = self.input_proj(x)
        y = self.pos_drop(y)

        # Encoder
        conv0 = self.encoderlayer_0(y)
        pool0 = self.downsample_0(conv0)

        conv1 = self.encoderlayer_1(pool0)
        pool1 = self.downsample_1(conv1)

        conv2 = self.encoderlayer_2(pool1)
        pool2 = self.downsample_2(conv2)

        conv3 = self.encoderlayer_3(pool2)
        pool3 = self.downsample_3(conv3)

        # Bottleneck
        conv4 = self.conv(pool3)

        # Decoder (with Skip Connections via Concatenation)
        up0 = self.upsample_0(conv4)
        deconv0 = torch.cat([up0, conv3], -1)
        deconv0 = self.decoderlayer_0(deconv0)

        up1 = self.upsample_1(deconv0)
        deconv1 = torch.cat([up1, conv2], -1)
        deconv1 = self.decoderlayer_1(deconv1)

        up2 = self.upsample_2(deconv1)
        deconv2 = torch.cat([up2, conv1], -1)
        deconv2 = self.decoderlayer_2(deconv2)

        up3 = self.upsample_3(deconv2)
        deconv3 = torch.cat([up3, conv0], -1)
        deconv3 = self.decoderlayer_3(deconv3)

        # Final Upsampling (if input was downscaled)
        if self.input_downscale:
            # deconv3 shape: (B, L, 2*C) -> reshape to (B, 2*C, H, W) done implicitly by final_up if needed
            # Convert deconv3 from (B, L, C) back to image layout for upsampling.
            B, L, C = deconv3.shape
            H = int(math.sqrt(L))
            W = int(math.sqrt(L))
            deconv3_spatial = deconv3.transpose(1, 2).view(B, C, H, W)
            out_feat = self.final_up(deconv3_spatial)  # (B, C_out, H*2, W*2)
            # Flatten again so OutputProj can keep its existing interface.
            out_feat_flat = out_feat.flatten(2).transpose(1, 2)  # (B, L*4, C_out)
            y = self.output_proj(out_feat_flat)
        else:
            y = self.output_proj(deconv3)

        output = y[:, :3, :, :] + x  # [2, 3, 512, 512]
        output = self.activation(output)
        flare_predict = y[:, 3:, :, :]  # [2, 3, 512, 512]
        flare_predict = self.activation(flare_predict)

        return output, flare_predict


if __name__ == "__main__":
    # Test instantiation
    model = RaLiFormer(img_size=512, embed_dim=32, depths=[2, 2, 2, 2, 2, 2, 2, 2, 2])
    # print(model)
    x = torch.randn(1, 3, 512, 512)
    y1, y2 = model(x)
    print(y1.shape)
    print("model params: %d" % count_parameters(model))
# python -m model.RaLiFormer
