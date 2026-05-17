import torch
import random
from math import log10
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0.0)


# recommend
def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
        # m.weight.data.normal_(0, 0.02)
        # m.bias.data.zero_()
        # nn.init.xavier_normal_(m.weight.data)
        nn.init.kaiming_normal(m.weight.data, mode="fan_out")
        # nn.init.xavier_normal_(m.bias.data)
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.02)
        m.bias.data.zero_()


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all statistics"""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """Update statistics"""
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def to_psnr(J, gt):
    mse = F.mse_loss(J, gt, reduction="none")
    mse_split = torch.split(mse, 1, dim=0)
    mse_list = [torch.mean(torch.squeeze(mse_split[ind])).item() for ind in range(len(mse_split))]
    intensity_max = 1.0
    psnr_list = [10.0 * log10(intensity_max / mse) for mse in mse_list]
    return psnr_list


def create_emamodel(net, ema=True):
    if ema:
        for param in net.parameters():
            param.detach_()
    return net


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_psnr_ssim(recoverd, clean):
    assert recoverd.shape == clean.shape
    recoverd = np.clip(recoverd.detach().cpu().numpy(), 0, 1)
    clean = np.clip(clean.detach().cpu().numpy(), 0, 1)
    recoverd = recoverd.transpose(0, 2, 3, 1)
    clean = clean.transpose(0, 2, 3, 1)
    psnr = 0
    ssim = 0

    for i in range(recoverd.shape[0]):
        psnr += peak_signal_noise_ratio(clean[i], recoverd[i], data_range=1)
        ssim += structural_similarity(
            clean[i], recoverd[i], data_range=1, win_size=3, channel_axis=-1
        )

    return psnr / recoverd.shape[0], ssim / recoverd.shape[0], recoverd.shape[0]


class FFTLoss(nn.Module):
    """Frequency-domain L1 loss over real-valued FFT coefficients."""

    def __init__(self):
        super(FFTLoss, self).__init__()
        self.criterion = torch.nn.L1Loss()

    def forward(self, pred, target):
        pred_fft = torch.fft.rfft2(pred, norm="ortho")
        target_fft = torch.fft.rfft2(target, norm="ortho")
        return self.criterion(torch.view_as_real(pred_fft), torch.view_as_real(target_fft))


class PatchInfoNCE(nn.Module):
    """
    Anchor: student output
    Positive: pseudo label
    Negative: input (unpaired_data_s)
    Patch-level InfoNCE, works even when batch size is small.
    """

    def __init__(self, feat_net: nn.Module, num_patches=128, temperature=0.1):
        super().__init__()
        self.feat_net = feat_net
        self.num_patches = num_patches
        self.temperature = temperature

        # freeze feature net params, but allow grad to flow to input tensors
        for p in self.feat_net.parameters():
            p.requires_grad = False
        self.feat_net.eval()

    def _feat(self, x):
        # feature map: [B,C,h,w]
        return self.feat_net(x)

    def forward(self, out_img, pseudo_img, in_img):
        """
        out_img:   [B,3,H,W]  student output (needs grad)
        pseudo_img:[B,3,H,W]  pseudo target (usually detach)
        in_img:    [B,3,H,W]  unpaired_data_s (usually detach)
        """
        fq = self._feat(out_img)  # [B,C,h,w]
        fk_pos = self._feat(pseudo_img)  # [B,C,h,w]
        fk_neg = self._feat(in_img)  # [B,C,h,w]

        B, C, h, w = fq.shape
        HW = h * w
        K = min(self.num_patches, HW)

        # flatten to [B,HW,C]
        q = fq.permute(0, 2, 3, 1).reshape(B, HW, C)
        kpos = fk_pos.permute(0, 2, 3, 1).reshape(B, HW, C)
        kneg = fk_neg.permute(0, 2, 3, 1).reshape(B, HW, C)

        # sample same patch indices for q/pos/neg
        idx = torch.randperm(HW, device=q.device)[:K]
        q = q[:, idx, :]  # [B,K,C]
        kpos = kpos[:, idx, :]  # [B,K,C]
        kneg = kneg[:, idx, :]  # [B,K,C]

        # normalize
        q = F.normalize(q, dim=-1)
        kpos = F.normalize(kpos, dim=-1)
        kneg = F.normalize(kneg, dim=-1)

        # reshape to [BK,C]
        q = q.reshape(B * K, C)
        kpos = kpos.reshape(B * K, C)
        kneg = kneg.reshape(B * K, C)

        # key bank: [2BK, C]  (pos first, then neg)
        bank = torch.cat([kpos, kneg], dim=0)  # [2BK, C]

        # logits: [BK, 2BK]
        logits = (q @ bank.t()) / self.temperature

        # each q_i should match kpos_i => label=i
        labels = torch.arange(B * K, device=q.device)
        loss = F.cross_entropy(logits, labels)
        return loss / 5
