import warnings

from PIL import Image
import random
import numpy as np
from math import sqrt
import torch

def mixup(src, trg, alpha):
    lam = np.random.uniform(0, alpha)

    mix_img = lam * src + (1 - lam) * trg

    return mix_img

    # if lam > 0.5:
    #     return mix_img
    # else:
    #     return trg


# abs : amplitude, angle : phase
def fft_amp_mix(src, int_src):
    fft_src = torch.fft.fftn(src)
    abs_src, angle_src = torch.abs(fft_src), torch.angle(fft_src)

    fft_int_src = torch.fft.fftn(int_src)
    abs_int_src, angle_int_src = torch.abs(fft_int_src), torch.angle(fft_int_src)

    fft_src = abs_int_src * torch.exp((1j) * angle_src)

    mixed_img = torch.abs(torch.fft.ifftn(fft_src))

    return mixed_img

def fft_amp_replace(source, target):
    fft_src = torch.fft.fftn(source)
    abs_src, angle_src = torch.abs(fft_src), torch.angle(fft_src)

    fft_trg = torch.fft.fftn(target)
    abs_trg, angle_trg = torch.abs(fft_trg), torch.angle(fft_trg)

    fft_src = abs_trg * torch.exp((1j) * angle_src)
    mixed_img = torch.abs(torch.fft.ifftn(fft_src))
    return mixed_img

def fda(source, target, ratio=1.0):
    b, c, h, w, d = source.shape
    h_crop = int(h * sqrt(ratio))
    w_crop = int(w * sqrt(ratio))
    d_crop = int(d * sqrt(ratio))

    h_start = h // 2 - h_crop // 2
    w_start = w // 2 - w_crop // 2
    d_start = d // 2 - d_crop // 2

    src_fft = torch.fft.fftn(source)
    src_abs, src_pha = torch.abs(src_fft), torch.angle(src_fft)
    trg_fft = torch.fft.fftn(target)
    trg_abs, trg_pha = torch.abs(trg_fft), torch.angle(trg_fft)

    src_abs[h_start:h_start+h_crop, w_start:w_start+w_crop, d_start:d_start+d_crop] = trg_abs[h_start:h_start+h_crop, w_start:w_start+w_crop, d_start:d_start+d_crop]

    fft_src = src_abs * torch.exp((1j) * src_pha)

    new_img = torch.abs(torch.fft.ifftn(fft_src))

    return new_img

def fft_mixup_block(source, target, ratio=1.0):
    lam = np.random.uniform(0, 1.0)
    b, c, h, w, d = source.shape
    h_crop = int(h * sqrt(ratio))
    w_crop = int(w * sqrt(ratio))
    d_crop = int(d * sqrt(ratio))

    h_start = h // 2 - h_crop // 2
    w_start = w // 2 - w_crop // 2
    d_start = d // 2 - d_crop // 2

    src_fft = torch.fft.fftn(source)
    src_abs, src_pha = torch.abs(src_fft), torch.angle(src_fft)
    trg_fft = torch.fft.fftn(target)
    trg_abs, trg_pha = torch.abs(trg_fft), torch.angle(trg_fft)

    trg_abs[h_start:h_start + h_crop, w_start:w_start + w_crop, d_start:d_start + d_crop] = lam * src_abs[h_start:h_start+h_crop, w_start:w_start+w_crop, d_start:d_start+d_crop] + (1 - lam) * trg_abs[h_start:h_start+h_crop, w_start:w_start+w_crop, d_start:d_start+d_crop]
    fft_trg = trg_abs * torch.exp((1j) * trg_pha)
    mixed_img = torch.abs(torch.fft.ifftn(fft_trg))


    return mixed_img

def fft_mixup(src, trg):
    lam = np.random.uniform(0, 1.0)
   
    fft_src = torch.fft.fftn(src)
    abs_src, angle_src = torch.abs(fft_src), torch.angle(fft_src)

    fft_trg = torch.fft.fftn(trg)
    abs_trg, angle_trg = torch.abs(fft_trg), torch.angle(fft_trg)

    abs_mix = lam * abs_src + (1 - lam) * abs_trg

    fft_trg = abs_mix * torch.exp((1j) * angle_trg)

    mixed_img = torch.abs(torch.fft.ifftn(fft_trg))

    return mixed_img


def fft_mixup_np(src, trg):
    #lam = np.random.uniform(0, 1.0)
    lam = 0.5
    fft_src = np.fft.fftshift(np.fft.fftn(src))
    abs_src, angle_src = np.abs(fft_src), np.angle(fft_src)

    fft_trg = np.fft.fftshift(np.fft.fftn(trg))
    abs_trg, angle_trg = np.abs(fft_trg), np.angle(fft_trg)

    abs_mix = lam * abs_src + (1 - lam) * abs_trg

    fft_trg = abs_mix * np.exp((1j) * angle_trg)

    #mixed_img = np.abs(np.fft.ifftn(fft_trg))
    mixed_img = np.abs(np.fft.ifftn(np.fft.ifftshift(fft_trg)))

    return mixed_img


def fft_amp_mix_np(src, int_src):
    fft_src = np.fft.fftshift(np.fft.fftn(src))
    abs_src, angle_src = np.abs(fft_src), np.angle(fft_src)

    fft_int_src = np.fft.fftshift(np.fft.fftn(int_src))
    abs_int_src, angle_int_src = np.abs(fft_int_src), np.angle(fft_int_src)

    fft_src = abs_int_src * np.exp((1j) * angle_src)

    #mixed_img = np.abs(np.fft.ifftn(fft_src))
    mixed_img = np.abs(np.fft.ifftn(np.fft.ifftshift(fft_src)))

    return mixed_img




