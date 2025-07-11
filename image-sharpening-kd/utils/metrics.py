from skimage.metrics import structural_similarity as ssim
import numpy as np
import torch

# SSIM evaluation function
def calculate_ssim(img1, img2):
    img1 = torch.clamp(img1, 0, 1).detach().cpu().numpy().transpose(1, 2, 0)
    img2 = torch.clamp(img2, 0, 1).detach().cpu().numpy().transpose(1, 2, 0)
    return ssim(img1, img2, data_range=1.0, channel_axis=-1)
