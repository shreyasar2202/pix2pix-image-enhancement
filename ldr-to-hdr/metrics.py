import torch
import sewar
import numpy as np
import torchvision.transforms as transforms

def psnr(original, generated):
    max_pixel_value = torch.tensor(255)
    original = (((original.detach().permute(0,2,3,1))*0.25 + 0.5) * 255)
    generated = (((generated.detach().permute(0,2,3,1))*0.25 + 0.5) * 255)
    mse = torch.mean(torch.square(original - generated))
    return 10*torch.log10(max_pixel_value**2 / mse).item()

def ssim(original, generated):
    original = (((original.detach().permute(0,2,3,1))*0.25 + 0.5) * 255).cpu().numpy()
    original = (original).astype(np.int32)
    generated = (((generated.detach().permute(0,2,3,1))*0.25 + 0.5) * 255).cpu().numpy()
    generated = (generated).astype(np.int32)
    total_ssim = 0
    for i in range(original.shape[0]):       
        total_ssim += np.mean(sewar.ssim(original[i], generated[i], MAX=255))
        
    return total_ssim/original.shape[0]

def uqi(original, generated):
    original = (((original.detach().permute(0,2,3,1))*0.25 + 0.5) * 255).cpu().numpy()
    original = (original).astype(np.int32)
    generated = (((generated.detach().permute(0,2,3,1))*0.25 + 0.5) * 255).cpu().numpy()
    generated = (generated).astype(np.int32)
    total_uqi = 0
    for i in range(original.shape[0]):       
        total_uqi += np.mean(sewar.uqi(original[i], generated[i]))
        
    return total_uqi/original.shape[0]