# Check whether psnr and ssim metric in scikit image and torch metrics are the same

import torch
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr_np
from skimage.metrics import structural_similarity as ssim_np
from torchmetrics.functional import peak_signal_noise_ratio as psnr_torch
from torchmetrics.functional import structural_similarity_index_measure as ssim_torch

t1 = np.random.randn(320, 320)
t2 = np.random.randn(320, 320)

# np
psnr1 = psnr_np(t2, t1, data_range=1.0)
ssim1 = ssim_np(t2, t1, data_range=1.0)
nmse1 = np.linalg.norm(t2 - t1) ** 2 / np.linalg.norm(t2) ** 2
print('PSNR (np): %f\nSSIM (np): %f\nNMSE (np): %f' % (psnr1, ssim1, nmse1))

# torch
psnr2 = psnr_torch(torch.tensor(t1, dtype=torch.float32).unsqueeze(0).unsqueeze(0), torch.tensor(t2, dtype=torch.float32).unsqueeze(0).unsqueeze(0), data_range=1.0)
ssim2 = ssim_torch(torch.tensor(t1, dtype=torch.float32).unsqueeze(0).unsqueeze(0), torch.tensor(t2, dtype=torch.float32).unsqueeze(0).unsqueeze(0), data_range=1.0, gaussian_kernel=False, kernel_size=7)
nmse2 = torch.linalg.norm(torch.tensor(t2) - torch.tensor(t1)) ** 2 / torch.linalg.norm(torch.tensor(t2)) ** 2
print('PSNR (np): %f\nSSIM (np): %f\nNMSE (np): %f' % (psnr2, ssim2, nmse2))

print()
