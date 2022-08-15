import torch

from utils.fourier import ifft2c as ift
from utils.math import complex_abs

from pathlib import Path
import matplotlib.pyplot as plt
import seaborn_image as isns
import os
from tqdm import tqdm

data_path = Path("D://Data//NYU_FastMRI_V2//Brain_MRI//val")
files = os.listdir(data_path)

for filename in tqdm(files):
    data = torch.load(os.path.join(data_path, filename))
    kspace_ori = data["kspace"]
    num_slices = kspace_ori.shape[0]
    target_list = []
    fname = filename.split('.')[0]
    for snum in range(num_slices):
        kspace_ori = data["kspace"][snum]
        target2 = ift(kspace_ori)
        target = complex_abs(target2)
        target_list.append(target)

    g = isns.ImageGrid(target_list, col_wrap=4, cbar=False, cmap='gray')
    plt.savefig(os.path.join(Path("D://Data//NYU_FastMRI_V2//Brain_MRI//val_images"), f'{fname}.jpg'), format='jpg', bbox_inches='tight', dpi=300)

print()
