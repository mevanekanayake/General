import torch

from utils.fourier import ifft2c as ift
from utils.math import complex_abs

import os
import math
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn_image as isns

data_path = Path("D://Data//NYU_FastMRI_V2//Brain_MRI")
lib_path = os.path.join(data_path, "library.pt")
library = torch.load(lib_path)

for seq in library["val"].keys():
    num_slices = len(library["val"][seq][0])
    fname = library["val"][seq][0][0][0]
    num_rows = math.ceil(num_slices / 4)
    num_cols = 4
    scale = 5

    # fig, axes = plt.subplots(1, 2, sharex=True, figsize=(10, 5))
    # fig.suptitle('Bigger 1 row x 2 columns axes with no data')
    # axes[0].set_title('Title of the first chart')

    # plt.figure(figsize=(num_rows*scale, num_cols*scale))
    # plt.subplots_adjust(hspace=0.5)
    # plt.suptitle(f"Daily closing prices", fontsize=18, y=0.95)
    target_list = []
    for snum in range(num_slices):
        data = torch.load(os.path.join(data_path, "val", f"{fname}.pt"))
        kspace_ori = data["kspace"][snum]
        target2 = ift(kspace_ori)
        target = complex_abs(target2)
        target_list.append(target)

    g = isns.ImageGrid(target_list, col_wrap=4, cbar=False, cmap='gray')
    plt.show()
    print()
