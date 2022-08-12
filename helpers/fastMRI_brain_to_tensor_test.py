import os.path
from pathlib import Path
import torch
import matplotlib.pyplot as plt
from utils.fourier import ifft2c as ift
from utils.math import complex_abs
import h5py

saved_path = Path("D://Data//NYU_FastMRI_V2//Brain_MRI//train")
file = os.path.join(saved_path, 'file_brain_AXT1_202_2020095.pt')
test_file = torch.load(file)

kspace = test_file["kspace"]
sequence = test_file["sequence"]

for i in range(kspace.shape[0]):
    target = complex_abs(ift(kspace[i]))
    plt.imshow(target, cmap='gray')
    plt.show()
print()
