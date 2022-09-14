import random
import h5py
from pathlib import Path
import matplotlib.pyplot as plt
# import seaborn_image as isns
import os

data_path = Path("D://Data//NYU_FastMRI//Knee_MRI//singlecoil_train")
files = os.listdir(data_path)
filename = random.choice(files)

# filename = 'file1002303.h5'

with h5py.File(os.path.join(data_path, filename), "r") as hf:

    target = hf["reconstruction_esc"][14]
    target = target/target.max()
    plt.imshow(target)
    plt.show()

print()

print()
