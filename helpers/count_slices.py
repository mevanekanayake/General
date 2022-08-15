import torch

import os
from pathlib import Path
import matplotlib.pyplot as plt

train_val = "train"
data_path = Path("D://Data//NYU_FastMRI_V2//Brain_MRI")
lib_path = os.path.join(data_path, "library.pt")
library = torch.load(lib_path)
count = {'AXT1': [], 'AXT1POST': [], 'AXT2': [], 'AXFLAIR': []}

for seq in library[train_val].keys():
    for vol in library[train_val][seq]:
        count[seq].append(len(vol))

for seq in count.keys():
    plt.hist(count[seq])
    plt.title(seq)
    plt.show()
