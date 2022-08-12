import torch
from collections import OrderedDict
from pathlib import Path
from tqdm import tqdm

seq_types = ["AXT1", "AXT1POST", "AXT2", "AXFLAIR"]
train_path = Path("D:\\Data\\NYU_FastMRI_V2\\Brain_MRI\\train")
val_path = Path("D:\\Data\\NYU_FastMRI_V2\\Brain_MRI\\val")

train_fpaths = sorted(list(Path(train_path).iterdir()))
val_fpaths = sorted(list(Path(val_path).iterdir()))

dataset_library = OrderedDict({})
dataset_library["train"] = OrderedDict({})
for seq_type in seq_types:
    dataset_library["train"][seq_type] = []
dataset_library["val"] = OrderedDict({})
for seq_type in seq_types:
    dataset_library["val"][seq_type] = []

for fpath in tqdm(sorted(train_fpaths), desc=f"Gathering training data"):
    data = torch.load(fpath)
    kspace = data["kspace"]
    sequence = data["sequence"]
    num_slices = kspace.shape[0]
    dataset_library["train"][sequence].append([(fpath.name.split('.')[0], slice_ind) for slice_ind in range(num_slices)])

for fpath in tqdm(sorted(val_fpaths), desc=f"Gathering validation data"):
    data = torch.load(fpath)
    kspace = data["kspace"]
    sequence = data["sequence"]
    num_slices = kspace.shape[0]
    dataset_library["val"][sequence].append([(fpath.name.split('.')[0], slice_ind) for slice_ind in range(num_slices)])

torch.save(dataset_library, 'library.pt')
