import torch
from torch.utils.data import Dataset

from pathlib import Path
from tqdm import tqdm
import os


class Data(Dataset):

    def __init__(self, root, subfolder, vsr=1.0, transform=None):

        root = os.path.join(root, subfolder)
        self.transform = transform
        fpaths = sorted(list(Path(root).iterdir()))

        if vsr < 1.0:
            num_volumes = round(len(fpaths) * vsr)
            fpaths = fpaths[:num_volumes]

        self.examples = []
        self.num_volumes = len(fpaths)
        self.seq_types = []

        for fpath in tqdm(sorted(fpaths), desc=f"Gathering {subfolder} data"):
            data = torch.load(fpath)
            kspace = data["kspace"]
            self.seq_types.append(data["sequence"]) if data["sequence"] not in self.seq_types else None
            num_slices = kspace.shape[0]
            self.examples += [(fpath, slice_ind) for slice_ind in range(num_slices)]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        fpath, slice_id = self.examples[i]
        data = torch.load(fpath)
        kspace = data["kspace"][slice_id]
        sequence = data["sequence"]
        sample = self.transform(kspace, fpath.name.split('.')[0], slice_id, sequence)

        return sample
