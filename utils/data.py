import torch
from torch.utils.data import Dataset

import os
import random


class Data(Dataset):

    def __init__(self, root, train, nv=0, transform=None):

        self.root = root
        self.transform = transform
        self.key = "train" if train else "val"

        lib_path = os.path.join(root, "library.pt")
        self.library = torch.load(lib_path)
        self.seq_types = self.library[self.key].keys()

        self.examples = []
        self.selected_examples = []

        # nv!= 0 means partial dataset. nv= 0 means full dataset.
        if nv != 0:
            vols_per_seq = int(nv / len(self.seq_types))
            self.num_volumes = vols_per_seq * len(self.seq_types)
            for seq_type in self.seq_types:
                self.examples += [item for sublist in self.library[self.key][seq_type][:vols_per_seq] for item in sublist]
                self.selected_examples += random.choices([item[0][0] for item in self.library[self.key][seq_type][:vols_per_seq]], k=3) if not train else []
                # collect the file name of the first slice from randomly selected 3 volume from each sequence out of the sampled volumes
        else:
            self.num_volumes = sum([len(self.library[self.key][seq_type]) for seq_type in self.seq_types])
            for seq_type in self.seq_types:
                self.examples += [item for sublist in self.library[self.key][seq_type] for item in sublist]
                self.selected_examples += random.choices([item[0][0] for item in self.library[self.key][seq_type]], k=3) if not train else []
                # collect the file name of the first slice from randomly 3 selected volume from each sequence out of all volumes

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        fname, slice_id = self.examples[i]
        data = torch.load(os.path.join(self.root, self.key, f"{fname}.pt"))
        kspace = data["kspace"][slice_id]
        sequence = data["sequence"]
        sample = self.transform(kspace, fname, slice_id, sequence)

        return sample
