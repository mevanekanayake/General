import torch
from torch.utils.data import Dataset

import os
import random


class Data(Dataset):

    def __init__(self, root, train, nv=0, transform=None, viz=False):

        self.root = root
        self.transform = transform
        self.key = "train" if train else "val"

        lib_path = os.path.join(root, "library.pt")
        self.library = torch.load(lib_path)
        self.seq_types = self.library[self.key].keys()

        self.examples = []
        self.selected_examples = []

        # COLLECT ALL SLICES
        # nv!= 0 means partial dataset. nv= 0 means full dataset.
        if nv != 0:
            vols_per_seq = int(nv / len(self.seq_types))
            self.num_volumes = vols_per_seq * len(self.seq_types)
            for seq_type in self.seq_types:
                self.examples += [item for sublist in self.library[self.key][seq_type][:vols_per_seq] for item in sublist]
                self.selected_examples += random.sample([item[0] for item in self.library[self.key][seq_type][:vols_per_seq]], k=5) if not train else []
                # collect the file names of randomly selected 5 volumes from each sequence out of the sampled volumes
        else:
            self.num_volumes = sum([len(self.library[self.key][seq_type]) for seq_type in self.seq_types])
            for seq_type in self.seq_types:
                self.examples += [item for sublist in self.library[self.key][seq_type] for item in sublist]
                self.selected_examples += random.sample([item[0] for item in self.library[self.key][seq_type]], k=5) if not train else []
                # collect the file names of randomly selected 5 volumes from each sequence out of all volumes

        # TO CREATE VIZ DATASET
        self.examples = self.selected_examples if viz else self.examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        fname, slice_id = self.examples[i]
        data = torch.load(os.path.join(self.root, self.key, f"{fname}.pt"))
        kspace = data["kspace"][slice_id]
        sequence = data["sequence"]
        sample = self.transform(kspace, fname, slice_id, sequence)

        return sample
