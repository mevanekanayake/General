import torch
from torch.utils.data import Dataset

import os
import random


class Data(Dataset):

    def __init__(self, root, train=False, nv=0, transform=None):

        self.root = root
        self.transform = transform
        self.train = train

        lib_path = os.path.join(root, "library.pt")
        self.library = torch.load(lib_path)
        self.seq_types = self.library["train"].keys()

        self.examples = []
        self.selected_examples = []

        # nv!= 0 means partial dataset. nv= 0 means full dataset.
        if nv != 0:
            n = int(nv / len(self.seq_types))
            self.num_volumes = n * len(self.seq_types)
            for seq_type in self.seq_types:
                if train:
                    self.examples += [item for sublist in self.library["train"][seq_type][:n] for item in sublist]
                else:
                    self.examples += [item for sublist in self.library["val"][seq_type][:n] for item in sublist]
                    self.selected_examples.append(random.choice([item[0][0] for item in self.library["val"][seq_type][:n]]))

        else:
            self.num_volumes = sum([len(self.library[f"{'train' if self.train else 'val'}"][seq_type]) for seq_type in self.seq_types])
            for seq_type in self.seq_types:
                if train:
                    self.examples += [item for sublist in self.library["train"][seq_type] for item in sublist]
                else:
                    self.examples += [item for sublist in self.library["val"][seq_type] for item in sublist]
                    self.selected_examples.append(random.choice([item[0][0] for item in self.library["val"][seq_type]]))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        fname, slice_id = self.examples[i]
        data = torch.load(os.path.join(self.root, f"{'train' if self.train else 'val'}", f"{fname}.pt"))
        kspace = data["kspace"][slice_id]
        sequence = data["sequence"]
        sample = self.transform(kspace, fname, slice_id, sequence)

        return sample
