import torch

import re
from typing import NamedTuple

from utils.fourier import ifft2c as ift
from utils.math import complex_abs
from utils.mask import apply_random_mask


class Sample(NamedTuple):
    kspace: torch.Tensor
    kspace_und: torch.Tensor
    mask: torch.Tensor
    image_zf: torch.Tensor
    image_zf2: torch.Tensor
    target: torch.Tensor
    target2: torch.Tensor
    fname: str
    slice_num: int
    sequence: str


class Transform:

    def __init__(self, train, mask_type, accelerations):
        self.train = train
        self.mask_type = mask_type
        self.accelerations = accelerations

    def __call__(self, kspace_ori, fname, slice_num, sequence):
        target2 = ift(kspace_ori)
        target = complex_abs(target2)

        seed = int("".join(re.findall(r"\d+", fname))) if not self.train else None

        if self.mask_type == 'random':
            kspace_und, mask = apply_random_mask(kspace_ori, self.accelerations, seed)
        else:
            kspace_und = kspace_ori
            mask = None

        image_zf2 = ift(kspace_und)
        image_zf = complex_abs(image_zf2)

        sample = Sample(
            kspace=kspace_ori.permute(2, 0, 1),
            kspace_und=kspace_und.permute(2, 0, 1),
            mask=mask.unsqueeze(0),
            image_zf=image_zf.unsqueeze(0),
            image_zf2=image_zf2.permute(2, 0, 1),
            target=target.unsqueeze(0),
            target2=target2.permute(2, 0, 1),
            fname=fname,
            slice_num=slice_num,
            sequence=sequence
        )

        return sample
