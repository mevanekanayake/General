from pathlib import Path
import h5py
import torch
import os
import numpy as np
from utils.fourier import fft2c as ft
from utils.fourier import ifft2c as ift
from utils.math import complex_center_crop, rss
import torch.nn.functional as F
from tqdm import tqdm


def transform_brain(kspacei, max_value):
    crop_size = (320, 320)

    kspacet = torch.tensor(np.stack((kspacei.real, kspacei.imag), axis=-1))
    image = ift(kspacet)
    image = rss(image)  # rss

    pad_size = crop_size[1] - image.shape[1]
    if pad_size > 0:
        pad_left = int(np.ceil(pad_size / 2))
        pad_right = int(np.floor(pad_size / 2))
        pad_tuple = (pad_left, pad_right)
        image = image.permute(2, 0, 1)
        image = F.pad(image, pad_tuple, "constant", 0)
        image = image.permute(1, 2, 0)

    image = complex_center_crop(image, crop_size)
    image = image / max_value
    kspace = ft(image)

    return kspace


def convert_():
    data_path = Path("D://Data//NYU_FastMRI//Brain_MRI//multicoil_val")
    save_path = Path("D://Data//NYU_FastMRI_V2//Brain_MRI//val")
    # data_path = Path("/fs02/ab57/Database/fastMRI/fastMriBrain/multicoil_train")
    # save_path = Path("/fs03/ab57/eeka0002/braintrain")

    files = list(Path(data_path).iterdir())

    for fname in tqdm(sorted(files), desc="Converting"):

        with h5py.File(fname, "r") as hf:
            max_val = torch.tensor(dict(hf.attrs)['max'])
            sequence = dict(hf.attrs)['acquisition']
            num_slices = torch.tensor(hf["kspace"].shape[0])
            ksps = torch.tensor(hf["kspace"][0:num_slices])

            ksp_list = []

            for j in range(num_slices):
                ksp = transform_brain(ksps[j], max_val)
                ksp_list.append(ksp)

            ksps = torch.stack(ksp_list, dim=0)
            if sequence == 'AXT1PRE':
                sequence = 'AXT1'
            save_name = fname.name.split('.')[0]

            info = {
                'kspace': ksps,
                'sequence': sequence,
            }

            torch.save(info, os.path.join(save_path, f'{save_name}.pt'))


if __name__ == '__main__':
    convert_()
    print('Done!')
