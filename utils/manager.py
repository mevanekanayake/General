import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import platform
import json
from datetime import datetime
import pandas as pd
from collections import OrderedDict
import time
import numpy as np
from utils import metrics
import random
import logging
import sys
import os
from pathlib import Path


def set_seed(seed=9):
    torch.manual_seed(seed)  # torch
    random.seed(seed)  # python
    np.random.seed(seed)  # numpy
    torch.cuda.manual_seed(seed)  # torch.cuda


def set_cuda(deterministic=True, benchmark=False):  # set deterministic to True if the input size remains same
    cudnn.deterministic = deterministic
    cudnn.benchmark = benchmark


def fetch_paths(dataset):
    node = platform.node()
    path_file = Path("utils\paths.json")
    f = json.load(open(path_file))
    data_path = Path(f[node][dataset])

    experiments_path = Path(f[node]["experiments"])
    folder_name = "Experiment_" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    exp_path = Path(os.path.join(experiments_path, folder_name))
    os.makedirs(exp_path)

    return data_path, exp_path


def set_logger(exp_path):
    logger = logging.getLogger()
    filehandler = logging.FileHandler(os.path.join(exp_path, f'{exp_path.name}_logs.log'))
    streamhandler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(message)s')
    streamhandler.setFormatter(formatter)
    filehandler.setFormatter(formatter)
    logger.addHandler(streamhandler)
    logger.addHandler(filehandler)
    logger.setLevel(logging.INFO)

    return logger


def set_device(model, args):
    if args.data_parallel:
        device_ids = [int(device_id) for device_id in args.data_parallel.split(',')]
        model = nn.DataParallel(model, device_ids=device_ids)
        args.device = f'cuda:{device_ids[0]}'
        model.to(args.device)
    else:
        model.to(args.device)
        device_ids = [0]

    return model, args, device_ids


class RunManager:
    epoch_start_time = float
    epoch_train_loss: torch.Tensor
    epoch_val_loss: torch.Tensor

    train_slice_count: int
    val_slice_count: int

    sequences: dict

    mse_vals: dict
    target_norms: dict
    ssim_vals: dict

    slice_stats: dict

    def __init__(self, experiments_path, ckpt, seq_types):

        self.experiments_path = experiments_path
        self.folder_name = experiments_path.name
        self.epoch_count = ckpt['epoch'] if ckpt else 0
        self.best_model_state_dict = ckpt['best_model_state_dict'] if ckpt else None
        self.best_val_loss = ckpt['best_val_loss'] if ckpt else float('inf')
        self.seq_types = seq_types
        self.summary = OrderedDict({"epoch_no": [],
                                    "epoch_duration": [],
                                    "train_loss": [],
                                    "val_loss": [],
                                    "val_NMSE": [],
                                    "val_PSNR": [],
                                    "val_SSIM": [],
                                    })
        for seq_type in self.seq_types:
            self.summary[f"{seq_type}_val_NMSE"] = []
            self.summary[f"{seq_type}_val_PSNR"] = []
            self.summary[f"{seq_type}_val_SSIM"] = []

    def begin_epoch(self):

        self.epoch_start_time = time.time()
        self.epoch_count += 1
        self.epoch_train_loss = 0
        self.epoch_val_loss = 0

        self.train_slice_count = 0
        self.val_slice_count = 0

        self.sequences = {}

        self.mse_vals = {}
        self.target_norms = {}

        self.slice_stats = {}

    def end_train_step(self, train_loss, size_of_train_batch):
        self.epoch_train_loss += train_loss * size_of_train_batch
        self.train_slice_count += size_of_train_batch

    def end_val_step(self, fnames, slice_nums, sequences, outputs, targets, val_loss):

        self.epoch_val_loss += val_loss * targets.shape[0]
        self.val_slice_count += targets.shape[0]

        for fname, slice_num, sequence, output, target in zip(fnames, slice_nums, sequences, outputs, targets):

            slice_num = slice_num.item()

            if fname not in self.sequences.keys():
                self.sequences[fname] = sequence
                self.mse_vals[fname] = {}
                self.target_norms[fname] = {}
                self.slice_stats[fname] = {}
                self.slice_stats[fname]["nmse"] = {}
                self.slice_stats[fname]["psnr"] = {}
                self.slice_stats[fname]["ssim"] = {}

            self.mse_vals[fname][slice_num] = torch.mean((target - output) ** 2)
            self.target_norms[fname][slice_num] = torch.mean((target - torch.zeros_like(target)) ** 2)

            # SAVING SLICE-WISE STATISTICS
            self.slice_stats[fname]["nmse"][slice_num] = (self.mse_vals[fname][slice_num] / self.target_norms[fname][slice_num]).item()
            self.slice_stats[fname]["psnr"][slice_num] = (10 * torch.log10(1.0 ** 2 / self.mse_vals[fname][slice_num])).item()
            self.slice_stats[fname]["ssim"][slice_num] = (metrics.ssim(target, output)).item()

    def end_epoch(self, model, optimizer):

        epoch_duration = time.time() - self.epoch_start_time
        volume_stats = {}

        for fname, sequence in self.sequences.items():
            v_mse_val = torch.mean(torch.tensor(list(self.mse_vals[fname].values())))
            v_target_norm = torch.mean(torch.tensor(list(self.target_norms[fname].values())))

            # SAVING VOLUME-WISE STATISTICS
            volume_stats[fname] = {}
            volume_stats[fname]["nmse"] = v_mse_val / v_target_norm
            volume_stats[fname]["psnr"] = 10 * torch.log10(1.0 ** 2 / v_mse_val)
            volume_stats[fname]["ssim"] = torch.mean(torch.tensor(list(self.slice_stats[fname]["ssim"].values())))
            volume_stats[fname]["num_slices"] = len(self.mse_vals[fname])

        avg_epoch_train_loss = self.epoch_train_loss / self.train_slice_count
        avg_epoch_val_loss = self.epoch_val_loss / self.val_slice_count

        # SAVE SUMMARY
        self.summary["epoch_no"].append(self.epoch_count)
        self.summary["epoch_duration"].append(epoch_duration)
        self.summary["train_loss"].append(avg_epoch_train_loss.item())
        self.summary["val_loss"].append(avg_epoch_val_loss.item())
        self.summary["val_NMSE"].append(torch.mean(torch.tensor([volume_stats[fname]["nmse"] for fname in volume_stats.keys()])).item())
        self.summary["val_PSNR"].append(torch.mean(torch.tensor([volume_stats[fname]["psnr"] for fname in volume_stats.keys()])).item())
        self.summary["val_SSIM"].append(torch.mean(torch.tensor([volume_stats[fname]["ssim"] for fname in volume_stats.keys()])).item())
        for seq_type in self.seq_types:
            self.summary[f"{seq_type}_val_NMSE"].append(torch.mean(torch.tensor([volume_stats[fname]["nmse"] for fname in volume_stats.keys() if self.sequences[fname] == seq_type])).item())
            self.summary[f"{seq_type}_val_PSNR"].append(torch.mean(torch.tensor([volume_stats[fname]["psnr"] for fname in volume_stats.keys() if self.sequences[fname] == seq_type])).item())
            self.summary[f"{seq_type}_val_SSIM"].append(torch.mean(torch.tensor([volume_stats[fname]["ssim"] for fname in volume_stats.keys() if self.sequences[fname] == seq_type])).item())

        pd.DataFrame.from_dict(self.summary, orient='columns').to_csv(Path(os.path.join(f'{self.experiments_path}', f'{self.folder_name}_summary.csv')), index=False)

        last_model_state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
        last_optimizer_state_dict = optimizer.state_dict()

        # SAVE VOLUME/SLICE-WISE STATS and STATES (only if best epoch)
        if avg_epoch_val_loss < self.best_val_loss:

            self.best_val_loss = avg_epoch_val_loss
            self.best_model_state_dict = last_model_state_dict

            volume_slice_stats = OrderedDict({})

            fnames = volume_stats.keys()
            volume_slice_stats["fname"] = fnames
            volume_slice_stats["sequence"] = [self.sequences[fname] for fname in fnames]
            volume_slice_stats["num_slices"] = [volume_stats[fname]["num_slices"] for fname in fnames]
            volume_slice_stats["NMSE"] = [volume_stats[fname]["nmse"].item() for fname in fnames]
            volume_slice_stats["PSNR"] = [volume_stats[fname]["psnr"].item() for fname in fnames]
            volume_slice_stats["SSIM"] = [volume_stats[fname]["ssim"].item() for fname in fnames]
            for slice_idx in range(max(volume_slice_stats["num_slices"])):
                volume_slice_stats[f"NMSE_S{str(slice_idx+1).zfill(2)}"] = [self.slice_stats[fname]["nmse"].get(slice_idx, '') for fname in fnames]
                volume_slice_stats[f"SSIM_S{str(slice_idx+1).zfill(2)}"] = [self.slice_stats[fname]["ssim"].get(slice_idx, '') for fname in fnames]
                volume_slice_stats[f"PSNR_S{str(slice_idx+1).zfill(2)}"] = [self.slice_stats[fname]["psnr"].get(slice_idx, '') for fname in fnames]

            pd.DataFrame.from_dict(volume_slice_stats, orient='columns').to_csv(Path(f'{self.experiments_path}', f'{self.folder_name}_volume_slice_stats.csv'), index=False)

        torch.save({'epoch': self.epoch_count,
                    'last_model_state_dict': last_model_state_dict,
                    'last_optimizer_state_dict': last_optimizer_state_dict,
                    'best_model_state_dict': self.best_model_state_dict,
                    'best_val_loss': self.best_val_loss,
                    }, os.path.join(self.experiments_path, f'{self.folder_name}_model.pth'))
