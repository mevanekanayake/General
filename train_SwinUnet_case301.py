import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tqdm import tqdm
import argparse
from pathlib import Path

from utils.data import Data
from utils.transform import Transform
from utils.manager import set_seed, set_cuda, fetch_paths, set_logger, set_device, RunManager
from utils.fourier import fft2c as ft
from utils.fourier import ifft2c as ift
from utils.math import complex_abs

from models.swinunet import SwinUnet


class Cascaded_SwinUnet(nn.Module):
    def __init__(self, embed_dim, in_chans, out_chans, num_iter):
        super(Cascaded_SwinUnet, self).__init__()

        self.num_iter = num_iter
        self.swin_unet = SwinUnet(embed_dim, in_chans, out_chans)

    def forward(self, loss_function, batch, args):
        yu = batch.kspace_und.to(args.dv)
        m = batch.mask.to(args.dv)
        x = batch.target.to(args.dv)

        y_temp = yu
        for i in range(self.num_iter):
            x_model = self.swin_unet(y_temp)
            y_model = ft(x_model.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            y_dc = (1 - m) * y_model + yu
            y_temp = y_dc

        x_hat = complex_abs(ift(y_temp.permute(0, 2, 3, 1))).unsqueeze(1)
        loss = loss_function(x_hat, x)

        return loss, x_hat


def train_():
    # SET ARGUMENTS
    parser = argparse.ArgumentParser()

    # DATA ARGS
    parser.add_argument("--acc", type=list, default=[4], help="Acceleration factors for the k-space undersampling")
    parser.add_argument("--tnv", type=int, default=80, help="Number of volumes used for training [set to 0 for the full dataset]")
    parser.add_argument("--vnv", type=int, default=20, help="Number of volumes used for validation [set to 0 for the full dataset]")
    parser.add_argument("--mtype", type=str, default="random", choices=("random", "equispaced"), help="Type of k-space mask")
    parser.add_argument("--dset", type=str, default="fastmribrain", choices=("fastmriknee", "fastmribrain"), help="Which dataset to use")
    parser.add_argument("--seq_types", type=str, default="AXT1,AXT1POST,AXT2,AXFLAIR", help="Which sequence types to use")

    # TRAIN ARGS
    parser.add_argument("--bs", type=int, default=4, help="Batch size for training and validation")
    parser.add_argument("--ne", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate for training")
    parser.add_argument("--dv", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device for model training")
    parser.add_argument("--dp", type=str, default=None, help="Whether to perform Data parallelism")

    # EXPERIMENT ARGS
    parser.add_argument("--ckpt", type=str, help="Continue trainings from checkpoint")
    parser.add_argument("--pf", type=int, default=10, help="Plotting frequency")

    # MODEL ARGS
    parser.add_argument("--embed_dim", type=int, default=96, help="Embedding dimension")
    parser.add_argument("--in_chans", type=int, default=2, help="Input channels")
    parser.add_argument("--out_chans", type=int, default=2, help="Ouput channels")
    parser.add_argument("--num_iter", type=int, default=5, help="Number of iterations")

    # LOAD ARGUMENTS
    args = parser.parse_args()
    args.seq_types = args.seq_types.split(',')

    # LOAD CHECKPOINT
    ckpt = torch.load(Path(args.ckpt), map_location='cpu') if args.ckpt else None
    args.ne = args.ne - ckpt['epoch'] if args.ckpt else args.ne

    # SET SEED
    set_seed()

    # SET CUDA
    set_cuda()

    # SET/CREATE PATHS
    data_path, exp_path = fetch_paths(args.dset)

    # LOG ARGS, PATHS
    logger = set_logger(exp_path)
    for entry in vars(args):
        logger.info(f'{entry}: {vars(args)[entry]}')
    logger.info(f'data_path = {str(data_path)}')
    logger.info(f'experiment_path = {str(exp_path)}')

    # LOAD MODEL
    model = Cascaded_SwinUnet(args.embed_dim, args.in_chans, args.out_chans, args.num_iter)
    model.load_state_dict(ckpt['last_model_state_dict']) if args.ckpt else None
    logger.info(f'No. of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    # SET GPUS
    model, args, device_ids = set_device(model, args)
    logger.info(f'num GPUs available: {torch.cuda.device_count()}')
    logger.info(f'num GPUs using: {len(device_ids)}')
    logger.info(f'GPU model: {torch.cuda.get_device_name(args.dv)}') if torch.cuda.device_count() > 0 else None

    # LOAD TRAINING DATA
    train_transform = Transform(train=True, mask_type=args.mtype, accelerations=args.acc)
    train_dataset = Data(root=data_path, train=True, seq_types=args.seq_types, transform=train_transform, nv=args.tnv)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.bs, num_workers=0, shuffle=True, pin_memory=True)
    logger.info(f'Training set: No. of volumes: {train_dataset.num_volumes} | No. of slices: {len(train_dataset)}')
    logger.info(f'{train_dataset.data_per_seq[:-1]}')

    # LOAD VALIDATION DATA
    val_transform = Transform(train=False, mask_type=args.mtype, accelerations=args.acc)
    val_dataset = Data(root=data_path, train=False, seq_types=args.seq_types, transform=val_transform, nv=args.vnv)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.bs, num_workers=0, shuffle=False, pin_memory=True)
    logger.info(f'Validation set: No. of volumes: {val_dataset.num_volumes} | No. of slices: {len(val_dataset)}')
    logger.info(f'{val_dataset.data_per_seq[:-1]}')

    # LOAD VISUALIZATION DATA
    viz_dataset = Data(root=data_path, train=False, seq_types=args.seq_types, transform=val_transform, nv=args.vnv, viz=True)
    viz_loader = DataLoader(dataset=viz_dataset, batch_size=args.bs, num_workers=0, shuffle=False, pin_memory=True)

    # LOSS FUNCTION
    loss_fn = nn.MSELoss()

    # SET OPTIMIZER
    logger.info(f'Optimizer: RMSprop')
    optimizer = torch.optim.RMSprop(params=model.parameters(), lr=args.lr)
    optimizer.load_state_dict(ckpt['last_optimizer_state_dict']) if args.ckpt else None

    # INITIALIZE RUN MANAGER
    m = RunManager(exp_path, ckpt, args.seq_types, args.pf, val_dataset.selected_examples)

    # LOOP
    for _ in range(args.ne):
        # BEGIN EPOCH
        m.begin_epoch()

        # BEGIN TRAINING LOOP
        model.train()
        with tqdm(train_loader, unit="batch") as train_epoch:

            for b in train_epoch:
                train_epoch.set_description(f"Epoch {m.epoch_count} [Training]")

                optimizer.zero_grad()

                train_loss, _ = model(loss_fn, b, args)

                train_loss.backward()
                optimizer.step()

                # END TRAINING STEP
                train_epoch.set_postfix(train_loss=train_loss.detach().item())
                m.end_train_step(train_loss.detach().to('cpu'), b[0].shape[0])

        model.eval()
        with torch.no_grad():

            # BEGIN VALIDATION LOOP
            with tqdm(val_loader, unit="batch") as val_epoch:
                for b in val_epoch:
                    val_epoch.set_description(f"Epoch {m.epoch_count} [Validation]")

                    val_loss, x_hat = model(loss_fn, b, args)

                    # END VALIDATION STEP
                    val_epoch.set_postfix(val_loss=val_loss.detach().item())
                    m.end_val_step(b.fname, b.slice_num, b.sequence, b.image_zf, x_hat.to('cpu'), b.target, val_loss.to('cpu'))

            # END EPOCH
            m.end_epoch(model, optimizer, logger)

            # VISUALIZATION
            if m.best_epoch or (m.epoch_count % args.pf == 0):
                with tqdm(viz_loader, unit="batch") as viz_epoch:
                    for b in viz_epoch:
                        viz_epoch.set_description(f"Epoch {m.epoch_count} [Visualization]")

                        _, x_hat = model(loss_fn, b, args)

                        # END VISUALIZATION STEP
                        m.visualize(b.fname, b.slice_num, b.sequence, b.image_zf, x_hat.to('cpu'), b.target)


if __name__ == '__main__':
    train_()
    print('Done!')
