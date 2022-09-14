import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tqdm import tqdm
import argparse
from pathlib import Path

from utils.data import Data
from utils.math import complex_abs
from utils.fourier import ifft2c as ift
from utils.transform import Transform
from utils.manager import set_seed, set_cuda, fetch_paths, set_logger, set_device, RunManager

from models.swinunet import SwinUnet


def train_():
    # SET ARGUMENTS
    parser = argparse.ArgumentParser()

    # DATA ARGS
    parser.add_argument("--acc", type=list, default=[4], help="Acceleration factors for the k-space undersampling")
    parser.add_argument("--tnv", type=int, default=80, help="Number of volumes used for training [set to 0 for the full dataset]")
    parser.add_argument("--vnv", type=int, default=20, help="Number of volumes used for validation [set to 0 for the full dataset]")
    parser.add_argument("--mtype", type=str, default="random", choices=("random", "equispaced"), help="Type of k-space mask")
    parser.add_argument("--dset", type=str, default="fastmribrain", choices=("fastmriknee", "fastmribrain"), help="Which dataset to use")

    # TRAIN ARGS
    parser.add_argument("--bs", type=int, default=8, help="Batch size for training and validation")
    parser.add_argument("--ne", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate for training")
    parser.add_argument("--dv", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device for model training")
    parser.add_argument("--dp", type=str, default=None, help="Whether to perform Data parallelism")

    # EXPERIMENT ARGS
    parser.add_argument("--ckpt", type=str, help="Continue trainings from checkpoint")
    parser.add_argument("--pf", type=int, default=2, help="Plotting frequency")

    # MODEL ARGS
    parser.add_argument("--embed_dim", type=int, default=96, help="Embedding dimension")

    # LOAD ARGUMENTS
    args = parser.parse_args()

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
    model = SwinUnet(args.embed_dim)
    model.load_state_dict(ckpt['last_model_state_dict']) if args.ckpt else None
    logger.info(f'No. of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    # SET GPUS
    model, args, device_ids = set_device(model, args)
    logger.info(f'num GPUs available: {torch.cuda.device_count()}')
    logger.info(f'num GPUs using: {len(device_ids)}')
    logger.info(f'GPU model: {torch.cuda.get_device_name(args.dv)}') if torch.cuda.device_count() > 0 else None

    # LOAD TRAINING DATA
    train_transform = Transform(train=True, mask_type=args.mtype, accelerations=args.acc)
    train_dataset = Data(root=data_path, train=True, transform=train_transform, nv=args.tnv)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.bs, num_workers=0, shuffle=True, pin_memory=True)
    logger.info(f'Training set: No. of volumes: {train_dataset.num_volumes} | No. of slices: {len(train_dataset)}')

    # LOAD VALIDATION DATA
    val_transform = Transform(train=False, mask_type=args.mtype, accelerations=args.acc)
    val_dataset = Data(root=data_path, train=False, transform=val_transform, nv=args.vnv)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.bs, num_workers=0, shuffle=False, pin_memory=True)
    logger.info(f'Validation set: No. of volumes: {val_dataset.num_volumes} | No. of slices: {len(val_dataset)}')

    # LOAD VISUALIZATION DATA
    viz_dataset = Data(root=data_path, train=False, transform=val_transform, nv=args.vnv, viz=True)
    viz_loader = DataLoader(dataset=viz_dataset, batch_size=args.bs, num_workers=0, shuffle=False, pin_memory=True)
    logger.info(f'Visualization set: No. of slices: {len(viz_dataset)}')

    # LOSS FUNCTION
    loss = nn.MSELoss()

    # SET OPTIMIZER
    logger.info(f'Optimizer: RMSprop')
    optimizer = torch.optim.RMSprop(params=model.parameters(), lr=args.lr)
    optimizer.load_state_dict(ckpt['last_optimizer_state_dict']) if args.ckpt else None

    # INITIALIZE RUN MANAGER
    m = RunManager(exp_path, ckpt, val_dataset.seq_types, args.pf, val_dataset.selected_examples)

    # LOOP
    for _ in range(args.ne):
        # BEGIN EPOCH
        m.begin_epoch()

        # BEGIN TRAINING LOOP
        model.train()
        with tqdm(train_loader, unit="batch") as train_epoch:

            for batch in train_epoch:
                train_epoch.set_description(f"Epoch {m.epoch_count} [Training]")

                yu = batch.kspace_und.to(args.dv)
                y = batch.kspace.to(args.dv)
                msk = batch.mask.to(args.dv)

                optimizer.zero_grad()
                y_hat = (1 - msk) * model(yu) + msk * y
                train_loss = loss(y_hat, y)
                train_loss.backward()
                optimizer.step()

                # END TRAINING STEP
                train_epoch.set_postfix(train_loss=train_loss.detach().item())
                m.end_train_step(train_loss.detach().to('cpu'), batch[0].shape[0])

        model.eval()
        with torch.no_grad():

            # BEGIN VALIDATION LOOP
            with tqdm(val_loader, unit="batch") as val_epoch:
                for batch in val_epoch:
                    val_epoch.set_description(f"Epoch {m.epoch_count} [Validation]")

                    yu = batch.kspace_und.to(args.dv)
                    y = batch.kspace.to(args.dv)
                    msk = batch.mask.to(args.dv)
                    fname = batch.fname
                    slice_num = batch.slice_num
                    sequence = batch.sequence

                    y_hat = (1 - msk) * model(yu) + msk * y
                    val_loss = loss(y_hat, y)

                    output = complex_abs(ift(y_hat.permute(0, 2, 3, 1))).unsqueeze(1)

                    # END VALIDATION STEP
                    val_epoch.set_postfix(val_loss=val_loss.detach().item())
                    m.end_val_step(fname, slice_num, sequence, batch.image_zf, output.to('cpu'), batch.target, val_loss.to('cpu'))

            # END EPOCH
            m.end_epoch(model, optimizer, logger)

            # VISUALIZATION
            if m.best_epoch or (m.epoch_count % args.pf == 0):
                with tqdm(viz_loader, unit="batch") as viz_epoch:
                    for batch in viz_epoch:
                        viz_epoch.set_description(f"Epoch {m.epoch_count} [Visualization]")

                        yu = batch.kspace_und.to(args.dv)
                        y = batch.kspace.to(args.dv)
                        msk = batch.mask.to(args.dv)
                        fname = batch.fname
                        slice_num = batch.slice_num
                        sequence = batch.sequence

                        y_hat = (1 - msk) * model(yu) + msk * y

                        output = complex_abs(ift(y_hat.permute(0, 2, 3, 1))).unsqueeze(1)

                        # END VISUALIZATION STEP
                        m.visualize(fname, slice_num, sequence, batch.image_zf, output.to('cpu'), batch.target)


if __name__ == '__main__':
    train_()
    print('Done!')
