import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tqdm import tqdm
import argparse
from pathlib import Path

from utils.data import Data
from utils.transform import Transform
from utils.manager import set_seed, set_cuda, fetch_paths, set_logger, set_device, RunManager

from models.miccan import MICCAN


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
    parser.add_argument("--bs", type=int, default=8, help="Batch size for training and validation")
    parser.add_argument("--ne", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate for training")
    parser.add_argument("--dv", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device for model training")
    parser.add_argument("--dp", type=str, default=None, help="Whether to perform Data parallelism")

    # EXPERIMENT ARGS
    parser.add_argument("--ckpt", type=str, help="Continue trainings from checkpoint")
    parser.add_argument("--pf", type=int, default=10, help="Plotting frequency")

    # MODEL ARGS
    parser.add_argument('--blocktype', default='UCA', type=str, help='model')
    parser.add_argument('--nblock', default=5, type=int, help='number of block')
    parser.add_argument('--in_channels', default=2, type=int, help='number of input channels')
    parser.add_argument('--out_channels', default=2, type=int, help='number of output channels')

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
    model = MICCAN(args)
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
    loss = nn.MSELoss()

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

            for batch in train_epoch:
                train_epoch.set_description(f"Epoch {m.epoch_count} [Training]")

                xu = batch.image_zf2.to(args.dv)
                x = batch.target.to(args.dv)
                yu = batch.kspace_und.to(args.dv)
                msk = batch.mask.to(args.dv)

                optimizer.zero_grad()
                x_hat = model(xu, yu, msk)
                train_loss = loss(x_hat, x)
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

                    xu = batch.image_zf2.to(args.dv)
                    x = batch.target.to(args.dv)
                    yu = batch.kspace_und.to(args.dv)
                    msk = batch.mask.to(args.dv)
                    fname = batch.fname
                    slice_num = batch.slice_num
                    sequence = batch.sequence

                    x_hat = model(xu, yu, msk)
                    val_loss = loss(x_hat, x)

                    # END VALIDATION STEP
                    val_epoch.set_postfix(val_loss=val_loss.detach().item())
                    m.end_val_step(fname, slice_num, sequence, batch.image_zf, x_hat.to('cpu'), x.to('cpu'), val_loss.to('cpu'))

            # END EPOCH
            m.end_epoch(model, optimizer, logger)

            # VISUALIZATION
            if m.best_epoch or (m.epoch_count % args.pf == 0):
                with tqdm(viz_loader, unit="batch") as viz_epoch:
                    for batch in viz_epoch:
                        viz_epoch.set_description(f"Epoch {m.epoch_count} [Visualization]")

                        xu = batch.image_zf2.to(args.dv)
                        x = batch.target.to(args.dv)
                        yu = batch.kspace_und.to(args.dv)
                        msk = batch.mask.to(args.dv)
                        fname = batch.fname
                        slice_num = batch.slice_num
                        sequence = batch.sequence

                        x_hat = model(xu, yu, msk)

                        # END VISUALIZATION STEP
                        m.visualize(fname, slice_num, sequence, batch.image_zf, x_hat.to('cpu'), x.to('cpu'))


if __name__ == '__main__':
    train_()
    print('Done!')
