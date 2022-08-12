import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tqdm import tqdm
import argparse
from pathlib import Path

from utils.data import Data
from utils.transform import Transform
from utils.manager import set_seed, set_cuda, fetch_paths, set_logger, set_device, RunManager

from models.unet import Unet


def train_():
    # SET ARGUMENTS
    parser = argparse.ArgumentParser()

    # DATA ARGS
    parser.add_argument("--acc", type=list, default=[2, 4, 6, 8], help="Acceleration factors for the k-space undersampling")
    parser.add_argument("--tvsr", type=float, default=1., help="Fraction of data volumes used for training")
    parser.add_argument("--vvsr", type=float, default=1., help="Fraction of data volumes used for validation")
    parser.add_argument("--mtype", type=str, choices=("random", "equispaced"), default="random", help="Type of k-space mask")
    parser.add_argument("--dset", choices=("fastmriknee", "fastmribrain"), default="fastmribrain", type=str, help="Which dataset is used")

    # TRAIN ARGS
    parser.add_argument("--bs", type=int, default=8, help="Batch size for training and validation")
    parser.add_argument("--ne", type=int, default=5, help="Number of epochs for training")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--dv", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device for model training")
    parser.add_argument("--dp", type=str, default=None, help="Whether to perform Data parallelism")
    parser.add_argument("--ckpt", type=str, help="Continue trainings from checkpoint")

    # MODEL ARGS
    parser.add_argument("--in_chans", type=int, default=1, help="Number of channels in the input image ")
    parser.add_argument("--out_chans", type=int, default=1, help="Number of channels in the output image ")
    parser.add_argument("--chans", type=int, default=32, help="Number of channels in top Layer")
    parser.add_argument("--num_pool_layers", type=int, default=4, help="Number of pooling layers of the U-Net")

    # LOAD ARGUMENTS
    args = parser.parse_args()

    # LOAD CHECKPOINT
    ckpt = torch.load(Path(args.checkpoint), map_location='cpu') if args.checkpoint else None
    args.num_epochs = args.num_epochs - ckpt['epoch'] if args.checkpoint else args.num_epochs

    # SET SEED
    set_seed()

    # SET CUDA
    set_cuda()

    # SET/CREATE PATHS
    data_path, exp_path = fetch_paths(args.dataset)

    # LOG ARGS, PATHS
    logger = set_logger(exp_path)
    for entry in vars(args):
        logger.info(f'{entry}: {vars(args)[entry]}')
    logger.info(f'data_path = {str(data_path)}')
    logger.info(f'experiments_path = {str(exp_path)}')

    # LOAD MODEL
    model = Unet(args)
    model.load_state_dict(ckpt['last_model_state_dict']) if args.checkpoint else None
    logger.info(f'No. of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    # SET GPUS
    model, args, device_ids = set_device(model, args)
    logger.info(f'num GPUs available: {torch.cuda.device_count()}')
    logger.info(f'num GPUs using: {len(device_ids)}')
    logger.info(f'GPU model: {torch.cuda.get_device_name(args.device)}') if torch.cuda.device_count() > 0 else None

    # LOAD TRAINING DATA
    train_transform = Transform(train=True, mask_type=args.mask_type, accelerations=args.acceleration_factors)
    train_dataset = Data(root=data_path, subfolder="train", transform=train_transform, vsr=args.tvsr)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, num_workers=1, shuffle=True, pin_memory=True)
    logger.info(f'Training set gathered: No. of volumes: {train_dataset.num_volumes} | No. of slices: {len(train_dataset)}')

    # LOAD VALIDATION DATA
    val_transform = Transform(train=False, mask_type=args.mask_type, accelerations=args.acceleration_factors)
    val_dataset = Data(root=data_path, subfolder="val", transform=val_transform, vsr=args.vvsr)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, num_workers=1, shuffle=False, pin_memory=True)
    logger.info(f'Validation set gathered: No. of volumes: {val_dataset.num_volumes} | No. of slices: {len(val_dataset)}')

    # LOSS FUNCTION
    loss = nn.L1Loss()

    # SET OPTIMIZER
    logger.info(f'Optimizer: RMSprop')
    optimizer = torch.optim.RMSprop(params=model.parameters(),
                                    lr=args.lr,
                                    weight_decay=0.)
    optimizer.load_state_dict(ckpt['optimizer_state_dict']) if args.checkpoint else None

    # INITIALIZE RUN MANAGER
    m = RunManager(exp_path, ckpt, val_dataset.seq_types)

    # LOOP
    for _ in range(args.num_epochs):
        # BEGIN EPOCH
        m.begin_epoch()

        # BEGIN TRAINING LOOP
        model.train()
        with tqdm(train_loader, unit="batch") as train_epoch:

            for train_batch in train_epoch:
                train_epoch.set_description(f"Epoch {m.epoch_count} [Training]")

                image = train_batch[2].to(args.device)
                target = train_batch[4].to(args.device)

                optimizer.zero_grad()
                output = model(image)
                train_loss = loss(output, target)
                train_loss.backward()
                optimizer.step()

                # END TRAINING STEP
                train_epoch.set_postfix(train_loss=train_loss.detach())
                m.end_train_step(train_loss.detach().to('cpu'), train_batch[0].shape[0])

        # BEGIN VALIDATION LOOP
        model.eval()
        with torch.no_grad():
            with tqdm(val_loader, unit="batch") as val_epoch:
                for val_batch in val_epoch:
                    val_epoch.set_description(f"Epoch {m.epoch_count} [Validation]")

                    image = val_batch[2].to(args.device)
                    target = val_batch[4].to(args.device)
                    fname = val_batch[6]
                    slice_num = val_batch[7]
                    sequence = val_batch[8]

                    output = model(image)
                    val_loss = loss(output, target)

                    # END VALIDATION STEP
                    val_epoch.set_postfix(val_loss=val_loss)
                    m.end_val_step(fname, slice_num, sequence, output.to('cpu'), target.to('cpu'), val_loss.to('cpu'))

        # END EPOCH
        m.end_epoch(model, optimizer)


if __name__ == '__main__':
    train_()
    print('Done!')