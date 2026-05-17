import os
import argparse
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset_all import TrainLabeled, TrainUnlabeled, ValLabeled
from trainer import Trainer
from utils import count_parameters, create_emamodel, setup_seed
from model.RaLiFormer import RaLiFormer


def main(gpu, args):
    args.local_rank = gpu

    setup_seed(2026)

    train_folder = args.data_dir
    paired_dataset = TrainLabeled(dataroot=train_folder, phase="labeled", finesize=args.crop_size)
    unpaired_dataset = TrainUnlabeled(
        dataroot=train_folder, phase="unlabeled", finesize=args.crop_size
    )
    val_dataset = ValLabeled(dataroot=train_folder, phase="val1", finesize=args.crop_size)
    paired_sampler = None
    unpaired_sampler = None
    val_sampler = None
    paired_loader = DataLoader(
        paired_dataset,
        batch_size=args.train_batchsize,
        sampler=paired_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    unpaired_loader = DataLoader(
        unpaired_dataset,
        batch_size=args.train_batchsize,
        sampler=unpaired_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.val_batchsize,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    print("there are total %s batches for train" % len(paired_loader))
    print("there are total %s batches for val" % len(val_loader))

    net = RaLiFormer()
    ema_net = RaLiFormer()
    ema_net = create_emamodel(ema_net)
    print(net)
    print("student model params: %d" % count_parameters(net))

    writer = SummaryWriter(log_dir=args.log_dir)
    trainer = Trainer(
        model=net,
        tmodel=ema_net,
        args=args,
        supervised_loader=paired_loader,
        unsupervised_loader=unpaired_loader,
        val_loader=val_loader,
        iter_per_epoch=len(unpaired_loader),
        writer=writer,
    )

    trainer.train()
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("-g", "--gpus", default=1, type=int, metavar="N")
    parser.add_argument("--num_epochs", default=40, type=int)
    parser.add_argument("--train_batchsize", default=4, type=int, help="train batchsize")
    parser.add_argument("--val_batchsize", default=2, type=int, help="val batchsize")
    parser.add_argument("--crop_size", default=512, type=int, help="crop size")
    parser.add_argument(
        "--num_workers", default=4, type=int, help="number of workers for data loader"
    )
    parser.add_argument("--resume", default="True", type=str, help="if resume")
    parser.add_argument(
        "--resume_path",
        default="./experiments/model_RaLiFormer1/ckpt/model_e5.pth",
        type=str,
        help="if resume",
    )
    parser.add_argument("--use_pretain", default="True", type=str, help="use pretrained model")
    parser.add_argument(
        "--pretrained_path", default="./pretained/net.pth", type=str, help="if pretrained"
    )
    parser.add_argument("--data_dir", default="/mnt/zxy/the_next_work/Semi-UIR/data", type=str, help="data root path")
    parser.add_argument("--save_path", default="./experiments/model_RaLiFormer1/ckpt/", type=str)
    parser.add_argument("--log_dir", default="./experiments/model_RaLiFormer1/log", type=str)

    args = parser.parse_args()
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    main(-1, args)

# CUDA_VISIBLE_DEVICES=1 nohup python -u train.py  > output1.log 2>&1 &
