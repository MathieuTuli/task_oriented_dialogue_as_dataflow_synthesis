"""
Author: Mathieu Tuli
GitHub: @MathieuTuli
Email: tuli.mathieu@gmail.com
"""
from argparse import ArgumentParser, APNamespace
from typing import Tuple, Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path

import time

from onmt.utils.logging import init_logger, logger

from torch.optim.lr_scheduler import StepLR, CosineAnnealingWarmRestarts
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.distributed as dist
import pandas as pd
import numpy as np
import torch
import yaml

from dataflow.pytorch.optim import get_optimizer_scheduler
from dataflow.pytorch.models import get_model
from dataflow.pytorch.args import train_args
from dataflow.pytorch.data import get_data

parser = ArgumentParser(description="train.py")
train_args(parser)


class TrainingAgent:
    config: Dict[str, Any] = None
    train_loader = None
    val_loader = None
    train_sampler = None
    num_classes: int = None
    network: torch.nn.Module = None
    optimizer: torch.optim.Optimizer = None
    scheduler = None
    loss = None
    output_filename: Path = None
    checkpoint = None

    def __init__(
            self,
            config_path: Path,
            device: str,
            output_path: Path,
            data_path: Path,
            checkpoint_path: Path,
            gpu: int = None,
            ngpus_per_node: int = 0,
            world_size: int = -1,
            rank: int = -1,
            dist: bool = False,
            mpd: bool = False,
            dist_url: str = None,
            dist_backend: str = None) -> None:

        self.gpu = gpu
        self.mpd = mpd
        self.dist = dist
        self.rank = rank
        self.best_acc = 0.
        self.start_epoch = 0
        self.start_trial = 0
        self.device = device
        self.dist_url = dist_url
        self.world_size = world_size
        self.dist_backend = dist_backend
        self.ngpus_per_node = ngpus_per_node
        self.date = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

        self.data_path = data_path
        self.output_path = output_path
        self.checkpoint_path = checkpoint_path

        self.load_config(config_path, data_path)

    def load_config(self, config_path: Path, data_path: Path) -> None:
        with config_path.open() as f:
            self.config = config = yaml.load(f)
        if self.device == 'cpu':
            logger.warning("Using CPU will be slow")
        elif self.dist:
            if self.gpu is not None:
                config['mini_batch_size'] = int(
                    config['mini_batch_size'] / self.ngpus_per_node)
                config['num_workers'] = int(
                    (config['num_workers'] + self.ngpus_per_node - 1) /
                    self.ngpus_per_node)
        self.train_loader, self.train_sampler,\
            self.val_loader, self.num_classes = get_data(
                root=data_path, mini_batch_size=config['mini_batch_size'],
                num_workers=config['num_workers'], dist=self.dist)
        self.criterion = torch.nn.CrossEntropyLoss().cuda(self.gpu) if \
            config['loss'] == 'cross_entropy' else None
        cudnn.benchmark = True

    def reset(self, config: Optional[Dict[str, Any]] = None) -> None:
        if config is None:
            config = self.config
        self.performance_statistics = dict()
        self.model = get_model(name=config['network'],
                               num_classes=self.num_classes)
        # TODO add other parallelisms
        if self.device == 'cpu':
            ...
        elif self.dist:
            if self.gpu is not None:
                torch.cuda.set_device(self.gpu)
                self.model.cuda(self.gpu)
                self.model = torch.nn.parallel.DistributedDataParallel(
                    self.model,
                    device_ids=[self.gpu])
            else:
                self.model.cuda()
                self.model = torch.nn.parallel.DistributedDataParallel(
                    self.model)
        elif self.gpu is not None:
            torch.cuda.set_device(self.gpu)
            self.model = self.model.cuda(self.gpu)
        else:
            self.model = torch.nn.DataParallel(self.model)
        self.optimizer, self.scheduler = get_optimizer_scheduler(
            optim_method=config['optimizer'],
            lr_scheduler=config['scheduler'],
            init_lr=config['init_lr'],
            net_parameters=self.model.parameters(),
            listed_params=list(self.model.parameters()),
            train_loader_len=len(self.train_loader),
            max_epochs=config['max_epochs'],
            optimizer_kwargs=config['optimizer_kwargs'],
            scheduler_kwargs=config['scheduler_kwargs'])
        self.early_stop.reset()
        self.best_acc = -1.

    def train(self) -> None:
        for trial in range(self.start_trial,
                           self.config['n_trials']):
            self.reset(self.config)
            epochs = range(0, self.config['max_epochs'])
            self.output_filename = \
                "train_results_" +\
                f"date={self.date}_" +\
                f"trial={trial}_" +\
                f"network={self.config['network']}_" +\
                f"dataset={self.config['dataset']}_" +\
                f"optimizer={self.config['optimizer']}_" +\
                '_'.join([f"{k}={v}" for k, v in
                          self.config['optimizer_kwargs'].items()]) +\
                f"_scheduler={self.config['scheduler']}_" +\
                '_'.join([f"{k}={v}" for k, v in
                          self.config['scheduler_kwargs'].items()]) +\
                f"_lr={self.config['init_lr']}" +\
                ".csv".replace(' ', '-')
            self.output_filename = str(
                self.output_path / self.output_filename)
            self.run_epochs(trial, epochs)

    def run_epochs(self, trial: int, epochs: List[int]) -> None:
        data = {}
        for epoch in epochs:
            if self.dist:
                self.train_sampler.set_epoch(epoch)
            start_time = time.time()
            train_loss, train_joint_ba = self.epoch_iteration(
                trial, epoch)
            val_loss, val_joint_ba = self.validate(epoch)
            end_time = time.time()
            if isinstance(self.scheduler, StepLR):
                self.scheduler.step()
            total_time = time.time()
            logger.info(
                f"T {trial + 1}/{self.config['n_trials']} | " +
                f"E {epoch + 1}/{epochs[-1] + 1} Ended | " +
                "E Time: {:.3f}s | ".format(end_time - start_time) +
                "~Time Left: {:.3f}s | ".format(
                    (total_time - start_time) * (epochs[-1] - epoch)),
                "Train Loss: {:.4f}% | ".format(
                    train_loss) +
                "Test Loss: {:.4f}% | ".format(
                    val_loss))
            df = pd.DataFrame(data=self.performance_statistics)
            df.to_csv(self.output_filename)
            if not self.mpd or \
                    (self.mpd and self.rank % self.ngpus_per_node == 0):
                data = {'epoch': epoch + 1,
                        'trial': trial,
                        'config': self.config,
                        'state_dict_network': self.model.state_dict(),
                        'state_dict_optimizer': self.optimizer.state_dict(),
                        'state_dict_scheduler': self.scheduler.state_dict(),
                        'best_acc': self.best_acc,
                        'performance_statistics': self.performance_statistics,
                        'output_filename': Path(self.output_filename).name,
                        'historical_metrics': self.metrics.historical_metrics}
                if epoch % self.save_freq == 0:
                    filename = f'trial_{trial}.pth.tar'
                    torch.save(data, str(self.checkpoint_path / filename))
                if np.greater(100., self.best_acc):
                    self.best_acc = 100.
                    torch.save(
                        data, str(self.checkpoint_path / 'best.pth.tar'))
        torch.save(data, str(self.checkpoint_path / 'last.pth.tar'))

    def epoch_iteration(self, trial: int, epoch: int):
        self.model.train()
        train_loss = 0
        joint_ba = AverageMeter()

        """train"""
        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            if self.gpu is not None:
                inputs = inputs.cuda(self.gpu, non_blocking=True)
            if self.device == 'cuda':
                targets = targets.cuda(self.gpu, non_blocking=True)
            if isinstance(self.scheduler, CosineAnnealingWarmRestarts):
                self.scheduler.step(epoch + batch_idx / len(self.train_loader))
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            acc = accuracy(outputs, targets, (1, 5))
            joint_ba.update(acc, inputs.size(0))
        self.performance_statistics[f'train_joint_ba_epoch_{epoch}'] = \
            joint_ba.avg.cpu().item() / 100.
        self.performance_statistics[f'train_loss_epoch_{epoch}'] = \
            train_loss / (batch_idx + 1)
        self.performance_statistics[
            f'learning_rate_epoch_{epoch}'] = \
            self.optimizer.param_groups[0]['lr']
        return train_loss / (batch_idx + 1), joint_ba.avg.cpu().item() / 100

    def validate(self, epoch: int):
        self.model.eval()
        val_loss = 0
        joint_ba = AverageMeter()
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.val_loader):
                if self.gpu is not None:
                    inputs = inputs.cuda(self.gpu, non_blocking=True)
                if self.device == 'cuda':
                    targets = targets.cuda(self.gpu, non_blocking=True)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                val_loss += loss.item()
                acc = accuracy(outputs, targets, topk=(1, 5))
                joint_ba.update(acc, inputs.size(0))

        self.performance_statistics[f'val_joint_ba_epoch_{epoch}'] = (
            joint_ba.avg.cpu().item() / 100.)
        self.performance_statistics[f'val_loss_epoch_{epoch}'] = val_loss / (
            batch_idx + 1)
        return val_loss / (batch_idx + 1), joint_ba.avg.cpu().item() / 100


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(outputs: torch.Tensor, targets: torch.Tensor):
    with torch.no_grad():
        return


def setup_dirs(args: APNamespace) -> Tuple[Path, Path, Path, Path]:
    config_path = Path(args.config).expanduser()
    data_path = Path(args.data).expanduser()
    output_path = Path(args.output).expanduser()
    checkpoint_path = Path(args.checkpoint).expanduser()

    if not config_path.exists():
        raise ValueError(
            f"Config path {config_path} does not exist")
    if not data_path.exists():
        logger.info(f"Data dir {data_path} does not exist, building")
        data_path.mkdir(exist_ok=True, parents=True)
    if not output_path.exists():
        logger.info(f"Output dir {output_path} does not exist, building")
        output_path.mkdir(exist_ok=True, parents=True)
    if not checkpoint_path.exists():
        checkpoint_path.mkdir(exist_ok=True, parents=True)
    if args.resume is not None:
        if not Path(args.resume).exists():
            raise ValueError("Resume path does not exist")
    return config_path, output_path, data_path, checkpoint_path,\
        Path(args.resume) if args.resume is not None else None


def main(args: APNamespace):
    args.config_path, args.output_path, \
        args.data_path, args.checkpoint_path, \
        args.resume = setup_dirs(args)
    ngpus_per_node = torch.cuda.device_count()
    args.distributed = args.mpd or args.world_size > 1
    if args.mpd:
        args.world_size *= ngpus_per_node
        mp.spawn(main_worker, nprocs=ngpus_per_node,
                 args=(ngpus_per_node, args))
    else:
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu: int, ngpus_per_node: int, args: APNamespace):
    args.gpu = gpu
    if args.distributed:
        if args.mpd:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(
            backend=args.dist_backend, init_method=args.dist_url,
            world_size=args.world_size, rank=args.rank)
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    training_agent = TrainingAgent(
        config_path=args.config_path,
        device=device,
        output_path=args.output_path,
        data_path=args.data_path,
        checkpoint_path=args.checkpoint_path,
        resume=args.resume,
        save_freq=args.save_freq,
        gpu=args.gpu,
        ngpus_per_node=ngpus_per_node,
        world_size=args.world_size,
        rank=args.rank,
        dist=args.distributed,
        mpd=args.mpd,
        dist_url=args.dist_url,
        dist_backend=args.dist_backend)
    logger.info(f"Pytorch device is set to {training_agent.device}")
    training_agent.train()


if __name__ == "__main__":
    init_logger("train_pytorch.log")
    args = parser.parse_args()
    main(args)
