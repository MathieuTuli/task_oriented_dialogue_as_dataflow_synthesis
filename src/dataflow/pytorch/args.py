"""
Author: Mathieu Tuli
GitHub: @MathieuTuli
Email: tuli.mathieu@gmail.com
"""
from argparser import ArgumentParser


def train_args(parser: ArgumentParser):
    parser.add_argument("-verbose", action="store_true", default=False)
    parser.add_argument(
        '--config', dest='config',
        default='config.yaml', type=str,
        help="Set configuration file path: Default = 'config.yaml'")
    parser.add_argument(
        '--data', dest='data',
        default='data', type=str,
        help="Set data directory path: Default = 'data'")
    parser.add_argument(
        '--output', dest='train-output',
        default='output', type=str,
        help="Set output directory path: Default = 'train-output'")
    parser.add_argument(
        '--checkpoint', dest='checkpoint',
        default='checkpoint', type=str,
        help="Set checkpoint directory path: Default = 'checkpoint'")
    sub_parser.add_argument(
        '--save-freq', default=25, type=int,
        help='Checkpoint epoch save frequency: Default = 25')
    parser.add_argument(
        '--cpu', action='store_true',
        dest='cpu',
        help="Flag: CPU bound training: Default = False")
    parser.set_defaults(cpu=False)
    parser.add_argument(
        '--gpu', default=0, type=int,
        help='GPU id to use: Default = 0')
    parser.add_argument(
        '--multiprocessing-distributed', action='store_true',
        dest='mpd',
        help='Use multi-processing distributed training to launch '
        'N processes per node, which has N GPUs. This is the '
        'fastest way to use PyTorch for either single node or '
        'multi node data parallel training: Default = False')
    parser.set_defaults(mpd=False)
    parser.add_argument(
        '--dist-url', default='tcp://127.0.0.1:23456', type=str,
        help="url used to set up distributed training:" +
             "Default = 'tcp://127.0.0.1:23456'")
    parser.add_argument(
        '--dist-backend', default='nccl', type=str,
        help="distributed backend: Default = 'nccl'")
    parser.add_argument(
        '--world-size', default=-1, type=int,
        help='Number of nodes for distributed training: Default = -1')
    parser.add_argument(
        '--rank', default=-1, type=int,
        help='Node rank for distributed training: Default = -1')
