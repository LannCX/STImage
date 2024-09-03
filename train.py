import os
import pprint
import argparse

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torch.backends.cudnn as cudnn

import init_path
from utils.logger import Logger
from data.data_loader import customer_data_loader
from arch.net_loader import customer_net_loader

# **************************************************************************
# Replace by your custom class
# **************************************************************************
from scheduler.coconv_scheduler import CoConvScheduler as Scheduler
from cfg.coconv_cfg import CoConvConfig as Config
# **************************************************************************


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')

    parser.add_argument('--cfg', default='experiments/something-something.yaml',
                        help='experiment configure file name',
                        # required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    parser.add_argument('--resume',
                        help='which epoch to resume',
                        type=int,
                        default=None)

    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')

    parser.add_argument('--modelDir',
                        help='output model directory',
                        type=str,
                        default='')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    cfg = Config(args).getcfg()
    exp_suffix = os.path.basename(args.cfg).split('.')[0]
    logger = Logger(os.path.join(cfg.LOG_DIR, '.'.join([cfg.SNAPSHOT_PREF, cfg.MODEL.NAME, exp_suffix, cfg.TRAIN.OPTIMIZER, str(cfg.TRAIN.LR)])))

    logger.log(pprint.pformat(args))
    logger.log(cfg)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    # define
    train_loader = customer_data_loader(cfg, cfg.DATASET.TRAIN_SET)
    val_loader = customer_data_loader(cfg, cfg.DATASET.TEST_SET)
    net = customer_net_loader(cfg=cfg)
    model = Scheduler(net, cfg=cfg, logger=logger)

    if cfg.IS_TRAIN:
        model.train(train_loader, val_loader=val_loader, which_epoch=args.resume)
    else:
        model.test(val_loader, weight_file=cfg.TEST.LOAD_WEIGHT)


if __name__ == '__main__':
    main()
