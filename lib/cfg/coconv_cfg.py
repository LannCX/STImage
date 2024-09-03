import os
from cfg import BaseConfig


class CoConvConfig(BaseConfig):
    def __init__(self, args=None):
        super(CoConvConfig, self).__init__(args=args)

        self.cfg.USE_SSL = False
        self.cfg.LOSS.LAMDA = 1.

        self.cfg.DATASET.ANNO_FILE=''
        self.cfg.DATASET.CLS_FILE = ''
        self.cfg.DATASET.K = 4
        self.cfg.DATASET.P_SHUFFLE = False
        self.cfg.DATASET.T_SIZE = 16

        if args is not None:
            self.update_config(args)
