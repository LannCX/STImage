import os
import sys
import json
sys.path.append('../')
from data.base_dataset import BaseDataset


class DTDBDataset(BaseDataset):
    def __init__(self, cfg, split='train'):
        super(DTDBDataset, self).__init__(cfg, split)
        trn_root = '/data/yangfeng/DTDB/BY_%s_FINAL/TRAIN' % self.version
        tst_root = '/data/yangfeng/DTDB/BY_%s_FINAL/TEST' % self.version
        data_root = trn_root if split=='train' else tst_root
        # data_root = tst_root if split == 'train' else trn_root
        class_file = cfg.DATASET.CLS_FILE % self.version
        self.name = 'DTDB-'+self.version
        self.class_2_id = json.load(open(class_file))
        self.id_2_class = {v: k for k, v in self.class_2_id.items()}

        for root, dirs, files in os.walk(data_root):
            for file in files:
                if 'Chaotic_motion_g3_c42.mp4'==file:
                    continue
                vid_id = file.split('.')[0]
                label = root.split('/')[-1]
                vid_path = os.path.join(root, file)
                if os.path.exists(vid_path):
                    self.anno.append({'id': vid_id, 'path': vid_path, 'label': label})

        print('Creating %s dataset completed, %d samples.' % (self.split, len(self.anno)))


import argparse
def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')

    parser.add_argument('--cfg', default='',
                        help='experiment configure file name',
                        # required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    from tqdm import tqdm
    from cfg.coconv_cfg import CoConvConfig as Config
    from data.data_loader import customer_data_loader
    args = parse_args()
    cfg = Config(args).getcfg()
    d_loader = customer_data_loader(cfg, cfg.DATASET.TRAIN_SET)
    nbar = tqdm(total=len(d_loader))
    for item in d_loader:
        nbar.update(1)
    nbar.close()
