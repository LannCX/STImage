import os
import matplotlib.pyplot as plt
from numpy.random import randint
from torch.utils.data import Dataset

from decord import VideoReader
from data.video_transforms import *
import torchvision.transforms as transf
from utils.nn_utils import video_to_stimg


class BaseDataset(Dataset):
    def __init__(self, cfg, split):
        self.split = split if split == 'train' else 'test'
        self.cfg = cfg
        self.k = cfg.DATASET.K
        self.eval = not cfg.IS_TRAIN
        self.version = cfg.DATASET.VERSION
        self.t_size = cfg.DATASET.T_SIZE
        self.scale_size = cfg.DATASET.SCALE_SIZE
        self.crop_size = cfg.DATASET.CROP_SIZE
        self.data_type = cfg.DATASET.DATA_TYPE
        self.image_tmpl = cfg.DATASET.IMG_FORMAT

        self.anno = []
        self.id_2_class = {}
        self.class_2_id = {}
        self.name = 'Base Dataset'

        # Transformation
        normalize = IdentityTransform() if 'Inception' in cfg.MODEL.NAME else GroupNormalize(cfg.DATASET.MEAN, cfg.DATASET.STD)
        self.roll = True if 'Inception' in cfg.MODEL.NAME else False
        self.div = not self.roll
        self.is_flip = False if 'something' in cfg.DATASET.NAME else True
        if self.split == 'train':
            if self.is_flip:
                self.transform = transf.Compose([
                    GroupMultiScaleCrop(self.crop_size, [1, .875, .75, .66], fix_crop=True,
                                        more_fix_crop=True),
                    GroupRandomHorizontalFlip(),
                    Stack(self.roll),
                    ToTorchFormatTensor(div=self.div),
                    normalize
                ])
            else:
                self.transform = transf.Compose([
                    GroupMultiScaleCrop(self.crop_size, [1, .875, .75, .66], fix_crop=True,
                                        more_fix_crop=True),
                    Stack(self.roll),
                    ToTorchFormatTensor(div=self.div),
                    normalize
                ])
        else:
            if self.eval:
                self.transform = transf.Compose([
                    GroupFullResSample(self.crop_size, self.scale_size, flip=False),
                    Stack(self.roll),
                    ToTorchFormatTensor(div=self.div),
                    normalize
                ])
            else:
                self.transform = transf.Compose([
                    GroupScale(self.scale_size),
                    GroupCenterCrop(self.crop_size),
                    Stack(self.roll),
                    ToTorchFormatTensor(div=self.div),
                    normalize
                ])

    def _load_image(self, directory, idx):
        return Image.open(os.path.join(directory, self.image_tmpl.format(idx))).convert('RGB')

    @staticmethod
    def _sample_indices(num_frames, need_f):
        if num_frames<=need_f:
            indices = list(range(1, num_frames+1))
            for i in range(need_f-num_frames):
                indices.append(num_frames)
        else:
            start = random.randint(1, num_frames - need_f+1)
            indices = list(range(start, start + need_f))
        return indices

    @staticmethod
    def _get_val_indices(num_frames, need_f):
        if num_frames <= need_f:
            indices = list(range(1, num_frames + 1))
            for i in range(need_f - num_frames):
                indices.append(num_frames)
        else:
            start = (num_frames-need_f)//2
            indices = list(range(start, start + need_f))
        return indices

    @staticmethod
    def _get_test_indices(num_frames, need_f):
        indices = []
        if num_frames <= need_f:
            indices = list(range(1, num_frames + 1))
            for i in range(need_f - num_frames):
                indices.append(num_frames)
        else:
            for start in range(1, num_frames-need_f+1, 8):
                indices.extend(list(range(start, start+need_f)))
        return indices

    def get_indices(self, nf, need_f):
        if self.eval:
            indices = self._get_test_indices(nf, need_f)
        else:
            indices = self._sample_indices(nf, need_f) if self.split == 'train' \
                else self._get_val_indices(nf, need_f)
        return indices

    def __getitem__(self, index):
        vid_info = self.anno[index]
        need_f = self.t_size ** 2
        if self.data_type=='video':
            vr = VideoReader(vid_info['path'])
            nf = len(vr)
            indices = [x-1 for x in self.get_indices(nf, need_f)]
            img_group = vr.get_batch(indices).asnumpy()
            img_group = [Image.fromarray(img) for img in img_group]
        elif self.data_type=='img':
            nf = vid_info['nf']
            indices = self.get_indices(nf, need_f)
            img_group = [self._load_image(vid_info['path'], int(ind)) for ind in indices]
        else:
            raise KeyError('Not supported data type: {}.'.format(self.data_type))

        img_tensor = self.transform(img_group)  # [T,C,H,W]
        if self.eval:
            vid_data = img_tensor.reshape((-1, need_f)+img_tensor.shape[-3:])
            out_img = [video_to_stimg(data, self.t_size) for data in vid_data]
            img_tensor = torch.stack(out_img, dim=0)  # N clips x C x H x W
        else:
            img_tensor = video_to_stimg(img_tensor, self.t_size)
            pass
        # img_tensor = img_tensor/255. * 2 - 1.  # Inception normalization

        return img_tensor, self.class_2_id[vid_info['label']]

    def __len__(self):
        return len(self.anno)

    def __name__(self):
        return self.name
