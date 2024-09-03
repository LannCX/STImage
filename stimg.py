import os
import copy
import time
import matplotlib.pyplot as plt
from decord import VideoReader

from utils import mkdir
from utils.cv_utils import uniform_temporal_sampling
from data.video_transforms import *


class STImage(object):
    def __init__(self, vid_path, input_size=224, k=1, t_size=8, mode='video',
                 transform=None, patch_shuffle=False, rand_sel=False,
                 random_rm=False, random_flip=False, random_fps=False, random_crop=False):
        self.stimg_size = t_size
        self.vid_path = vid_path
        self.image_tmpl = '{:05d}.jpg'
        if isinstance(input_size, int):
            self.input_size = input_size
        elif isinstance(input_size, list) or isinstance(input_size, tuple):
            self.input_size = input_size[0]
        else:
            raise(TypeError, 'Wrong image size type!')

        # Load frames
        if mode == 'video':
            vr = VideoReader(vid_path)
            self.__fr_cnt = len(vr)
        else:
            self.__fr_cnt = len(os.listdir(vid_path))

        # Resize to t_size
        # Random selection if fr_cnt>2*size**2
        tmp = uniform_temporal_sampling(self.stimg_size ** 2, list(range(self.fr_cnt)), rand_sel=rand_sel)
        # tmp = uniform_temporal_sampling(self.stimg_size ** 2, list(range(self.fr_cnt)), random_fps)
        self.t_img = np.array(tmp).reshape(t_size, t_size)

        # Load frames
        indices = [x for x in self.t_img.reshape((-1)).tolist()]
        if mode == 'video':
            img_group = vr.get_batch(indices).asnumpy()
            img_group = [Image.fromarray(img) for img in img_group]
        else:
            img_group = [self._load_image(vid_path, int(ind)+1) for ind in indices]

        # Image data augmentation
        if transform is not None:
            self.vid_data = transform(img_group)
        else:
            self.vid_data = img_group
        assert len(self.vid_data) > 0

    def _load_image(self, directory, idx):
        return Image.open(os.path.join(directory, self.image_tmpl.format(idx))).convert('RGB')

    def construct(self, vis=False, is_save=False):
        t, c, h, w = self.vid_data.shape

        # image as pixel
        inv = self.input_size

        # pixels = self.vid_data.as_strided(
        #     (c, self.stimg_size, inv, self.stimg_size, inv),
        #     (inv**2, c*h*w*self.stimg_size, inv, c*h*w, 1)
        # ).contiguous()
        # pixels = pixels.permute(0, 2, 1, 4, 3).contiguous()
        # out_img = pixels.as_strided(
        #     (c, inv*self.stimg_size, inv*self.stimg_size),
        #     (self.stimg_size**2*inv**2, self.stimg_size*inv, 1)
        # ).contiguous()
        out_img = self.vid_data.permute(1, 2, 3, 0).reshape(c, h, -1)

        if vis:
            plt.figure('patches after stitching')
            plt.imshow(Image.fromarray(out_img[0:3, :, :].permute(1, 2, 0).numpy().astype(np.uint8)))
            plt.show()

        if is_save:
            save_dir = os.path.join('./out_images', self.vid_path.split('/')[-1].split('.')[0]+'.bmp')
            # mkdir(save_dir)
            img = Image.fromarray(out_img.permute(1, 2, 0).numpy().astype(np.uint8))
            img.save(save_dir)

        return out_img

    def decompose(self, x, vis=False):
        c, h, w = x.shape
        assert h == w
        inv = h // self.stimg_size
        grids = x.as_strided(
            (self.stimg_size, self.stimg_size, c, inv, inv),
            (w, 1, h * w, self.stimg_size * w, self.stimg_size)
        ).contiguous()
        grids = grids.as_strided(
            (self.stimg_size ** 2, c, inv, inv),
            (c * inv ** 2, inv ** 2, inv, 1)
        ).contiguous()

        if vis:
            plt.figure('reconstructed video')
            plt.imshow(Image.fromarray(grids[1, ...].permute(1, 2, 0).numpy().astype(np.uint8)))
            plt.show()

        return grids

    @property
    def fr_cnt(self):
        return self.__fr_cnt


if __name__ == '__main__':
    tic = time.time()
    v_path = '/raid/data/something-something/v1/imgs/31199'
    transf = torchvision.transforms.Compose([
        GroupScale(128),
        GroupCenterCrop(112),
        Stack(),
        ToTorchFormatTensor(div=False)
    ])
    # transf = None
    IMG = STImage(v_path, t_size=4, input_size=112, transform=transf, mode='img')
    img_o = IMG.construct(vis=True)
    # IMG.decompose(img_o, vis=True)
    print(time.time()-tic)
