import os
import copy
import matplotlib.pyplot as plt
from decord import VideoReader

from utils import mkdir
from data.video_transforms import *


class STImage(object):
    def __init__(self, vid_path, input_size=224, t_size=8,
                 transform=None, patch_shuffle=False, rand_sel=False,
                 random_rm=False, random_flip=False, random_fps=False, random_crop=False):
        self.stimg_size = t_size
        self.vid_path = vid_path
        self.p_shuffle = patch_shuffle
        self.random_rm = random_rm
        self.random_fps = random_fps
        self.random_flip = random_flip
        self.random_crop = random_crop

        if isinstance(input_size, int):
            self.input_size = input_size
        elif isinstance(input_size, list) or isinstance(input_size, tuple):
            self.input_size = input_size[0]
        else:
            raise(TypeError, 'Wrong image size type!')

        # Load frames
        vr = VideoReader(vid_path)
        self.__fr_cnt = len(vr)

        # Resize to t_size
        # Random selection if fr_cnt>2*size**2
        tmp = uniform_temporal_sampling(self.stimg_size ** 2, list(range(self.fr_cnt)), rand_sel=rand_sel)
        # tmp = uniform_temporal_sampling(self.stimg_size ** 2, list(range(self.fr_cnt)), random_fps)
        self.t_img = np.array(tmp).reshape(t_size, t_size)

        # Augmentations
        if self.random_flip:
            self.t_img = self.t_img[:, ::-1]

        # Load frames
        indices = [x for x in self.t_img.reshape((-1)).tolist()]
        img_group = vr.get_batch(indices).asnumpy()
        img_group = [Image.fromarray(img) for img in img_group]

        # Image data augmentation
        if transform is not None:
            self.vid_data = transform(img_group)
        else:
            self.vid_data = img_group

    def construct(self, vis=False, is_save=False):
        _, c, h, w = self.vid_data.shape

        # image as pixel
        inv = self.input_size//self.k_grid
        patch_ids = list(range(self.k_grid**2))
        mask = torch.zeros(c*self.k_grid**2, inv*self.stimg_size, inv*self.stimg_size)

        if self.p_shuffle:
            random.shuffle(patch_ids)
        pixels = self.vid_data.as_strided(
            (_, self.k_grid, self.k_grid, c, inv, inv),
            (c*h*w, inv*w, inv, h*w, w, 1)
        ).contiguous()
        pixels = pixels.as_strided(
            (_, c*self.k_grid**2, inv, inv),
            (c*h*w, inv*inv, inv, 1)
        ).contiguous()
        inds = np.arange(c*self.k_grid**2).reshape((-1, c))
        pixels = pixels[:, inds[patch_ids, :].reshape(-1), ...]

        if self.random_rm:
            idx = random.randint(0, _-2)
            row, col = idx//self.stimg_size, idx%self.stimg_size
            mask[:, row*inv:row*inv+inv, col*inv:col*inv+inv] = 1.
        pixels = pixels.as_strided(
            (c * self.k_grid ** 2, self.stimg_size, inv, self.stimg_size, inv),
            (inv**2, c*h*w*self.stimg_size, inv, c*h*w, 1)
        ).contiguous()
        out_img = pixels.as_strided(
            (c*self.k_grid**2, inv*self.stimg_size, inv*self.stimg_size),
            (self.stimg_size**2*inv**2, self.stimg_size*inv, 1)
        ).contiguous()

        if vis:
            plt.figure('patches after stitching')
            plt.imshow(Image.fromarray(out_img[0:3, :, :].permute(1, 2, 0).numpy().astype(np.uint8)))
            plt.figure('removed patch')
            plt.imshow(Image.fromarray((out_img[0:3, :, :]*(1-mask)).permute(1, 2, 0).numpy().astype(np.uint8)))
            plt.show()

        if is_save:
            save_dir = os.path.join('./out_images', self.vid_path.split('/')[-1].split('.')[0])
            mkdir(save_dir)
            for i in range(self.k_grid**2):
                img = Image.fromarray(out_img[3*i:3*i+3, :, :].permute(1, 2, 0).numpy().astype(np.uint8))
                img.save('%s/k_%d.jpg' % (save_dir, i))

        return out_img, np.array(patch_ids), mask

    @property
    def fr_cnt(self):
        return self.__fr_cnt

    @property
    def shape(self):
        return self.input_size//self.k_grid, self.input_size//self.k_grid, self.stimg_size, self.stimg_size

    def __repr__(self):
        return '<'+self.vid_path+', size: '+'('+\
               ','.join([str(self.shape[0]), str(self.shape[1]), str(self.shape[2]), str(self.shape[3])])+')'+'>'


if __name__ == '__main__':
    tic = time.time()
    v_path = '/data_ssd/chenxu/workspace/Kinetics-400/train/catching_fish/9MphJyd0244_000090_000100.mp4'
    transf = torchvision.transforms.Compose([
        GroupScale(128),
        GroupRandomCrop(112),
        GroupRandomHorizontalFlip(),
        Stack(),
        ToTorchFormatTensor(div=False)
    ])
    # transf = None
    IMG = STImage(v_path, t_size=8, input_size=112, transform=transf, random_rm=True)
    IMG.construct(vis=True)
    print(time.time()-tic)
