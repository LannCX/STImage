import math
import time
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F


class S4D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3,3,3,3), stride=1,
                 padding=0, dilation=8, groups=1, bias=False,
                 tau=1, alpha=2, beta=4):
        super(S4D, self).__init__()
        k_s = kernel_size[2:]
        self.k_n, self.k_t = kernel_size[0], kernel_size[1]
        self.d = dilation
        self.d_list = [alpha*tau, tau, beta*tau]
        for i in range(3,self.k_n):
            self.d_list = [self.d_list[0]].extend(self.d_list) if i%2==0 else self.d_list.append(self.d_list[-1])
        assert len(self.d_list)==self.k_n
        self.s_ker = nn.Conv2d(in_channels,
                                out_channels,
                                kernel_size=k_s,
                                stride=stride,
                                padding=k_s[0]//2,
                                groups=groups,
                                bias=False)

        self.t_kers = nn.ModuleList([
            nn.Conv3d(out_channels, out_channels, (self.k_t,1,1), dilation=(d, 1, 1))
            for d in self.d_list
        ])
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels), requires_grad=True)
        else:
            self.bias = None
        self.reset_parameters()

    def reset_parameters(self):
        # for weight in self.weights:
        #     init.kaiming_uniform_(weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.s_ker.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        b,c,t,h,w = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(-1, c, h, w)
        x = self.s_ker(x)
        _, c_o, h_o, w_o = x.shape
        x = x.reshape(b, t, c_o, h_o, w_o).permute(0,2,1,3,4)
        # Padding
        x_s = F.pad(x, [0, 0, 0, 0, self.k_n // 2 * self.d,
                          self.k_n // 2 * self.d + (self.k_t-1) * self.d_list[-1]])

        for k, ker in enumerate(self.t_kers):
            inds = [i for i in range(k*self.d,
                                     x_s.shape[2]-self.d*(self.k_n-1-k)-(self.k_t-1)*self.d_list[-1]+(self.k_t-1)*self.d_list[k])]
            temp = x_s[:,:,inds,...]
            x_o = ker(temp)
            if k==0:
                out = x_o
            else:
                out += x_o

        if self.bias is not None:
            out += self.bias.repeat(self.t_size*h_o, self.t_size*w_o, 1).permute(2, 0, 1)

        return out


class CoCov(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size_h=1,
                 kernel_size_l=3, t_size=16, groups=1, padding=0,
                 stride=1, bias=False, name='Cocov'):
        super(CoCov, self).__init__()
        self.p = padding
        self.t_size = t_size
        self.kernel_size_h = kernel_size_h
        self.name = name
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels), requires_grad=True)
        else:
            self.bias = None
        if kernel_size_l == 1 and kernel_size_h == 1:
            self.normal = True
            self.kernels = torch.nn.ModuleList([nn.Conv2d(in_channels,
                                                          out_channels,
                                                          kernel_size=1,
                                                          stride=stride,
                                                          groups=groups,
                                                          padding=self.p,
                                                          bias=bias)])
        else:
            self.normal = False
            self.kernels = torch.nn.ModuleList([nn.Conv2d(in_channels,
                                                          out_channels,
                                                          kernel_size_l,
                                                          stride=stride,
                                                          padding=kernel_size_l//2,
                                                          groups=groups,
                                                          bias=False) for _ in range(kernel_size_h**2)])
        self.reset_parameters()

    def reset_parameters(self):
        # for weight in self.weights:
        #     init.kaiming_uniform_(weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.kernels[0].weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, inp):
        # tic = time.time()
        if self.normal:
            return self.kernels[0](inp)

        b, c, h, w = inp.shape
        assert h == w
        inv = h//self.t_size
        inp = F.pad(inp, [self.p*inv, self.p*inv, self.p*inv, self.p*inv])
        size = self.t_size+2*self.p
        h += 2*self.p*inv
        w += 2*self.p*inv

        grids = inp.as_strided(
            (b, size, size, c, inv, inv),
            (c * h * w, inv * w, inv, h * w, w, 1)
        ).contiguous()
        grids = grids.as_strided(
            (b*size**2, c, inv, inv),
            (c*inv**2, inv**2, inv, 1)
        ).contiguous()

        h_o, w_o = 0, 0
        for k, ker in enumerate(self.kernels):
            row, col = k // self.kernel_size_h, k % self.kernel_size_h
            inds = [b_i*size**2+i*size+j for b_i in range(b) for i in range(row, size-self.kernel_size_h+1+row)
                    for j in range(col, size-self.kernel_size_h+1+col)]
            temp = grids[inds, :, :, :]
            assert len(inds)==self.t_size**2*b
            # t_h, t_w = size-row, size-col

            # start = time.time()
            x = ker(temp)
            # x = temp
            _, c_o, h_o, w_o = x.shape
            # x = x.reshape(b, self.t_size**2, c_o, h_o, w_o)
            # print('conv time:%.4f' % (time.time()-start))

            # tic = time.time()
            x_o = x.as_strided(
                (b, c_o, self.t_size, h_o, self.t_size, w_o),
                (c_o * h_o * w_o * self.t_size**2, h_o * w_o,
                 self.t_size * c_o * h_o * w_o, w_o, c_o * h_o * w_o, 1)
            ).contiguous()
            x_o = x_o.as_strided(
                (b, c_o, self.t_size*h_o, self.t_size*w_o),
                (c_o*h_o*w_o*self.t_size*self.t_size, self.t_size**2*h_o*w_o, self.t_size*w_o, 1)
            ).contiguous()
            # x_o = F.pad(x_o, [0, w_o*col, 0, h_o*row])
            # print('arran time:%.4f' % (time.time()-tic))
            if k==0:
                out = x_o
            else:
                out += x_o
            # print('Each Conv kernel:%.4f' % (time.time() - start))

        out_x = out

        if self.bias is not None:
            out_x += self.bias.repeat(self.t_size*h_o, self.t_size*w_o, 1).permute(2, 0, 1)

        return out_x


class Unit4d(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, size=8, groups=1):
        super(Unit4d, self).__init__()
        self.s_conv = CoCov(in_planes, out_planes, kernel_size_h=1, kernel_size_l=3,
                            t_size=size, stride=stride, groups=groups, bias=False)
        self.bn_s = nn.BatchNorm2d(out_planes, eps=1e-3, momentum=0.001, affine=True)
        self.relu_s = nn.ReLU()

        self.t_conv = CoCov(out_planes, out_planes, kernel_size_h=3, kernel_size_l=1,
                            t_size=size, stride=1, groups=groups, padding=1, bias=False)
        self.bn_t = nn.BatchNorm2d(out_planes, eps=1e-3, momentum=0.001, affine=True)
        self.relu_t = nn.ReLU()

    def forward(self, inp):
        x = self.s_conv(inp)
        x = self.bn_s(x)
        x = self.relu_s(x)

        x = self.t_conv(x)
        x = self.bn_t(x)
        x = self.relu_t(x)

        return x


class CoCovTranspose(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size_h=3, padding=1,
                 kernel_size_l=3, t_size=16, stride=2, s_mode=True, name='Cocov'):
        super(CoCovTranspose, self).__init__()
        self.p = padding
        self.name = name
        self.s_mode = s_mode
        self.t_size = t_size
        self.stride = stride
        self.kernel_size_h = kernel_size_h
        if self.s_mode:
            self.kernels = torch.nn.ModuleList([nn.ConvTranspose2d(in_channels,
                                                                   out_channels,
                                                                   kernel_size=3,
                                                                   stride=2,
                                                                   padding=1,
                                                                   output_padding=1) for _ in range(kernel_size_h**2)])
        else:
            self.kernels = torch.nn.ModuleList([nn.Conv2d(in_channels,
                                                          out_channels,
                                                          kernel_size=kernel_size_l,
                                                          stride=1,
                                                          padding=kernel_size_l // 2,
                                                          ) for _ in range(kernel_size_h ** 2)])

    def forward(self, inp):
        # tic = time.time()
        b, c, h, w = inp.shape
        assert h == w
        inv = h//self.t_size

        if self.s_mode:
            inp = F.pad(inp, [self.p*inv, self.p*inv, self.p*inv, self.p*inv])
            size = self.t_size+2*self.p
            h += 2*self.p*inv
            w += 2*self.p*inv

            grids = inp.as_strided(
                (b, size, size, c, inv, inv),
                (c * h * w, inv * w, inv, h * w, w, 1)
            ).contiguous()
            grids = grids.as_strided(
                (b * size ** 2, c, inv, inv),
                (c * inv ** 2, inv ** 2, inv, 1)
            ).contiguous()
            new_size = self.t_size
        else:
            org_grids = inp.as_strided(
                (b, self.t_size, self.t_size, c, inv, inv),
                (c * h * w, inv * w, inv, h * w, w, 1)
            ).contiguous()
            org_grids = org_grids.as_strided(
                (b * self.t_size ** 2, c, inv, inv),
                (c * inv ** 2, inv ** 2, inv, 1)
            ).contiguous()
            new_size = self.t_size * self.stride
            size = new_size+2*self.p
            grids = inp.new_zeros((b*size**2, c, inv, inv))
            inds = [b_i*size**2+i*size+j for b_i in range(b) for i in range(self.p, size-self.p)
                    for j in range(self.p, size-self.p)
                    if (i-self.p)%self.stride==0 and (j-self.p)%self.stride==0]
            grids[inds] = org_grids

        h_o, w_o = 0, 0
        for k, ker in enumerate(self.kernels):
            row, col = k // self.kernel_size_h, k % self.kernel_size_h
            inds = [b_i*size**2+i*size+j for b_i in range(b) for i in range(row, size-self.kernel_size_h+1+row)
                    for j in range(col, size-self.kernel_size_h+1+col)]
            temp = grids[inds, :, :, :]
            assert len(inds)==new_size**2*b
            # t_h, t_w = size-row, size-col

            # start = time.time()
            x = ker(temp)
            # x = temp
            _, c_o, h_o, w_o = x.shape
            # x = x.reshape(b, self.t_size**2, c_o, h_o, w_o)
            # print('conv time:%.4f' % (time.time()-start))

            # tic = time.time()
            x_o = x.as_strided(
                (b, c_o, new_size, h_o, new_size, w_o),
                (c_o * h_o * w_o * new_size**2, h_o * w_o,
                 new_size * c_o * h_o * w_o, w_o, c_o * h_o * w_o, 1)
            ).contiguous()
            x_o = x_o.as_strided(
                (b, c_o, new_size*h_o, new_size*w_o),
                (c_o*h_o*w_o*new_size**2, new_size**2*h_o*w_o, new_size*w_o, 1)
            ).contiguous()
            # x_o = F.pad(x_o, [0, w_o*col, 0, h_o*row])
            # print('arran time:%.4f' % (time.time()-tic))
            if k==0:
                out = x_o
            else:
                out += x_o
            # print('Each Conv kernel:%.4f' % (time.time() - start))

        out_x = out

        return out_x


class SpatialMaxPool(nn.MaxPool2d):
    def __init__(self, t_size, **kwargs):
        super(SpatialMaxPool, self).__init__(**kwargs)
        self.t_size = t_size

    def compute_pad(self, s):
        if s % self.stride == 0:
            return max(self.kernel_size - self.stride, 0)
        else:
            return max(self.kernel_size - (s % self.stride), 0)

    def forward(self, x):
        (b, c, h, w) = x.size()
        assert h == w
        inv = h//self.t_size
        grids = x.as_strided(
            (b, self.t_size, self.t_size, c, inv, inv),
            (c * h * w, w, 1, h*w, self.t_size*w, self.t_size)
        ).contiguous()
        grids = grids.as_strided(
            (b * self.t_size ** 2, c, inv, inv),
            (c * inv ** 2, inv ** 2, inv, 1)
        ).contiguous()

        # compute 'same' padding
        pad = self.compute_pad(inv)

        pad_f = pad//2
        pad_b = pad - pad_f

        pad = [pad_f, pad_b, pad_f, pad_b]
        x = F.pad(grids, pad)
        x = super(MaxPool2dSamePadding, self).forward(x)

        c_o = c
        h_o, w_o = inv//self.stride, inv//self.stride
        x_o = x.as_strided(
            (b, c_o, self.t_size, h_o, self.t_size, w_o),
            (c_o * h_o * w_o * self.t_size ** 2, h_o * w_o,
             self.t_size * c_o * h_o * w_o, w_o, c_o * h_o * w_o, 1)
        ).contiguous()
        x_o = x_o.as_strided(
            (b, c_o, self.t_size * h_o, self.t_size * w_o),
            (c_o * h_o * w_o * self.t_size ** 2, self.t_size * h_o * self.t_size * w_o, self.t_size * w_o, 1)
        ).contiguous()

        return x_o


class GridMaxPool(nn.Module):
    def __init__(self, t_size, kernel_size=3, stride=1):
        super(GridMaxPool, self).__init__()
        self.t_size = t_size
        self.kernel_size = kernel_size
        self.max_pool = nn.MaxPool2d(kernel_size=kernel_size,
                                     stride=stride,
                                     padding=kernel_size//2)

    def forward(self, inp):
        b, c, h, w = inp.shape
        assert h == w
        inv = h // self.t_size

        out = []
        for row in range(self.kernel_size):
            for col in range(self.kernel_size):
                x = inp[:, :, row*inv:, col*inv:]
                pad = [0, col*inv, 0, row*inv]
                x = F.pad(x, pad)
                x = self.max_pool(x)
                out.append(x)
        out = torch.stack(out).max(dim=0)[0]

        return out


class MaxPool2dSamePadding(nn.MaxPool2d):
    def __init__(self, t_size, **kwargs):
        super(MaxPool2dSamePadding, self).__init__(**kwargs)
        self.t_size = t_size

    def compute_pad(self, s):
        if s % self.stride == 0:
            return max(self.kernel_size - self.stride, 0)
        else:
            return max(self.kernel_size - (s % self.stride), 0)

    def forward(self, x):
        (b, c, h, w) = x.size()
        assert h == w
        inv = h//self.t_size
        grids = x.as_strided(
            (b, self.t_size, self.t_size, c, inv, inv),
            (c * h * w, inv * w, inv, h * w, w, 1)
        ).contiguous()
        grids = grids.as_strided(
            (b * self.t_size ** 2, c, inv, inv),
            (c * inv ** 2, inv ** 2, inv, 1)
        ).contiguous()

        # compute 'same' padding
        pad = self.compute_pad(inv)

        pad_f = pad//2
        pad_b = pad - pad_f

        pad = [pad_f, pad_b, pad_f, pad_b]
        x = F.pad(grids, pad)
        x = super(MaxPool2dSamePadding, self).forward(x)

        c_o = c
        h_o, w_o = inv//self.stride, inv//self.stride
        x_o = x.as_strided(
            (b, c_o, self.t_size, h_o, self.t_size, w_o),
            (c_o * h_o * w_o * self.t_size ** 2, h_o * w_o,
             self.t_size * c_o * h_o * w_o, w_o, c_o * h_o * w_o, 1)
        ).contiguous()
        x_o = x_o.as_strided(
            (b, c_o, self.t_size * h_o, self.t_size * w_o),
            (c_o * h_o * w_o * self.t_size ** 2, self.t_size * h_o * self.t_size * w_o, self.t_size * w_o, 1)
        ).contiguous()

        return x_o


class MyM(nn.Module):
    def __init__(self, **kwargs):
        super(MyM, self).__init__()
        self.conv = nn.Conv3d(**kwargs)

    def forward(self, x):
        return self.conv(x)


if __name__ == '__main__':
    d_in = torch.ones(1, 3, 5, 56, 56)
    m = S4D(3, 12, alpha=2, beta=4)
    from utils.thop import profile, clever_format
    macs, params = profile(m, inputs=(d_in,))
    macs, params = clever_format([macs, params], "%.3f")
    print('Macs:' + macs + ', Params:' + params)
