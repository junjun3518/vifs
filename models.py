# from https://github.com/jaywalnut310/vits
# from https://github.com/ncsoft/avocodo
# from https://github.com/anonymous-pits/pits
import copy
import math
import torch
from torch import nn
from torch.nn import functional as F

import commons
import modules
import attentions

from torch.nn import Conv1d, ConvTranspose1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
from commons import init_weights, get_padding

from pqmf import PQMF


class StochasticDurationPredictor(nn.Module):

    def __init__(self,
                 in_channels,
                 filter_channels,
                 kernel_size,
                 p_dropout,
                 n_flows=4,
                 gin_channels=0):
        super().__init__()
        filter_channels = in_channels  # it needs to be removed from future version.
        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        self.log_flow = modules.Log()
        self.flows = nn.ModuleList()
        self.flows.append(modules.ElementwiseAffine(2))
        for i in range(n_flows):
            self.flows.append(
                modules.ConvFlow(2, filter_channels, kernel_size, n_layers=3))
            self.flows.append(modules.Flip())

        self.post_pre = nn.Conv1d(1, filter_channels, 1)
        self.post_proj = nn.Conv1d(filter_channels, filter_channels, 1)
        self.post_convs = modules.DDSConv(filter_channels,
                                          kernel_size,
                                          n_layers=3,
                                          p_dropout=p_dropout)
        self.post_flows = nn.ModuleList()
        self.post_flows.append(modules.ElementwiseAffine(2))
        for i in range(4):
            self.post_flows.append(
                modules.ConvFlow(2, filter_channels, kernel_size, n_layers=3))
            self.post_flows.append(modules.Flip())

        self.pre = nn.Conv1d(in_channels, filter_channels, 1)
        self.proj = nn.Conv1d(filter_channels, filter_channels, 1)
        self.convs = modules.DDSConv(filter_channels,
                                     kernel_size,
                                     n_layers=3,
                                     p_dropout=p_dropout)
        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, filter_channels, 1)

    def forward(self, x, w=None, g=None, reverse=False, noise_scale=1.0):
        x = torch.detach(x)
        x = self.pre(x)
        if g is not None:
            g = torch.detach(g)
            x = x + self.cond(g)
        x = self.convs(x)
        x = self.proj(x)

        if not reverse:
            flows = self.flows
            assert w is not None

            logdet_tot_q = 0
            h_w = self.post_pre(w)
            h_w = self.post_convs(h_w)
            h_w = self.post_proj(h_w)
            e_q = torch.randn(w.size(0), 2, w.size(2)).to(device=x.device,
                                                          dtype=x.dtype)
            z_q = e_q
            for flow in self.post_flows:
                z_q, logdet_q = flow(z_q, g=(x + h_w))
                logdet_tot_q += logdet_q
            z_u, z1 = torch.split(z_q, [1, 1], 1)
            u = torch.sigmoid(z_u)
            z0 = (w - u)
            logdet_tot_q += torch.sum((F.logsigmoid(z_u) + F.logsigmoid(-z_u)),
                                      [1, 2])
            logq = torch.sum(-0.5 * (math.log(2 * math.pi) +
                                     (e_q**2)), [1, 2]) - logdet_tot_q

            logdet_tot = 0
            z0, logdet = self.log_flow(z0)
            logdet_tot += logdet
            z = torch.cat([z0, z1], 1)
            for flow in flows:
                z, logdet = flow(z, g=x, reverse=reverse)
                logdet_tot = logdet_tot + logdet
            nll = torch.sum(0.5 * (math.log(2 * math.pi) +
                                   (z**2)), [1, 2]) - logdet_tot
            return nll + logq  # [b]
        else:
            flows = list(reversed(self.flows))
            flows = flows[:-2] + [flows[-1]]  # remove a useless vflow
            z = torch.randn(x.size(0), 2, x.size(2)).to(
                device=x.device, dtype=x.dtype) * noise_scale
            for flow in flows:
                z = flow(z, g=x, reverse=reverse)
            z0, z1 = torch.split(z, [1, 1], 1)
            logw = z0
            return logw


class DurationPredictor(nn.Module):

    def __init__(self,
                 in_channels,
                 filter_channels,
                 kernel_size,
                 p_dropout,
                 gin_channels=0):
        super().__init__()

        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.gin_channels = gin_channels

        self.drop = nn.Dropout(p_dropout)
        self.conv_1 = nn.Conv1d(in_channels,
                                filter_channels,
                                kernel_size,
                                padding=kernel_size // 2)
        self.norm_1 = modules.LayerNorm(filter_channels)
        self.conv_2 = nn.Conv1d(filter_channels,
                                filter_channels,
                                kernel_size,
                                padding=kernel_size // 2)
        self.norm_2 = modules.LayerNorm(filter_channels)
        self.proj = nn.Conv1d(filter_channels, 1, 1)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, in_channels, 1)

    def forward(self, x, g=None):
        x = torch.detach(x)
        if g is not None:
            g = torch.detach(g)
            x = x + self.cond(g)
        x = self.conv_1(x)
        x = torch.relu(x)
        x = self.norm_1(x)
        x = self.drop(x)
        x = self.conv_2(x)
        x = torch.relu(x)
        x = self.norm_2(x)
        x = self.drop(x)
        x = self.proj(x)
        return x


class PriorEncoder(nn.Module):

    def __init__(self, n_class, out_channels, hidden_channels, kernel_size,
                 dilation_rate, n_layers):
        super().__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.n_layers = n_layers
        self.kernel_size = kernel_size

        self.emb = nn.Embedding(n_class, hidden_channels)
        nn.init.normal_(self.emb.weight, 0.0, hidden_channels**-0.5)
        self.positional = nn.Parameter(torch.zeros(1, hidden_channels, 344))
        #self.encoder = attentions.Encoder(hidden_channels, filter_channels,
        #                                  n_heads, n_layers, kernel_size,
        #                                  p_dropout)
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)
        self.encoder = modules.WN(hidden_channels,
                                  kernel_size,
                                  dilation_rate,
                                  n_layers,
                                  gin_channels=0)

    def forward(self, x, z=None):
        x = self.emb(x)  # [b, h]
        if z is None:
            x = x + torch.randn_like(x) * self.hidden_channels**(-0.5)
        else:
            x = x + z * self.hidden_channels**(-0.5)
        x = x.unsqueeze(-1).expand(*x.shape, 344)  # [b, h, t]
        x = x + self.positional
        x = self.encoder(x)
        stats = self.proj(x)

        #m, logs = torch.split(stats, self.out_channels, dim=1)
        return stats


class FramePriorNetwork(nn.Module):

    def __init__(self, channels, kernel_size, n_layers):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.convs = nn.ModuleList()
        for i in range(n_layers):
            self.convs.append(
                weight_norm(
                    nn.Conv1d(channels,
                              channels,
                              kernel_size,
                              1,
                              dilation=1,
                              padding=get_padding(kernel_size, 1))))

    def forward(self, x):
        for i in range(self.n_layers):
            xt = F.leaky_relu(x, 0.2)
            xt = self.convs[i](xt)
            x = xt + x
        m, logs = torch.split(x, self.channels // 2, dim=1)
        return m, logs


class ResidualCouplingBlock(nn.Module):

    def __init__(self,
                 channels,
                 hidden_channels,
                 kernel_size,
                 dilation_rate,
                 n_layers,
                 n_flows=4,
                 gin_channels=0):
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        self.flows = nn.ModuleList()
        for i in range(n_flows):
            self.flows.append(
                modules.ResidualCouplingLayer(channels,
                                              hidden_channels,
                                              kernel_size,
                                              dilation_rate,
                                              n_layers,
                                              gin_channels=gin_channels,
                                              mean_only=True))
            self.flows.append(modules.Flip())

    def forward(self, x, g=None, reverse=False):
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, g=g, reverse=reverse)
        else:
            for flow in reversed(self.flows):
                x = flow(x, g=g, reverse=reverse)
        return x


class PosteriorEncoder(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden_channels,
                 kernel_size,
                 dilation_rate,
                 n_layers,
                 gin_channels=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = modules.WN(hidden_channels,
                              kernel_size,
                              dilation_rate,
                              n_layers,
                              gin_channels=gin_channels)
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, g=None):
        x = self.pre(x)
        x = self.enc(x, g=g)
        stats = self.proj(x)
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs))
        return z, m, logs


class Generator(nn.Module):

    def __init__(self,
                 initial_channel,
                 resblock,
                 resblock_kernel_sizes,
                 resblock_dilation_sizes,
                 upsample_rates,
                 upsample_initial_channel,
                 upsample_kernel_sizes,
                 gin_channels=0):
        super(Generator, self).__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = Conv1d(initial_channel,
                               upsample_initial_channel,
                               7,
                               1,
                               padding=3)
        resblock = modules.ResBlock1 if resblock == '1' else modules.ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(
                    ConvTranspose1d(upsample_initial_channel // (2**i),
                                    upsample_initial_channel // (2**(i + 1)),
                                    k,
                                    u,
                                    padding=(k - u) // 2)))

        self.resblocks = nn.ModuleList()
        self.conv_posts = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2**(i + 1))
            for j, (k, d) in enumerate(
                    zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(resblock(ch, k, d))
            if i >= len(self.ups) - 3:
                self.conv_posts.append(
                    Conv1d(ch, 1, 7, 1, padding=3, bias=False))
        self.ups.apply(init_weights)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

    def forward(self, x, g=None):
        x = self.conv_pre(x)
        if g is not None:
            x = x + self.cond(g)

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                xs = xs + self.resblocks[i * self.num_kernels + j](x) if xs is not None \
                     else self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_posts[-1](x)
        x = torch.tanh(x)

        return x

    def hier_forward(self, x, g=None):
        outs = []
        x = self.conv_pre(x)
        if g is not None:
            x = x + self.cond(g)

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                xs = xs + self.resblocks[i * self.num_kernels + j](x) if xs is not None \
                     else self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
            if i >= self.num_upsamples - 3:
                _x = F.leaky_relu(x)
                _x = self.conv_posts[i - self.num_upsamples + 3](_x)
                _x = torch.tanh(_x)
                outs.append(_x)
        return outs

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()


class DiscriminatorP(torch.nn.Module):

    def __init__(self,
                 period,
                 kernel_size=5,
                 stride=3,
                 use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        self.use_spectral_norm = use_spectral_norm
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(
                Conv2d(1,
                       32, (kernel_size, 1), (stride, 1),
                       padding=(get_padding(kernel_size, 1), 0))),
            norm_f(
                Conv2d(32,
                       128, (kernel_size, 1), (stride, 1),
                       padding=(get_padding(kernel_size, 1), 0))),
            norm_f(
                Conv2d(128,
                       512, (kernel_size, 1), (stride, 1),
                       padding=(get_padding(kernel_size, 1), 0))),
            norm_f(
                Conv2d(512,
                       1024, (kernel_size, 1), (stride, 1),
                       padding=(get_padding(kernel_size, 1), 0))),
            norm_f(
                Conv2d(1024,
                       1024, (kernel_size, 1),
                       1,
                       padding=(get_padding(kernel_size, 1), 0))),
        ])
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class DiscriminatorS(torch.nn.Module):

    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv1d(1, 16, 15, 1, padding=7)),
            norm_f(Conv1d(16, 64, 41, 4, groups=4, padding=20)),
            norm_f(Conv1d(64, 256, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
            norm_f(Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
            norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):

    def __init__(self, use_spectral_norm=False):
        super(MultiPeriodDiscriminator, self).__init__()
        periods = [2, 3, 5, 7, 11]

        discs = [DiscriminatorS(use_spectral_norm=use_spectral_norm)]
        discs = discs + [
            DiscriminatorP(i, use_spectral_norm=use_spectral_norm)
            for i in periods
        ]
        self.discriminators = nn.ModuleList(discs)

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


##### Avocodo
class CoMBDBlock(torch.nn.Module):

    def __init__(
            self,
            h_u,  # List[int],
            d_k,  # List[int],
            d_s,  # List[int],
            d_d,  # List[int],
            d_g,  # List[int],
            d_p,  # List[int],
            op_f,  # int,
            op_k,  # int,
            op_g,  # int,
            use_spectral_norm=False):
        super(CoMBDBlock, self).__init__()
        norm_f = weight_norm if use_spectral_norm is False else spectral_norm

        self.convs = nn.ModuleList()
        filters = [[1, h_u[0]]]
        for i in range(len(h_u) - 1):
            filters.append([h_u[i], h_u[i + 1]])
        for _f, _k, _s, _d, _g, _p in zip(filters, d_k, d_s, d_d, d_g, d_p):
            self.convs.append(
                norm_f(
                    Conv1d(in_channels=_f[0],
                           out_channels=_f[1],
                           kernel_size=_k,
                           stride=_s,
                           dilation=_d,
                           groups=_g,
                           padding=_p)))
        self.projection_conv = norm_f(
            Conv1d(in_channels=filters[-1][1],
                   out_channels=op_f,
                   kernel_size=op_k,
                   groups=op_g))

    def forward(self, x, b_y, b_y_hat):
        fmap_r = []
        fmap_g = []
        for block in self.convs:
            x = block(x)
            x = F.leaky_relu(x, 0.2)
            f_r, f_g = x.split([b_y, b_y_hat], dim=0)
            fmap_r.append(f_r.tile([2, 1, 1]) if b_y < b_y_hat else f_r)
            fmap_g.append(f_g)
        x = self.projection_conv(x)
        x_r, x_g = x.split([b_y, b_y_hat], dim=0)
        return x_r.tile([2, 1, 1
                         ]) if b_y < b_y_hat else x_r, x_g, fmap_r, fmap_g


class CoMBD(torch.nn.Module):

    def __init__(self, use_spectral_norm=False):
        super(CoMBD, self).__init__()
        self.pqmf_list = nn.ModuleList([
            PQMF(4, 192, 0.13, 10.0),  #lv2
            PQMF(2, 256, 0.25, 10.0)  #lv1
        ])
        combd_h_u = [[16, 64, 256, 1024, 1024, 1024] for _ in range(3)]
        combd_d_k = [[7, 11, 11, 11, 11, 5], [11, 21, 21, 21, 21, 5],
                     [15, 41, 41, 41, 41, 5]]
        combd_d_s = [[1, 1, 4, 4, 4, 1] for _ in range(3)]
        combd_d_d = [[1, 1, 1, 1, 1, 1] for _ in range(3)]
        combd_d_g = [[1, 4, 16, 64, 256, 1] for _ in range(3)]

        combd_d_p = [[3, 5, 5, 5, 5, 2], [5, 10, 10, 10, 10, 2],
                     [7, 20, 20, 20, 20, 2]]
        combd_op_f = [1, 1, 1]
        combd_op_k = [3, 3, 3]
        combd_op_g = [1, 1, 1]

        self.blocks = nn.ModuleList()
        for _h_u, _d_k, _d_s, _d_d, _d_g, _d_p, _op_f, _op_k, _op_g in zip(
                combd_h_u,
                combd_d_k,
                combd_d_s,
                combd_d_d,
                combd_d_g,
                combd_d_p,
                combd_op_f,
                combd_op_k,
                combd_op_g,
        ):
            self.blocks.append(
                CoMBDBlock(
                    _h_u,
                    _d_k,
                    _d_s,
                    _d_d,
                    _d_g,
                    _d_p,
                    _op_f,
                    _op_k,
                    _op_g,
                ))

    def _block_forward(self, ys, ys_hat, blocks):
        outs_real = []
        outs_fake = []
        f_maps_real = []
        f_maps_fake = []
        for y, y_hat, block in zip(ys, ys_hat,
                                   blocks):  #y:B, y_hat: 2B if i!=-1 else B,B
            b_y = y.shape[0]
            b_y_hat = y_hat.shape[0]
            cat_y = torch.cat([y, y_hat], dim=0)
            out_real, out_fake, f_map_r, f_map_g = block(cat_y, b_y, b_y_hat)
            outs_real.append(out_real)
            outs_fake.append(out_fake)
            f_maps_real.append(f_map_r)
            f_maps_fake.append(f_map_g)
        return outs_real, outs_fake, f_maps_real, f_maps_fake

    def _pqmf_forward(self, ys, ys_hat):
        # preprocess for multi_scale forward
        multi_scale_inputs_hat = []
        for pqmf_ in self.pqmf_list:
            multi_scale_inputs_hat.append(pqmf_.analysis(ys_hat[-1])[:, :1, :])

        # real
        # for hierarchical forward
        #outs_real_, f_maps_real_ = self._block_forward(
        #    ys, self.blocks)

        # for multi_scale forward
        #outs_real, f_maps_real = self._block_forward(
        #        ys[:-1], self.blocks[:-1], outs_real, f_maps_real)
        #outs_real.extend(outs_real[:-1])
        #f_maps_real.extend(f_maps_real[:-1])

        #outs_real = [torch.cat([o,o], dim=0) if i!=len(outs_real_)-1 else o for i,o in enumerate(outs_real_)]
        #f_maps_real = [[torch.cat([fmap,fmap], dim=0) if i!=len(f_maps_real_)-1 else fmap for fmap in fmaps ] \
        #        for i,fmaps in enumerate(f_maps_real_)]

        inputs_fake = [
            torch.cat([y, multi_scale_inputs_hat[i]], dim=0)
            if i != len(ys_hat) - 1 else y for i, y in enumerate(ys_hat)
        ]
        outs_real, outs_fake, f_maps_real, f_maps_fake = self._block_forward(
            ys, inputs_fake, self.blocks)

        # predicted
        # for hierarchical forward
        #outs_fake, f_maps_fake = self._block_forward(
        #    inputs_fake, self.blocks)

        #outs_real_, f_maps_real_ = self._block_forward(
        #    ys, self.blocks)
        # for multi_scale forward
        #outs_fake, f_maps_fake = self._block_forward(
        #    multi_scale_inputs_hat, self.blocks[:-1], outs_fake, f_maps_fake)

        return outs_real, outs_fake, f_maps_real, f_maps_fake

    def forward(self, ys, ys_hat):
        outs_real, outs_fake, f_maps_real, f_maps_fake = self._pqmf_forward(
            ys, ys_hat)
        return outs_real, outs_fake, f_maps_real, f_maps_fake


class MDC(torch.nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 strides,
                 kernel_size,
                 dilations,
                 use_spectral_norm=False):
        super(MDC, self).__init__()
        norm_f = weight_norm if not use_spectral_norm else spectral_norm
        self.d_convs = nn.ModuleList()
        for _k, _d in zip(kernel_size, dilations):
            self.d_convs.append(
                norm_f(
                    Conv1d(in_channels=in_channels,
                           out_channels=out_channels,
                           kernel_size=_k,
                           dilation=_d,
                           padding=get_padding(_k, _d))))
        self.post_conv = norm_f(
            Conv1d(in_channels=out_channels,
                   out_channels=out_channels,
                   kernel_size=3,
                   stride=strides,
                   padding=get_padding(_k, _d)))
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x):
        _out = None
        for _l in self.d_convs:
            _x = torch.unsqueeze(_l(x), -1)
            _x = F.leaky_relu(_x, 0.2)
            _out = torch.cat([_out, _x], axis=-1) if _out is not None \
                   else _x
        x = torch.sum(_out, dim=-1)
        x = self.post_conv(x)
        x = F.leaky_relu(x, 0.2)  # @@

        return x


class SBDBlock(torch.nn.Module):

    def __init__(self,
                 segment_dim,
                 strides,
                 filters,
                 kernel_size,
                 dilations,
                 use_spectral_norm=False):
        super(SBDBlock, self).__init__()
        norm_f = weight_norm if not use_spectral_norm else spectral_norm
        self.convs = nn.ModuleList()
        filters_in_out = [(segment_dim, filters[0])]
        for i in range(len(filters) - 1):
            filters_in_out.append([filters[i], filters[i + 1]])

        for _s, _f, _k, _d in zip(strides, filters_in_out, kernel_size,
                                  dilations):
            self.convs.append(
                MDC(in_channels=_f[0],
                    out_channels=_f[1],
                    strides=_s,
                    kernel_size=_k,
                    dilations=_d,
                    use_spectral_norm=use_spectral_norm))
        self.post_conv = norm_f(
            Conv1d(in_channels=_f[1],
                   out_channels=1,
                   kernel_size=3,
                   stride=1,
                   padding=3 // 2))  # @@

    def forward(self, x):
        fmap_r = []
        fmap_g = []
        for _l in self.convs:
            x = _l(x)
            f_r, f_g = torch.chunk(x, 2, dim=0)
            fmap_r.append(f_r)
            fmap_g.append(f_g)
        x = self.post_conv(x)  # @@
        x_r, x_g = torch.chunk(x, 2, dim=0)
        return x_r, x_g, fmap_r, fmap_g


class MDCDConfig:

    def __init__(self):
        self.pqmf_params = [16, 256, 0.03, 10.0]
        self.f_pqmf_params = [64, 256, 0.1, 9.0]
        self.filters = [[64, 128, 256, 256, 256], [64, 128, 256, 256, 256],
                        [64, 128, 256, 256, 256], [32, 64, 128, 128, 128]]
        self.kernel_sizes = [[[7, 7, 7], [7, 7, 7], [7, 7, 7], [7, 7, 7],
                              [7, 7, 7]],
                             [[5, 5, 5], [5, 5, 5], [5, 5, 5], [5, 5, 5],
                              [5, 5, 5]],
                             [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3],
                              [3, 3, 3]],
                             [[5, 5, 5], [5, 5, 5], [5, 5, 5], [5, 5, 5],
                              [5, 5, 5]]]
        self.dilations = [[[5, 7, 11], [5, 7, 11], [5, 7, 11], [5, 7, 11],
                           [5, 7, 11]],
                          [[3, 5, 7], [3, 5, 7], [3, 5, 7], [3, 5, 7],
                           [3, 5, 7]],
                          [[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3],
                           [1, 2, 3]],
                          [[1, 2, 3], [1, 2, 3], [1, 2, 3], [2, 3, 5],
                           [2, 3, 5]]]
        self.strides = [[1, 1, 3, 3, 1], [1, 1, 3, 3, 1], [1, 1, 3, 3, 1],
                        [1, 1, 3, 3, 1]]
        self.band_ranges = [[0, 6], [0, 11], [0, 16], [0, 64]]
        self.transpose = [False, False, False, True]
        self.segment_size = 8192


class SBD(torch.nn.Module):

    def __init__(self, use_spectral_norm=False):
        super(SBD, self).__init__()
        self.config = MDCDConfig()
        self.pqmf = PQMF(*self.config.pqmf_params)
        if True in self.config.transpose:
            self.f_pqmf = PQMF(*self.config.f_pqmf_params)
        else:
            self.f_pqmf = None

        self.discriminators = torch.nn.ModuleList()

        for _f, _k, _d, _s, _br, _tr in zip(self.config.filters,
                                            self.config.kernel_sizes,
                                            self.config.dilations,
                                            self.config.strides,
                                            self.config.band_ranges,
                                            self.config.transpose):
            if _tr:
                segment_dim = self.config.segment_size // _br[1] - _br[0]
            else:
                segment_dim = _br[1] - _br[0]

            self.discriminators.append(
                SBDBlock(segment_dim=segment_dim,
                         filters=_f,
                         kernel_size=_k,
                         dilations=_d,
                         strides=_s,
                         use_spectral_norm=use_spectral_norm))

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        y_in = self.pqmf.analysis(y)
        y_hat_in = self.pqmf.analysis(y_hat)
        y_in_f = self.f_pqmf.analysis(y)
        y_hat_in_f = self.f_pqmf.analysis(y_hat)

        for d, br, tr in zip(self.discriminators, self.config.band_ranges,
                             self.config.transpose):
            if not tr:
                _y_in = y_in[:, br[0]:br[1], :]
                _y_hat_in = y_hat_in[:, br[0]:br[1], :]
            else:
                _y_in = y_in_f[:, br[0]:br[1], :]
                _y_hat_in = y_hat_in_f[:, br[0]:br[1], :]
                _y_in = torch.transpose(_y_in, 1, 2)
                _y_hat_in = torch.transpose(_y_hat_in, 1, 2)
            #y_d_r, fmap_r = d(_y_in)
            #y_d_g, fmap_g = d(_y_hat_in)
            cat_y = torch.cat([_y_in, _y_hat_in], dim=0)
            y_d_r, y_d_g, fmap_r, fmap_g = d(cat_y)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class AvocodoDiscriminator(nn.Module):

    def __init__(self, use_spectral_norm=False):
        super(AvocodoDiscriminator, self).__init__()
        self.combd = CoMBD(use_spectral_norm)
        self.sbd = SBD(use_spectral_norm)

    def forward(self, y, ys_hat):
        ys = [
            self.combd.pqmf_list[0].analysis(y)[:, :1],  #lv2
            self.combd.pqmf_list[1].analysis(y)[:, :1],  #lv1
            y
        ]
        y_c_rs, y_c_gs, fmap_c_rs, fmap_c_gs = self.combd(ys, ys_hat)
        y_s_rs, y_s_gs, fmap_s_rs, fmap_s_gs = self.sbd(y, ys_hat[-1])
        y_c_rs.extend(y_s_rs)
        y_c_gs.extend(y_s_gs)
        fmap_c_rs.extend(fmap_s_rs)
        fmap_c_gs.extend(fmap_s_gs)
        return y_c_rs, y_c_gs, fmap_c_rs, fmap_c_gs

    def infer(self, y):
        ys = [
            self.combd.pqmf_list[0].analysis(y)[:, :1],  #lv2
            self.combd.pqmf_list[1].analysis(y)[:, :1],  #lv1
            y
        ]
        B = y.shape[0]
        dummy = [torch.zeros_like(_y) for _y in ys]
        y_c_rs, _, _, _ = self.combd(ys, dummy)
        score = torch.stack([y_c_r[:B].mean((1, 2)) for y_c_r in y_c_rs],
                            dim=-1).mean(-1)
        return score


##### Avocodo


class SynthesizerTrn(nn.Module):
    """
  Synthesizer for Training
  """

    def __init__(self,
                 n_class,
                 spec_channels,
                 segment_size,
                 inter_channels,
                 hidden_channels,
                 filter_channels,
                 n_heads,
                 n_layers,
                 kernel_size,
                 p_dropout,
                 resblock,
                 resblock_kernel_sizes,
                 resblock_dilation_sizes,
                 upsample_rates,
                 upsample_initial_channel,
                 upsample_kernel_sizes,
                 n_speakers=0,
                 gin_channels=0,
                 use_sdp=True,
                 **kwargs):

        super().__init__()
        self.n_class = n_class
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.resblock = resblock
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.segment_size = segment_size
        self.gin_channels = gin_channels

        self.enc_p = PriorEncoder(
            n_class,
            inter_channels,
            hidden_channels,
            5,
            1,
            6,
        )

        self.dec = Generator(inter_channels,
                             resblock,
                             resblock_kernel_sizes,
                             resblock_dilation_sizes,
                             upsample_rates,
                             upsample_initial_channel,
                             upsample_kernel_sizes,
                             gin_channels=gin_channels)
        self.enc_q = PosteriorEncoder(spec_channels,
                                      inter_channels,
                                      hidden_channels,
                                      5,
                                      1,
                                      16,
                                      gin_channels=gin_channels)
        self.flow = ResidualCouplingBlock(inter_channels,
                                          hidden_channels,
                                          5,
                                          1,
                                          4,
                                          gin_channels=gin_channels)
        self.fpn = FramePriorNetwork(inter_channels * 2, 35, 6)

        self.emb_g = nn.Embedding(n_class, gin_channels)

    def forward(self, x, x_lengths, y):
        g = self.emb_g(x).unsqueeze(-1)  # [b, h, 1]
        z, m_q, logs_q = self.enc_q(y, g=g)  # [B,D,T]
        prior_z = torch.sum(z * torch.softmax(z.norm(dim=1), dim=1).unsqueeze(1), dim = -1)
        prior_z = F.normalize(prior_z, dim=1)
        z_p = self.flow(z, g=g)
        stats = self.enc_p(x, prior_z)
        m_p, logs_p = self.fpn(stats)

        z_slice, ids_slice = commons.rand_slice_segments(
            z, x_lengths, self.segment_size)
        o = self.dec.hier_forward(z_slice, g=g)
        return o, ids_slice, (z, z_p, m_p, logs_p, m_q, logs_q)

    def infer(
        self,
        x,
        x_lengths,
        noise_scale=1,
        length_scale=1,
        noise_scale_w=1.,
    ):
        stats = self.enc_p(x)
        m_p, logs_p = self.fpn(stats)

        g = self.emb_g(x).unsqueeze(-1)  # [b, h, 1]

        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
        z = self.flow(z_p, g=g, reverse=True)
        o = self.dec(z, g=g)
        return o
