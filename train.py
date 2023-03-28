import os
import json
import argparse
import itertools
import math
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler

import commons
import utils
from dataloader_foley import (AudioSet, get_dataset_filelist, AudioCollate, DistributedBucketSampler, clas_dict)

from models import (
    SynthesizerTrn,
    AvocodoDiscriminator,
)
from losses import (generator_loss, discriminator_loss, feature_loss, kl_loss)
from mel_processing import mel_spectrogram_torch, spec_to_mel_torch

from phaseaug.phaseaug import PhaseAug

torch.backends.cudnn.benchmark = True
global_step = 0


def main():
    """Assume Single Node Multi GPUs Training Only"""
    assert torch.cuda.is_available(), "CPU training is not allowed."

    n_gpus = torch.cuda.device_count()
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '80000'

    hps = utils.get_hparams()
    mp.spawn(run, nprocs=n_gpus, args=(
        n_gpus,
        hps,
    ))


def run(rank, n_gpus, hps):
    global global_step
    if rank == 0:
        logger = utils.get_logger(hps.model_dir)
        logger.info(hps)
        utils.check_git_hash(hps.model_dir)
        writer = SummaryWriter(log_dir=hps.model_dir)

    dist.init_process_group(backend='nccl',
                            init_method='env://',
                            world_size=n_gpus,
                            rank=rank)
    torch.manual_seed(hps.train.seed)
    torch.cuda.set_device(rank)

    train_file_list: List[dict] = get_dataset_filelist()
    train_set = AudioSet(train_file_list, 22050, 22050 * 4, hps.data.filter_length, hps.data.hop_length, hps.data.win_length)
    train_sampler = DistributedSampler(
        train_set,
        num_replicas=n_gpus,
        rank=rank,
        shuffle=True,
        drop_last=True,
        )
    train_loader = DataLoader(
        train_set,
        batch_size=hps.train.batch_size,
        num_workers=8,
        shuffle=False,
        pin_memory=True,
	sampler=train_sampler
    )

    #
    #    train_dataset = TextAudioSpeakerLoader(hps.data.training_files, hps.data)

    #    collate_fn = TextAudioSpeakerCollate()
    #    train_loader = DataLoader(train_dataset,
    #                              num_workers=8,
    #                              shuffle=False,
    #                              pin_memory=True,
    #                              collate_fn=collate_fn,
    #                              batch_sampler=train_sampler)

    net_g = SynthesizerTrn(len(clas_dict),
                           hps.data.filter_length // 2 + 1,
                           hps.train.segment_size // hps.data.hop_length,
                           **hps.model).cuda(rank)
    net_d = AvocodoDiscriminator(hps.model.use_spectral_norm).cuda(rank)
    optim_g = torch.optim.AdamW(net_g.parameters(),
                                hps.train.learning_rate,
                                betas=hps.train.betas,
                                eps=hps.train.eps)
    optim_d = torch.optim.AdamW(net_d.parameters(),
                                hps.train.learning_rate,
                                betas=hps.train.betas,
                                eps=hps.train.eps)
    net_g = DDP(net_g, device_ids=[rank])
    net_d = DDP(net_d, device_ids=[rank])

    try:
        _, _, _, epoch_str = utils.load_checkpoint(
            utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), net_g,
            optim_g)
        _, _, _, epoch_str = utils.load_checkpoint(
            utils.latest_checkpoint_path(hps.model_dir, "D_*.pth"), net_d,
            optim_d)
        global_step = (epoch_str - 1) * len(train_loader)
    except:
        epoch_str = 1
        global_step = 0

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2)

    scaler = GradScaler(enabled=hps.train.fp16_run)

    for epoch in range(epoch_str, hps.train.epochs + 1):
        if rank == 0:
            train_and_evaluate(rank, epoch, hps, [net_g, net_d],
                               [optim_g, optim_d], [scheduler_g, scheduler_d],
                               scaler, train_loader, logger,
                               writer)
        else:
            train_and_evaluate(rank, epoch, hps, [net_g, net_d],
                               [optim_g, optim_d], [scheduler_g, scheduler_d],
                               scaler, train_loader, None, None)
        scheduler_g.step()
        scheduler_d.step()


def train_and_evaluate(rank, epoch, hps, nets, optims, schedulers, scaler,
                       train_loader, logger, writer):
    net_g, net_d = nets
    optim_g, optim_d = optims
    scheduler_g, scheduler_d = schedulers
    aug = PhaseAug().cuda(rank)

    global global_step

    net_g.train()
    net_d.train()
    for batch_idx, (y, x_lengths, spec, x) in enumerate(train_loader):
        x, x_lengths = x.cuda(rank, non_blocking=True), x_lengths.cuda(
            rank, non_blocking=True)
        spec = spec.cuda(
            rank, non_blocking=True)
        y= y.cuda(rank, non_blocking=True)
        with autocast(enabled=hps.train.fp16_run):
            y_hat, ids_slice,\
            (z, z_p, m_p, logs_p, m_q, logs_q) = net_g(x, x_lengths, spec)

            mel = spec_to_mel_torch(spec, hps.data.filter_length,
                                    hps.data.n_mel_channels,
                                    hps.data.sampling_rate, hps.data.mel_fmin,
                                    hps.data.mel_fmax)
            y_mel = commons.slice_segments(
                mel, ids_slice, hps.train.segment_size // hps.data.hop_length)
            y_hat_mel = mel_spectrogram_torch(
                y_hat[-1].squeeze(1), hps.data.filter_length,
                hps.data.n_mel_channels, hps.data.sampling_rate,
                hps.data.hop_length, hps.data.win_length, hps.data.mel_fmin,
                hps.data.mel_fmax)

            y = commons.slice_segments(y, ids_slice * hps.data.hop_length,
                                       hps.train.segment_size)  # slice

            # Discriminator
            #y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())
            with autocast(enabled=False):
                aug_y_, aug_y_hat_last = aug.forward_sync(
                    y.float(), y_hat[-1].detach().float())
                aug_y_hat_ = [_y.detach() for _y in y_hat[:-1]]
                aug_y_hat_.append(aug_y_hat_last)
            y_d_hat_r, y_d_hat_g, _, _ = net_d(aug_y_, aug_y_hat_)
            with autocast(enabled=False):
                loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(
                    y_d_hat_r, y_d_hat_g)
                loss_disc_all = loss_disc
        optim_d.zero_grad()
        scaler.scale(loss_disc_all).backward()
        scaler.unscale_(optim_d)
        grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
        scaler.step(optim_d)

        with autocast(enabled=hps.train.fp16_run):
            # Generator
            #y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
            with autocast(enabled=False):
                aug_y_, aug_y_hat_last = aug.forward_sync(y.float(), y_hat[-1].float())
                aug_y_hat_ = y_hat
                aug_y_hat_[-1] = aug_y_hat_last
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(aug_y_, aug_y_hat_)
            with autocast(enabled=False):
                loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
                loss_kl = kl_loss(z_p, logs_q, m_p, logs_p ) * hps.train.c_kl

                loss_fm = feature_loss(fmap_r, fmap_g)
                loss_gen, losses_gen = generator_loss(y_d_hat_g)
                loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl
        optim_g.zero_grad()
        scaler.scale(loss_gen_all).backward()
        scaler.unscale_(optim_g)
        grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
        scaler.step(optim_g)
        scaler.update()

        if rank == 0:
            if global_step % hps.train.log_interval == 0:
                lr = optim_g.param_groups[0]['lr']
                losses = [
                    loss_disc, loss_gen, loss_fm, loss_mel, loss_kl
                ]
                logger.info('Train Epoch: {} [{:.0f}%]'.format(
                    epoch, 100. * batch_idx / len(train_loader)))
                logger.info([x.item() for x in losses] + [global_step, lr])

                scalar_dict = {
                    "loss/g/total": loss_gen_all,
                    "loss/d/total": loss_disc_all,
                    "learning_rate": lr,
                    "grad_norm_d": grad_norm_d,
                    "grad_norm_g": grad_norm_g
                }
                scalar_dict.update({
                    "loss/g/fm": loss_fm,
                    "loss/g/mel": loss_mel,
                    "loss/g/kl": loss_kl
                })

                scalar_dict.update({
                    "loss/g/{}".format(i): v
                    for i, v in enumerate(losses_gen)
                })
                scalar_dict.update({
                    "loss/d_r/{}".format(i): v
                    for i, v in enumerate(losses_disc_r)
                })
                scalar_dict.update({
                    "loss/d_g/{}".format(i): v
                    for i, v in enumerate(losses_disc_g)
                })
                image_dict = {
                    "slice/mel_org":
                    utils.plot_spectrogram_to_numpy(
                        y_mel[0].data.cpu().numpy()),
                    "slice/mel_gen":
                    utils.plot_spectrogram_to_numpy(
                        y_hat_mel[0].data.cpu().numpy()),
                    "all/mel":
                    utils.plot_spectrogram_to_numpy(mel[0].data.cpu().numpy()),
                }
                utils.summarize(writer=writer,
                                global_step=global_step,
                                images=image_dict,
                                scalars=scalar_dict)

            if global_step % hps.train.eval_interval == 0:
                evaluate(rank, hps, net_g, writer)
                utils.save_checkpoint(
                    net_g, optim_g, hps.train.learning_rate, epoch,
                    os.path.join(hps.model_dir,
                                 "G_{}.pth".format(global_step)))
                utils.save_checkpoint(
                    net_d, optim_d, hps.train.learning_rate, epoch,
                    os.path.join(hps.model_dir,
                                 "D_{}.pth".format(global_step)))
        global_step += 1

    if rank == 0:
        logger.info('====> Epoch: {}'.format(epoch))


def evaluate(rank, hps, generator, writer):
    generator.eval()
    with torch.no_grad():
        x = torch.arange(len(clas_dict), device=generator.device)
#        for batch_idx, (x, x_lengths, spec, spec_lengths, y, y_lengths,
#                        speakers) in enumerate(eval_loader):
#            x, x_lengths = x.cuda(0), x_lengths.cuda(0)
#            spec, spec_lengths = spec.cuda(0), spec_lengths.cuda(0)
#            y, y_lengths = y.cuda(0), y_lengths.cuda(0)
#            speakers = speakers.cuda(0)
        y_hat = generator.module.infer(x,                 344,
                                             )
        y_hat_lengths = 344

        y_hat_mel = mel_spectrogram_torch(
            y_hat.squeeze(1).float(), hps.data.filter_length,
            hps.data.n_mel_channels, hps.data.sampling_rate,
            hps.data.hop_length, hps.data.win_length, hps.data.mel_fmin,
            hps.data.mel_fmax)
        if rank==0:
            image_dict = {
                f"gen/mel{i}": utils.plot_spectrogram_to_numpy(y_hat_mel[i].cpu().numpy()) \
                        for i in range(len(clas_dict))
            }
            audio_dict = {
                    f"gen/audio{i}": y_hat[i, :, :] \
                        for i in range(len(clas_dict))
            }       
            utils.summarize(writer=writer,
                            global_step=global_step,
                            images=image_dict,
                            audios=audio_dict,
                            audio_sampling_rate=hps.data.sampling_rate)
    generator.train()


if __name__ == "__main__":
    main()
