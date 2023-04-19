import os
import json
import argparse
import itertools
import math
import torch
import soundfile as sf
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


def main(args):
    """Assume Single Node Multi GPUs Training Only"""
    assert torch.cuda.is_available(), "CPU training is not allowed."

    n_gpus = torch.cuda.device_count()
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '80000'

    hps = utils.get_hparams(args)
    mp.spawn(run, nprocs=n_gpus, args=(
        n_gpus,
        hps,
        args,
    ))


def run(rank, n_gpus, hps, args):
    global global_step
    if rank == 0:
        logger = utils.get_logger(hps.model_dir)
        #logger.info(hps)
        utils.check_git_hash(hps.model_dir)
        writer = SummaryWriter(log_dir=hps.model_dir)

    dist.init_process_group(backend='nccl',
                            init_method='env://',
                            world_size=n_gpus,
                            rank=rank)
    torch.manual_seed(hps.train.seed)
    torch.cuda.set_device(rank)

    train_file_list: List[dict] = get_dataset_filelist()
    train_set = AudioSet(train_file_list, 22050, 22050 * 4, hps.data.filter_length, hps.data.hop_length, hps.data.win_length, args.initial_run and rank==0)
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
        sampler=train_sampler,
        )
    if args.initial_run:
        if rank!=0:
            print(f'rank: {rank} is waiting...')
        dist.barrier()
        if rank==0:
            logger.info('Training Started')

    net_g = SynthesizerTrn(len(clas_dict),
                           hps.data.filter_length // 2 + 1,
                           hps.train.segment_size // hps.data.hop_length,
                           **hps.model).cuda(rank)
    net_g = DDP(net_g, device_ids=[rank])

    _, _, _, epoch_str = utils.load_checkpoint(
        utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), net_g)
    global_step = (epoch_str - 1) * len(train_loader)


    label_dict = dict({0: "DogBark", 1: "Footstep", 2: "GunShot", 3: "Keyboard",
                       4: "MovingMotorVehicle", 5: "Rain", 6: "SneezeCough"})

    generated_dir = os.path.join(hps.model_dir, f"generated_epoch{epoch_str}")
    generated_dirs = [os.path.join(generated_dir, f"{label_dict[class_idx]}") for class_idx in range(7)]
    for directory in generated_dirs:
        if not os.path.isdir(directory):
            os.makedirs(directory, exist_ok=True)


    for audio_idx in range(args.n_audio):
        audios = evaluate(net_g)
        for class_idx in range(7):
            audio = audios[class_idx].squeeze(0).cpu().numpy()
            sf.write(os.path.join(generated_dirs[class_idx], f"generated_{audio_idx}.wav"),
                     audio, hps.data.sampling_rate)


def evaluate(generator):
    generator.eval()
    with torch.no_grad():
        x = torch.arange(len(clas_dict), device=generator.device)
        y_hat = generator.module.infer(x, 344)
        y_hat_lengths = 344

    return y_hat


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c',
                        '--config',
                        type=str,
                        default="./configs/default.yaml",
                        help='Path to configuration file')
    parser.add_argument('-m',
                        '--model',
                        type=str,
                        required=True,
                        help='Model name')
    parser.add_argument('-r',
                        '--resume',
                        type=str,
                        help='Path to checkpoint for resume')
    parser.add_argument('-f',
                        '--force_resume',
                        type=str,
                        help='Path to checkpoint for force resume')
    parser.add_argument('-t',
                        '--transfer',
                        type=str,
                        help='Path to baseline checkpoint for transfer')
    parser.add_argument('-w',
                        '--ignore_warning',
                        action="store_true",
                        help='Ignore warning message')
    parser.add_argument('-i',
                        '--initial_run',
                        action="store_true",
                        help='Inintial run for saving pt files')
    parser.add_argument('-n',
                        '--n_audio',
                        type=int,
                        default=100,
                        help='number of audio files generated')
    args = parser.parse_args()
    if args.ignore_warning:
        import warnings
        warnings.filterwarnings(action='ignore')

    main(args)
