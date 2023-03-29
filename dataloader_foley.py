import os
import torch
import librosa
from scipy.io.wavfile import read as loadwav
import numpy as np
from typing import List

from torch.utils.data import Dataset, DataLoader
from collections import namedtuple

import warnings

from mel_processing import spec_to_mel_torch, spectrogram_torch
warnings.filterwarnings("ignore")

MAX_WAV_VALUE = 32768.0
""" Mel-Spectrogram extraction code from Turab ood_audio"""

# def mel_spectrogram(audio, n_fft, n_mels, hop_length, sample_rate):
#     # Compute mel-scaled spectrogram
#     mel_fb = librosa.filters.mel(sr=sample_rate, n_fft=n_fft, n_mels=n_mels)
#     spec = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
#     mel = np.dot(mel_fb, np.abs(spec))
#
#     # return librosa.power_to_db(mel, ref=0., top_db=None)
#     return np.log(mel + 1e-9)
""" Mel-Spectrogram extraction code from HiFi-GAN meldataset.py"""

CodeRow = namedtuple('CodeRow', ['bottom', 'class_id', 'salience', 'filename'])
clas_dict: dict = {
    "DogBark": 0,
    "Footstep": 1,
    "GunShot": 2,
    "Keyboard": 3,
    "MovingMotorVehicle": 4,
    "Rain": 5,
    "Sneeze_Cough": 6,
}


def get_dataset_filelist() -> list:
    training_files: List[dict] = list()
    for root_dir, _, file_list in os.walk("/foley_home/DCASEFoleySoundSynthesisDevSet"):
        for file_name in file_list:
            if os.path.splitext(file_name)[-1] == ".wav":
                training_files.append({
                    "class_id":
                    clas_dict[root_dir.split("/")[-1]],
                    "file_path":
                    f"{root_dir}/{file_name}",
                })
                # training_files.append((clas_dict[root_dir.split("/")[-1]],root_dir.split("/")[-2],))
    return training_files


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


mel_basis = {}
hann_window = {}

class AudioSet(Dataset):

    def __init__(
        self,
        audio_files,
        sample_rate,
        max_length,
        filter_length,
        hop_length,
        win_length,
        pt_run=False,
    ):
        self.audio_files = audio_files
        self.max_length = max_length  # max length of audio
        self.sample_rate = sample_rate
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        if pt_run:
            for i in range(len(self.audio_files)):
                _ = self.__getitem__(i, True)

    def __getitem__(self, index, pt_run =False):
        filename = self.audio_files[index]['file_path']
        class_id = self.audio_files[index][
            'class_id']  # datasets.get_class_id(filename)
        salience = 1  # datasets.get_salience(filename)

        sample_rate, audio = loadwav(filename)
        audio = audio / MAX_WAV_VALUE
        audio = (audio / ((np.abs(audio)+1e-6).max()) ) * 0.95
        
        if sample_rate != self.sample_rate:
            raise ValueError("{} sr doesn't match {} sr ".format(
                sample_rate, self.sample_rate))

        if len(audio) > self.max_length:
            # raise ValueError("{} length overflow".format(filename))
            audio = audio[0:self.max_length]

        # pad audio to max length, 4s for Urbansound8k dataset
        if len(audio) < self.max_length:
            # audio = torch.nn.functional.pad(audio, (0, self.max_length - audio.size(1)), 'constant')
            audio = np.pad(audio, (0, self.max_length - len(audio)),
                           'constant')
        audio = torch.tensor(audio, dtype=torch.float).unsqueeze(0)
        spec_filename = filename.replace(".wav", ".spec.pt")
        if os.path.exists(spec_filename) and not pt_run:
            spec = torch.load(spec_filename, map_location='cpu')
        else:
            spec = spectrogram_torch(
                    audio, 
                    self.filter_length,
                    self.sample_rate, 
                    self.hop_length, 
                    self.win_length,
                    center=False
                )

            spec = torch.squeeze(spec, 0)
            torch.save(spec, spec_filename)

        # mel = mel_spectrogram(audio, n_fft=self.n_fft, n_mels=self.n_mels, hop_length=self.hop_length, sample_rate=self.sample_rate)

        #mel_spec = mel_spectrogram_hifi(
        #    audio,
        #    n_fft=self.n_fft,
        #    n_mels=self.n_mels,
        #    hop_length=self.hop_length,
        #    sample_rate=self.sample_rate,
        #    fmin=self.fmin,
        #    fmax=self.fmax,
        #)

        # print(mel_spec.shape)
        return audio, spec.shape[-1], spec,  class_id

    def __len__(self):
        return len(self.audio_files)


def extract_flat_mel_from_Audio2Mel(Audio2Mel):
    mel = []

    for item in Audio2Mel:
        mel.append(item[0].flatten())

    return np.array(mel)


if __name__ == '__main__':
    train_file_list= get_dataset_filelist()

    print(train_file_list)

    train_set = AudioSet(train_file_list[0:2], 22050, 22050 * 4, 1024,256,1024)
    print(train_set[0])
    print(train_set[0][0].shape)
    train_loader = DataLoader(
        train_set,
        num_workers=16,
        shuffle=False,
        pin_memory=True,
        batch_size=  4,
    )
    for (x,xl,spec,id) in train_loader:
        print(x, xl,spec, id)
        exit()

class AudioCollate():
    """ Zero-pads model inputs and targets
    """

    def __init__(self, return_ids=False):
        self.return_ids = return_ids

    def __call__(self, batch):
        """Collate's training batch from normalized text and aduio
        PARAMS
        ------
        batch: [text_normalized, spec_normalized, wav_normalized]
        """
        # Right zero-pad all one-hot text sequences to max input length
        _, ids_sorted_decreasing = torch.sort(torch.LongTensor(
            [x[1].size(1) for x in batch]),
                                              dim=0,
                                              descending=True)

        max_text_len = max([len(x[0]) for x in batch])
        max_spec_len = max([x[1].size(1) for x in batch])
        max_wav_len = max([x[2].size(1) for x in batch])

        text_lengths = torch.LongTensor(len(batch))
        spec_lengths = torch.LongTensor(len(batch))
        wav_lengths = torch.LongTensor(len(batch))

        text_padded = torch.LongTensor(len(batch), max_text_len)
        spec_padded = torch.FloatTensor(len(batch), batch[0][1].size(0),
                                        max_spec_len)
        wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len)
        text_padded.zero_()
        spec_padded.zero_()
        wav_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            text = row[0]
            text_padded[i, :text.size(0)] = text
            text_lengths[i] = text.size(0)

            spec = row[1]
            spec_padded[i, :, :spec.size(1)] = spec
            spec_lengths[i] = spec.size(1)

            wav = row[2]
            wav_padded[i, :, :wav.size(1)] = wav
            wav_lengths[i] = wav.size(1)

        if self.return_ids:
            return text_padded, text_lengths, spec_padded, spec_lengths, wav_padded, wav_lengths, ids_sorted_decreasing
        return text_padded, text_lengths, spec_padded, spec_lengths, wav_padded, wav_lengths


class DistributedBucketSampler(torch.utils.data.distributed.DistributedSampler
                               ):
    """
    Maintain similar input lengths in a batch.
    Length groups are specified by boundaries.
    Ex) boundaries = [b1, b2, b3] -> any batch is included either {x | b1 < length(x) <=b2} or {x | b2 < length(x) <= b3}.
  
    It removes samples which are not included in the boundaries.
    Ex) boundaries = [b1, b2, b3] -> any x s.t. length(x) <= b1 or length(x) > b3 are discarded.
    """

    def __init__(self,
                 dataset,
                 batch_size,
                 boundaries,
                 num_replicas=None,
                 rank=None,
                 shuffle=True):
        super().__init__(dataset,
                         num_replicas=num_replicas,
                         rank=rank,
                         shuffle=shuffle)
        self.lengths = len(dataset)
        self.batch_size = batch_size
        self.boundaries = boundaries

        self.buckets, self.num_samples_per_bucket = self._create_buckets()
        self.total_size = sum(self.num_samples_per_bucket)
        self.num_samples = self.total_size // self.num_replicas

    def _create_buckets(self):
        buckets = [[] for _ in range(len(self.boundaries) - 1)]
        for i in range(self.lengths):
            length = 344
            idx_bucket = self._bisect(length)
            if idx_bucket != -1:
                buckets[idx_bucket].append(i)

        for i in range(len(buckets) - 1, 0, -1):
            if len(buckets[i]) == 0:
                buckets.pop(i)
                self.boundaries.pop(i + 1)

        num_samples_per_bucket = []
        for i in range(len(buckets)):
            len_bucket = len(buckets[i])
            total_batch_size = self.num_replicas * self.batch_size
            rem = (total_batch_size -
                   (len_bucket % total_batch_size)) % total_batch_size
            num_samples_per_bucket.append(len_bucket + rem)
        return buckets, num_samples_per_bucket

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)

        indices = []
        if self.shuffle:
            for bucket in self.buckets:
                indices.append(
                    torch.randperm(len(bucket), generator=g).tolist())
        else:
            for bucket in self.buckets:
                indices.append(list(range(len(bucket))))

        batches = []
        for i in range(len(self.buckets)):
            bucket = self.buckets[i]
            len_bucket = len(bucket)
            ids_bucket = indices[i]
            num_samples_bucket = self.num_samples_per_bucket[i]

            # add extra samples to make it evenly divisible
            rem = num_samples_bucket - len_bucket
            ids_bucket = ids_bucket + ids_bucket * (
                rem // len_bucket) + ids_bucket[:(rem % len_bucket)]

            # subsample
            ids_bucket = ids_bucket[self.rank::self.num_replicas]

            # batching
            for j in range(len(ids_bucket) // self.batch_size):
                batch = [
                    bucket[idx]
                    for idx in ids_bucket[j * self.batch_size:(j + 1) *
                                          self.batch_size]
                ]
                batches.append(batch)

        if self.shuffle:
            batch_ids = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[i] for i in batch_ids]
        self.batches = batches

        assert len(self.batches) * self.batch_size == self.num_samples
        return iter(self.batches)

    def _bisect(self, x, lo=0, hi=None):
        if hi is None:
            hi = len(self.boundaries) - 1

        if hi > lo:
            mid = (hi + lo) // 2
            if self.boundaries[mid] < x and x <= self.boundaries[mid + 1]:
                return mid
            elif x <= self.boundaries[mid]:
                return self._bisect(x, lo, mid)
            else:
                return self._bisect(x, mid + 1, hi)
        else:
            return -1

    def __len__(self):
        return self.num_samples // self.batch_size
