#!/usr/bin/env python3

import logging
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, FileType
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import IterableDataset
import torchaudio
import pandas as pd
import numpy as np
from collections import defaultdict
import yaml
import random
import time
import math
import os


def split_evenly(source, keyfunc, frac):
    '''
    Split a sequence into two, in proportions specified by frac,
    such that each set grouped by keyfunc is represented proportionally
    in each split. This is used to partition a data set into train and test
    sets, while preserving the proportion of each class. 
    '''
    def _split_list(s, frac):
        s = list(s)
        random.shuffle(s)
        point = int(frac * len(s))
        logging.debug(f'split: {point}, {len(s) - point}')
        return s[:point], s[point:]

    m = defaultdict(set)
    for item in source:
        m[keyfunc(item)].add(item)

    x = set([])
    y = set([])
    for v in m.values():
        x1, y1 = _split_list(v, frac)
        x.update(x1)
        y.update(y1)
    logging.debug(f'split total: {len(x)}, {len(y)}')
    return list(x), list(y)

# assumptions:
#   all inputs have the same sampling frequency
class AudioDataset(IterableDataset):
    def __init__(self, songs, music_dir, segment_length, downsample_rate):
        super(AudioDataset, self).__init__()
        self.songs = songs
        self.shuffle()
        self.music_dir = music_dir
        self.segment_length = segment_length # audio samples per datum
        self.downsample_rate = downsample_rate

        labels = sorted(list(set(s.label for s in songs)))   # all the labels
        self.label_number_map = {v:i for i,v in enumerate(labels)}   # {'foo':0, 'bar':1, ...}

    def num_classes(self):
        return len(self.label_number_map.keys())

    def shuffle(self):
        random.shuffle(self.songs)

    def whole_songs(self):
        '''yield audio segments batched by song'''
        for song in self.songs:
            data = torch.stack(list(song.audio_segments(self.music_dir,
                                                   self.segment_length, 
                                                   self.downsample_rate)))
            yield data, self.label_number_map[song.label]

    def __iter__(self):
        '''yield audio segments, one at a time'''
        start = 0
        stop = len(self.songs)
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            per_worker = math.ceil(len(self.songs) / worker_info.num_workers)
            start = worker_info.id * per_worker
            stop = min(start + per_worker, stop)
            logging.debug(f'Worker {worker_info.id} will get {start}-{stop}')
        for i in range(start, stop):
            song = self.songs[i]
            for segment in song.audio_segments(self.music_dir,
                                               self.segment_length, 
                                               self.downsample_rate):
                yield segment, self.label_number_map[song.label]


class Song():
    '''Each instance holds the metadata from one song and can provide
    audio data on demand.'''
    def __init__(self, fields):
        self.fields = fields
        self.__dict__.update(fields)

    def __repr__(self):
        return 'Song<{}>'.format(str(list(self.fields.values())))

    def audio_segments(self, music_dir, segment_length, downsample_rate):
        # Caching proved not to be worthwhile when loading from wav files.
        # mp3 files are much slower to load, but rather than cache audio,
        # it's better to convert the data set in advance.
        (audio, samplingfreq) = torchaudio.load(os.path.join(music_dir, self.filename))
        audio = audio.permute(1, 0) # 1×N ~ N×1
        # downsample, and keep the amount amount of data specified
        audio = audio[::downsample_rate]
        num_segments = int(len(audio) / segment_length)
        #logging.debug(f'{self.title} has {num_segments} segments')
        for i in range(num_segments):
            yield audio[i * segment_length: (i+1) * segment_length].permute(1, 0)

class ConvBn(nn.Module):
    '''A convenience class for setting up Conv1d + BatchNorm1d components'''
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, *args, **kwargs)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))

class ManyConvMaxPool(nn.Module):
    '''Set up multiple ConvBn + MaxPool1d components'''
    def __init__(self, conv_count, maxpool_kernel, in_channels, out_channels, *args, **kwargs):
        super().__init__()
        self.pool = nn.MaxPool1d(maxpool_kernel)
        self.conv_layers = nn.ModuleList([ConvBn(in_channels, out_channels, *args, **kwargs)])
        for _ in range(conv_count - 1):
            self.conv_layers.append(ConvBn(out_channels, out_channels, *args, **kwargs))

    def forward(self, x):
        for layer in self.conv_layers:
            x = layer.forward(x)
        return self.pool(x)

class Net(nn.Module):
    '''Our main network'''
    def __init__(self, num_classes):
        super(Net, self).__init__()
        # M11 network from https://arxiv.org/pdf/1610.00087.pdf
        self.layers = nn.ModuleList([ManyConvMaxPool(1, 4, 1, 64, 80, stride=4),
                                     ManyConvMaxPool(2, 4, 64, 64, 3),
                                     ManyConvMaxPool(2, 4, 64, 128, 3),
                                     ManyConvMaxPool(3, 4, 128, 256, 3),
                                     ManyConvMaxPool(2, 4, 256, 512, 3)])
        self.avgPool = nn.AvgPool1d(6)
        self.fc1 = nn.Linear(512, num_classes)

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        x = self.avgPool(x)
        x = x.permute(0, 2, 1) #change the 512x1 to 1x512
        x = self.fc1(x)
        return F.log_softmax(x, dim = 2)


class Thing():
    def __init__(self, log_interval):
        self.log_interval = log_interval
        self.test_results = []

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.debug(f'Device: {self.device}')
        if torch.cuda.is_available():
            logging.debug(f'we have {torch.cuda.device_count()} GPU(s)')


    def train(self, epoch):
        self.model.train()
        self.train_set.shuffle()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            data = data.to(self.device)
            target = target.to(self.device)
            data = data.requires_grad_() #set requires_grad to True for training
            output = self.model(data)
            output = output.permute(1, 0, 2) #original output dimensions are batchSizex1x10 
            loss = F.nll_loss(output[0], target) #the loss functions expects a batchSizex10 input
            loss.backward()
            self.optimizer.step()
            if batch_idx % self.log_interval == 0: #print training stats
                logging.info(f'Train Epoch: {epoch} \tLoss: {loss}')

    def test(self, epoch):
        self.test_set.shuffle()
        self.model.eval()
        correct = 0
        total_predictions = 0
        for data, target in self.test_loader:
            data = data.to(self.device)
            target = target.to(self.device)
            output = self.model(data)
            output = output.permute(1, 0, 2)   # to match target
            pred = output.max(2)[1]  # get the index of the max log-probability
            total_predictions += max(pred.size())
            correct += pred.eq(target).cpu().sum().item()
        accuracy = correct / total_predictions
        logging.info('Test set: Accuracy: {}/{} ({:.0f}%)'.format(
            correct, total_predictions, 100. * accuracy))
        return accuracy

    def test_whole_songs(self, epoch):
        self.model.eval()
        correct = 0
        total_predictions = 0
        for data, target in self.test_set.whole_songs():
            data = data.to(self.device)
            output = self.model(data)
            output = output.permute(1, 0, 2)
            predictions = output.max(2)[1]
            winner = predictions.mode()[0].item()
            votes = predictions.eq(predictions.mode()[0].item()).sum().item()
            total_predictions += 1
            if target == winner:
                correct += 1
                logging.debug(f'correct with {votes} of {output.size(1)} votes')
            else:
                logging.debug(f'wrong with {votes} of {output.size(1)} votes')
        accuracy = correct / total_predictions
        logging.info('Test set: Accuracy: {}/{} ({:.0f}%)'.format(
            correct, total_predictions, 100. * accuracy))
        return accuracy

    def go(self, epochs):
        start_time = time.time()
        for epoch in range(1, epochs + 1):
            epoch_start_time = time.time()
            self.train(epoch)
            self.test_results.append(self.test_whole_songs(epoch))
            self.scheduler.step()
            logging.info('Epoch {}, epoch time: {}, total time: {} seconds'.format(epoch,
                                                                                   int(time.time() - epoch_start_time),
                                                                                   int(time.time() - start_time)))


def main():
    description = ''
    parser = ArgumentParser(description=description,
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--epochs', type=int, default=80,
                        help='training epochs')
    parser.add_argument('-l', '--segment_length', type=int, default=32000,
                        help='samples per segment')
    parser.add_argument('-r', '--downsample_rate', type=int, default=4,
                        help='downsample the audio by this')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('music_dir', help='directory containing music files')
    parser.add_argument('metadata', type=FileType('r'),
                        help='yaml file containing song metadata')
    args = parser.parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)


    train_songs, test_songs = split_evenly((Song(s) for s in yaml.safe_load(args.metadata)),
                                           lambda s: s.label, 0.9)

    thing = Thing(log_interval = 20)
    thing.train_set = AudioDataset(train_songs, args.music_dir,
                                   args.segment_length, args.downsample_rate)
    thing.test_set = AudioDataset(test_songs, args.music_dir,
                                  args.segment_length, args.downsample_rate)
    logging.info(f'Train set size: {len(thing.train_set.songs)} songs')
    logging.info(f'Test set size: {len(thing.test_set.songs)} songs')

    if thing.device == torch.device('cuda'):
        kwargs = {'num_workers': 2, 'pin_memory': True}
    else:
        kwargs = {'num_workers': 4}
    thing.train_loader = torch.utils.data.DataLoader(thing.train_set, batch_size = 128, **kwargs)
    thing.test_loader = torch.utils.data.DataLoader(thing.test_set, batch_size = 128, **kwargs)

    thing.model = Net(thing.train_set.num_classes())
    thing.model.to(thing.device)
    
    thing.optimizer = optim.Adam(thing.model.parameters())
    thing.scheduler = optim.lr_scheduler.StepLR(thing.optimizer,
                                                step_size = int(args.epochs*2/3),
                                                gamma = 0.1)

    thing.go(args.epochs)

    if args.plot:
        import matplotlib.pyplot as plt
        plt.plot(thing.test_results)
        plt.show()


if __name__ == '__main__':
    main()

