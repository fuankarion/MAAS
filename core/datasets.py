import os
import math
import torch
import random

import numpy as np
import core.io as io
import core.clip_utils as cu

from torch.utils import data
from core.util import csv_to_list, postprocess_speech_label, postprocess_entity_label
from core.augmentations import video_temporal_crop


class CachedAVSource(data.Dataset):
    def __init__(self):
        # Cached data
        self.entity_data = {}
        self.speech_data = {}
        self.entity_list = []

        # Reproducibilty
        torch.manual_seed(11)
        random.seed(33)
        np.random.seed(44)

    def _cache_entity_data(self, csv_file_path):
        entity_set = set()

        csv_data = csv_to_list(csv_file_path)
        csv_data.pop(0)  # CSV header
        for csv_row in csv_data:
            video_id = csv_row[0]
            entity_id = csv_row[-3]
            timestamp = csv_row[1]

            speech_label = postprocess_speech_label(csv_row[-2])
            entity_label = postprocess_entity_label(csv_row[-2])
            minimal_entity_data = (entity_id, timestamp, entity_label)

            # Store minimal entity data
            if video_id not in self.entity_data.keys():
                self.entity_data[video_id] = {}
            if entity_id not in self.entity_data[video_id].keys():
                self.entity_data[video_id][entity_id] = []
                entity_set.add((video_id, entity_id))
            self.entity_data[video_id][entity_id].append(minimal_entity_data)

            # Store speech meta-data
            if video_id not in self.speech_data.keys():
                self.speech_data[video_id] = {}
            if timestamp not in self.speech_data[video_id].keys():
                self.speech_data[video_id][timestamp] = speech_label

            # Max operation yields if someone is speaking.
            new_speech_label = max(self.speech_data[video_id][timestamp], speech_label)
            self.speech_data[video_id][timestamp] = new_speech_label

        return entity_set

    def _cache_entity_data_forward(self, csv_file_path, target_video):
        entity_list = list()

        csv_data = csv_to_list(csv_file_path)
        csv_data.pop(0)  # CSV header
        for csv_row in csv_data:
            video_id = csv_row[0]
            if video_id != target_video:
                continue

            entity_id = csv_row[-3]
            timestamp = csv_row[1]
            entity_label = postprocess_entity_label(csv_row[-2])

            entity_list.append((video_id, entity_id, timestamp))
            minimal_entity_data = (entity_id, timestamp, entity_label) # safe to ingore label here

            if video_id not in self.entity_data.keys():
                self.entity_data[video_id] = {}

            if entity_id not in self.entity_data[video_id].keys():
                self.entity_data[video_id][entity_id] = []
            self.entity_data[video_id][entity_id].append(minimal_entity_data)

        return entity_list

    def _entity_list_postprocessing(self, entity_set):
        print('Initial', len(entity_set))

        # filter out missing data on disk
        all_disk_data = set(os.listdir(self.video_root))
        for video_id, entity_id in entity_set.copy():
            if entity_id not in all_disk_data:
                entity_set.remove((video_id, entity_id))
        print('Pruned not in disk', len(entity_set))

        for video_id, entity_id in entity_set.copy():
            dir = os.path.join(self.video_root, entity_id)
            if len(os.listdir(dir)) != len(self.entity_data[video_id][entity_id]):
                entity_set.remove((video_id, entity_id))

        print('Pruned not complete', len(entity_set))
        self.entity_list = sorted(list(entity_set))


class STEDataset(CachedAVSource):
    def __init__(self, audio_root, video_root, csv_file_path, clip_lenght,
                 video_transform=None, do_video_augment=False, crop_ratio=0.8):
        super().__init__()

        # Data directories
        self.audio_root = audio_root
        self.video_root = video_root

        # Post-processing
        self.video_transform = video_transform
        self.do_video_augment = do_video_augment
        self.crop_ratio = crop_ratio

        # Clip arguments
        self.clip_lenght = clip_lenght
        self.half_clip_length = math.floor(self.clip_lenght/2)

        entity_set = self._cache_entity_data(csv_file_path)
        self._entity_list_postprocessing(entity_set)

    def __len__(self):
        return int(len(self.entity_list)/1)

    def __getitem__(self, index):
        # Get meta-data
        video_id, entity_id = self.entity_list[index]
        entity_metadata = self.entity_data[video_id][entity_id]

        audio_offset = float(entity_metadata[0][1])
        mid_index = random.randint(0, len(entity_metadata)-1)
        midone = entity_metadata[mid_index]
        target = int(midone[-1])
        target_audio = self.speech_data[video_id][midone[1]]

        clip_meta_data = cu.generate_clip_meta(entity_metadata, mid_index,
                                               self.half_clip_length)
        video_data, audio_data = io.load_av_clip_from_metadata(clip_meta_data,
                                 self.video_root, self.audio_root, audio_offset)

        if self.do_video_augment:
            video_data = video_temporal_crop(video_data, self.crop_ratio)

        if self.video_transform is not None:
            video_data = [self.video_transform(vd) for vd in video_data]

        video_data = torch.cat(video_data, dim=0)
        return (np.float32(audio_data), video_data), target, target_audio
