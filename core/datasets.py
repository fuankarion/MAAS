import os
import math
import glob
import torch
import random

import numpy as np
import core.io as io
import core.clip_utils as cu
import multiprocessing as mp

from torch.utils import data
from torch_geometric.data import Data
from core.augmentations import video_temporal_crop
from core.util import csv_to_list, postprocess_speech_label, postprocess_entity_label


# Super Classes, dont instantiate
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


class ContextualDataset(data.Dataset):
    def get_speaker_context(self, ts_to_entity, video_id, target_entity_id,
                            center_ts, candidate_speakers):
        context_entities = list(ts_to_entity[video_id][center_ts])
        random.shuffle(context_entities)
        context_entities.remove(target_entity_id)

        if not context_entities:  # nos mamamos la lista
            context_entities.insert(0, target_entity_id)  # make sure is at 0
            while len(context_entities) < candidate_speakers:
                context_entities.append(random.choice(context_entities))
        elif len(context_entities) < candidate_speakers:
            context_entities.insert(0, target_entity_id)  # make sure is at 0
            while len(context_entities) < candidate_speakers:
                context_entities.append(random.choice(context_entities[1:]))
        else:
            context_entities.insert(0, target_entity_id)  # make sure is at 0
            context_entities = context_entities[:candidate_speakers]

        return context_entities

    def get_simple_speaker_context(self, ts_to_entity, video_id, center_ts):
        context_entities = list(ts_to_entity[video_id][center_ts])

        random.shuffle(context_entities)
        return context_entities

    def _decode_feature_data_from_csv(self, feature_data):
        feature_data = feature_data[1:-1]
        feature_data = feature_data.split(',')
        return np.asarray([float(fd) for fd in feature_data])

    def get_time_context(self, entity_data, video_id, target_entity_id,
                         center_ts, half_time_length, stride):
        all_ts = list(entity_data[video_id][target_entity_id].keys())
        center_ts_idx = all_ts.index(str(center_ts))

        start = center_ts_idx-(half_time_length*stride)
        end = center_ts_idx+((half_time_length+1)*stride)
        selected_ts_idx = list(range(start, end, stride))

        selected_ts = []
        for i, idx in enumerate(selected_ts_idx):
            if idx < 0:
                idx = 0
            if idx >= len(all_ts):
                idx = len(all_ts)-1
            selected_ts.append(all_ts[idx])

        return selected_ts

    def get_time_indexed_feature(self, video_id, entity_id, selected_ts):
        time_features = []
        for ts in selected_ts:
            time_features.append(self.entity_data[video_id][entity_id][ts][0])
        return np.asarray(time_features)

    def get_time_indexed_labels(self, video_id, entity_id, selected_ts):
        time_features = []
        for ts in selected_ts:
            time_features.append(self.entity_data[video_id][entity_id][ts][1])
        return np.asarray(time_features)

    def _cache_feature_file(self, csv_file):
        entity_data = {}
        feature_list = []
        ts_to_entity = {}

        print('load feature data', csv_file)
        csv_data = csv_to_list(csv_file)
        for csv_row in csv_data:
            video_id = csv_row[0]
            ts = csv_row[1]
            entity_id = csv_row[2]
            features_a = self._decode_feature_data_from_csv(csv_row[-2])
            features_v = self._decode_feature_data_from_csv(csv_row[-1])
            features = np.concatenate((features_a, features_v))
            label = int(float(csv_row[3]))

            # entity_data
            if video_id not in entity_data.keys():
                entity_data[video_id] = {}
            if entity_id not in entity_data[video_id].keys():
                entity_data[video_id][entity_id] = {}
            if ts not in entity_data[video_id][entity_id].keys():
                entity_data[video_id][entity_id][ts] = []
            entity_data[video_id][entity_id][ts] = (features, label)
            feature_list.append((video_id, entity_id, ts))

            # ts_to_entity
            if video_id not in ts_to_entity.keys():
                ts_to_entity[video_id] = {}
            if ts not in ts_to_entity[video_id].keys():
                ts_to_entity[video_id][ts] = []
            ts_to_entity[video_id][ts].append(entity_id)

        print('loaded ', len(feature_list), ' features from ', os.path.basename(csv_file))
        return entity_data, feature_list, ts_to_entity


####################
### STE Datasets ###
####################

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


class STEDatasetForward(CachedAVSource):
    def __init__(self, target_video, audio_root, video_root, csv_file_path,
                 clip_lenght, target_size, video_transform=None,
                 do_video_augment=False):
        super().__init__()

        # Data directories
        self.audio_root = audio_root
        self.video_root = video_root

        # Post-processing
        self.video_transform = video_transform
        self.do_video_augment = do_video_augment
        self.target_video = target_video

        # Clip arguments
        self.clip_lenght = clip_lenght
        self.half_clip_length = math.floor(self.clip_lenght/2)
        self.target_size = target_size

        self.entity_list = self._cache_entity_data_forward(csv_file_path, self.target_video )
        print('len(self.entity_list)', len(self.entity_list))

    def _where_is_ts(self, entity_metadata, ts):
        for idx, val in enumerate(entity_metadata):
            if val[1] == ts:
                return idx

        raise Exception('time stamp not found')

    def __len__(self):
        return int(len(self.entity_list))

    def __getitem__(self, index):
        # Get meta-data
        video_id, entity_id, ts = self.entity_list[index]
        entity_metadata = self.entity_data[video_id][entity_id]

        audio_offset = float(entity_metadata[0][1])
        mid_index = self._where_is_ts(entity_metadata, ts)
        midone = entity_metadata[mid_index]
        gt = midone[-1]

        clip_meta_data = cu.generate_clip_meta(entity_metadata, mid_index,
                                               self.half_clip_length)

        video_data, audio_data = io.load_av_clip_from_metadata_forward(clip_meta_data,
                                 self.video_root, self.audio_root, self.target_size, audio_offset)

        if self.do_video_augment:
            # random flip
            if bool(random.getrandbits(1)):
                video_data = [s.transpose(Image.FLIP_LEFT_RIGHT) for s in video_data]

            # random crop
            width, height = video_data[0].size
            f = random.uniform(0.5, 1)
            i, j, h, w = RandomCrop.get_params(video_data[0], output_size=(int(height*f), int(width*f)))
            video_data = [s.crop(box=(j, i, w, h)) for s in video_data]

        if self.video_transform is not None:
            video_data = [self.video_transform(vd) for vd in video_data]

        video_data = torch.cat(video_data, dim=0)
        return np.float32(audio_data), video_data, video_id, ts, entity_id, gt


######################
### MAAS Datasets ###
#####################

class MAASDataset(ContextualDataset):
    def __init__(self, csv_file_path, time_length, stride, candidate_speakers,
                 noise=0.0, augment_context=False, limit_videos=None):
        # Space config
        self.time_length = time_length
        self.stride = stride
        self.half_time_length = math.floor(self.time_length/2)
        self.candidate_speakers = candidate_speakers
        self.augment_context = augment_context

        self.noise = noise

        # In memory data
        self.feature_list = []
        self.ts_to_entity = {}
        self.entity_data = {}

        # Load metadata
        self._cache_feature_data(csv_file_path, limit_videos)

    # Parallel load of feature files
    def _cache_feature_data(self, dataset_dir, limit_videos):
        pool = mp.Pool(int(mp.cpu_count()))
        files = glob.glob(dataset_dir)
        files.sort()
        if limit_videos:
            files = files[:limit_videos]
        results = pool.map(self._cache_feature_file, files)
        pool.close()

        for r_set in results:
            e_data, f_list, ts_ent = r_set
            print('unpack ', len(f_list))
            self.entity_data.update(e_data)
            self.feature_list.extend(f_list)
            self.ts_to_entity.update(ts_ent)

    def __len__(self):
        return int(len(self.feature_list)/20)

    def __getitem__(self, index):
        video_id, target_entity_id, center_ts = self.feature_list[index]
        time_context = self.get_time_context(self.entity_data, video_id,
                                             target_entity_id, center_ts,
                                             self.half_time_length,
                                             self.stride)

        target_set = []
        feature_set = None

        src = []
        dst = []
        all_audio_nodes = []

        for tc in time_context:
            entity_context = self.get_speaker_context(self.ts_to_entity, video_id,
                                                      target_entity_id, tc,
                                                      self.candidate_speakers)
            feats, label = self.entity_data[video_id][target_entity_id][tc]

            # add audio feat
            target_set.append(label)
            if feature_set is None:
                feature_set = np.expand_dims(feats[:512], axis=0)
            else:
                a_feats = np.expand_dims(feats[:512], axis=0)
                a_feats = a_feats + np.random.normal(0, self.noise, a_feats.shape)
                feature_set = np.concatenate([feature_set, a_feats], axis=0)

            audio_node_idx = feature_set.shape[0]-1
            all_audio_nodes.append(audio_node_idx)

            for ctx_entity in entity_context:
                feats, label = self.entity_data[video_id][ctx_entity][tc]
                v_feats = np.expand_dims(feats[512:], axis=0)
                v_feats = v_feats + np.random.normal(0, self.noise, v_feats.shape)
                feature_set = np.concatenate([feature_set, v_feats], axis=0)
                target_set.append(label)
                video_node_idx = feature_set.shape[0]-1

                # mini assignation graph links
                src.extend([all_audio_nodes[-1], video_node_idx, video_node_idx])
                dst.extend([video_node_idx, all_audio_nodes[-1], video_node_idx])

        # audio links
        for an_s in all_audio_nodes:
            for an_d in all_audio_nodes:
                src.append(an_s)
                dst.append(an_d)

        # video links
        for s in range(1, self.candidate_speakers+1):
            for d in range(self.time_length):
                src.append(s)
                dst.append(s+((self.candidate_speakers+1)*d))

        batch_edges = torch.tensor([src, dst], dtype=torch.long)
        return Data(x=torch.tensor(feature_set, dtype=torch.float), edge_index=batch_edges, y=torch.tensor(target_set))
