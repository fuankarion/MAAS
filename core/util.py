import os
import csv
import json
import python_speech_features

import numpy as np
import torch.nn as nn


def postprocess_speech_label(speech_label):
    speech_label = int(speech_label)
    if speech_label == 2:  # Remember 2 = SPEAKING_NOT_AUDIBLE
        speech_label = 0
    return speech_label


def postprocess_entity_label(entity_label):
    entity_label = int(entity_label)
    if entity_label == 2:  # Remember 2 = SPEAKING_NOT_AUDIBLE
        entity_label = 0
    return entity_label


class Logger():
    def __init__(self, targetFile, separator=';'):
        self.targetFile = targetFile
        self.separator = separator

    def writeHeaders(self, headers):
        with open(self.targetFile, 'a') as fh:
            for aHeader in headers:
                fh.write(aHeader + self.separator)
            fh.write('\n')

    def writeDataLog(self, dataArray):
        with open(self.targetFile, 'a') as fh:
            for dataItem in dataArray:
                fh.write(str(dataItem) + self.separator)
            fh.write('\n')


def freeze_model_bn(model):
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            if hasattr(module, 'weight'):
                module.weight.requires_grad_(False)
            if hasattr(module, 'bias'):
                module.bias.requires_grad_(False)
            module.eval()


def csv_to_list(csv_path):
    as_list = None
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        as_list = list(reader)
    return as_list


def generate_mel_spectrogram(audio_clip, sample_rate):
    mfcc = zip(*python_speech_features.mfcc(audio_clip, sample_rate))
    audio_features = np.stack([np.array(i) for i in mfcc])
    audio_features = np.expand_dims(audio_features, axis=0)
    return audio_features


def configure_backbone(backbone, size, pretrained_arg=True, d_ratio=0.0, num_classes_arg=2):
    return backbone(pretrained=pretrained_arg, rgb_stack_size=size,
                    num_classes=num_classes_arg)


def configure_backbone_forward_phase(backbone, pretrained_weights_path, size, pretrained_arg=True, num_classes_arg=2):
    return backbone(pretrained_weights_path, rgb_stack_size=size,
                    num_classes=num_classes_arg)


def set_up_log_and_ws_out(models_out, opt_config, experiment_name, headers=None):
    target_logs = os.path.join(models_out, experiment_name + '/logs.csv')
    target_models = os.path.join(models_out, experiment_name)
    print('target_models', target_models)
    if not os.path.isdir(target_models):
        os.makedirs(target_models)
    log = Logger(target_logs, ';')

    if headers is None:
        log.writeHeaders(['epoch', 'train_loss', 'train_audio_loss',
                          'train_video_loss', 'train_map',
                          'val_loss', 'val_audio_loss', 'val_video_loss',
                          'val_map'])
    else:
        log.writeHeaders(headers)

    # Dump cfg to json
    dump_cfg = opt_config.copy()
    for key, value in dump_cfg.items():
        if callable(value):
            try:
                dump_cfg[key] = value.__name__
            except:
                dump_cfg[key] = 'CrossEntropyLoss'
    json_cfg = os.path.join(models_out, experiment_name+'/cfg.json')
    with open(json_cfg, 'w') as json_file:
      json.dump(dump_cfg, json_file)

    models_out = os.path.join(models_out, experiment_name)
    return log, models_out


def load_ava_test_video_set():
    files = os.listdir('/Dataset/ava_active_speaker/csv/gt/challenge')
    videos = [f[:-18] for f in files]
    videos.sort()
    return videos


def load_ava_val_video_set():
    files = os.listdir('/Dataset/ava_active_speaker/csv/gt/ava_activespeaker_test_v1.0')
    videos = [f[:-18] for f in files]
    videos.sort()
    return videos


def load_ava_train_video_set():
    files = os.listdir('/Dataset/ava_active_speaker/csv/gt/ava_activespeaker_train_v1.0')
    videos = [f[:-18] for f in files]
    videos.sort()
    return videos
