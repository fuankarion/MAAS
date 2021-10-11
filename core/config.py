import torch.nn as nn
import torch.optim as optim
import core.models_ste as ste


# STE Train
STE_inputs = {
    # input files
    'csv_train_full': '/Dataset/ava_active_speaker/csv/ava_activespeaker_train_augmented.csv',
    'csv_val_full': '/Dataset/ava_active_speaker/csv/ava_activespeaker_val_augmented.csv',
    'csv_test_full': '/Dataset/ava_active_speaker/csv/ava_activespeaker_test_augmented.csv',

    # Data config
    'audio_dir': '/Dataset/ava_active_speaker/instance_wavs_time/',
    'video_dir': '/Dataset/ava_active_speaker/instance_crops_time/',
    'models_out': '/home/alcazajl/Models/ASC2/ste/'
}
STE_2D_optimization_params = {
    # Net Arch
    'backbone': ste.resnet18_two_streams,

    # Optimization config
    'optimizer': optim.Adam,
    'criterion': nn.CrossEntropyLoss(),
    'learning_rate': 5e-4,
    'epochs': 55,
    'step_size': 30,
    'gamma': 0.1,

    # Batch Config
    'batch_size': 128,
    'threads': 12
}


# Forward STE
STE_inputs_forward = {
    # input files
    'csv_train_full': '/Dataset/ava_active_speaker/csv/ava_activespeaker_train_augmented.csv',
    'csv_val_full': '/Dataset/ava_active_speaker/csv/ava_activespeaker_val_augmented.csv',
    'csv_test_full': '/Dataset/ava_active_speaker/csv/ava_activespeaker_test_augmented.csv',

    # Data config
    'audio_dir': '/Dataset/ava_active_speaker/instance_wavs_time/',
    'video_dir': '/Dataset/ava_active_speaker/instance_crops_time/',
}
STE_forward_params = {
    # Net Arch
    'backbone': ste.resnet18_two_streams_forward,

    # Batch Config
    'batch_size': 1,
    'threads': 1
}


# MAAS Module Train
MAAS_inputs = {
    # input files
    'features_train_full': '/home/alcazajl/Forwards/ICCV/ste11_train//*.csv',
    'features_val_full': '/home/alcazajl/Forwards/ICCV/ste11_val/*.csv',

    # Data config
    'models_out': '/home/alcazajl/Models/ASC2/maas/'
}
MAAS_optimization_params = {
    # Optimizer config
    'optimizer': optim.Adam,
    'criterion': nn.CrossEntropyLoss(),
    'learning_rate': 3e-4,
    'epochs': 5,
    'step_size': 7,
    'gamma': 0.1,

    # Batch Config
    'batch_size': 1024,
    'threads': 15
}
