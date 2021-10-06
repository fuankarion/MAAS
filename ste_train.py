import os
import sys
import torch

from torchvision import transforms
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from core.optimization_ste import optimize_ste_losses
from core.short_term_dataset import STEDataset
from core.util import configure_backbone, set_up_log_and_ws_out

import core.config as exp_conf
import core.custom_transforms as ct


if __name__ == '__main__':
    # Reproducibility
    torch.manual_seed(11)
    torch.cuda.manual_seed(22)
    torch.backends.cudnn.deterministic = True

    # Experiments arguments
    cuda_device_number = str(sys.argv[1])
    clip_lenght = int(sys.argv[2])
    # Change size if needed, larger images perfrom sliglty better
    image_size = (160, 160)

    # check these 3 are in order, everythine else is kind of automated
    model_name = 'ste_res18_clip'+str(clip_lenght)
    io_config = exp_conf.STE_inputs
    opt_config = exp_conf.STE_2D_optimization_params

    # Transforms
    video_train_transform = transforms.Compose([transforms.Resize(image_size), ct.video_train])
    video_val_transform = transforms.Compose([transforms.Resize(image_size), ct.video_val])

    # output config
    log, target_models = set_up_log_and_ws_out(io_config['models_out'],
                                               opt_config, model_name)

    # cuda config
    backbone = configure_backbone(opt_config['backbone'], clip_lenght)
    has_cuda = torch.cuda.is_available()
    device = torch.device('cuda:'+cuda_device_number if has_cuda else 'cpu')
    print('has_cuda', has_cuda)
    print('device', device)
    backbone = backbone.to(device)

    # Optimization config
    criterion = opt_config['criterion']
    optimizer = opt_config['optimizer'](backbone.parameters(),
                                        lr=opt_config['learning_rate'])
    scheduler = lr_scheduler.StepLR(optimizer, step_size=opt_config['step_size'],
                                    gamma=opt_config['gamma'])

    video_train_path = os.path.join(io_config['video_dir'], 'train')
    audio_train_path = os.path.join(io_config['audio_dir'], 'train')
    video_val_path = os.path.join(io_config['video_dir'], 'val')
    audio_val_path = os.path.join(io_config['audio_dir'], 'val')

    d_train = STEDataset(audio_train_path, video_train_path,
                         io_config['csv_train_full'], clip_lenght,
                         video_train_transform, do_video_augment=True)
    d_val = STEDataset(audio_val_path, video_val_path,
                       io_config['csv_val_full'], clip_lenght,
                       video_val_transform, do_video_augment=False)

    dl_train = DataLoader(d_train, batch_size=opt_config['batch_size'],
                          shuffle=True, num_workers=opt_config['threads'],
                          pin_memory=True)
    dl_val = DataLoader(d_val, batch_size=opt_config['batch_size'],
                        shuffle=True, num_workers=opt_config['threads'],
                        pin_memory=True)

    model = optimize_ste_losses(backbone, dl_train, dl_val, device,
                                criterion, optimizer, scheduler,
                                num_epochs=opt_config['epochs'],
                                models_out=target_models, log=log)
