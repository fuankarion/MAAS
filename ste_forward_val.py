import os
import csv
import sys
import torch

from torchvision import transforms
from torch.utils.data import DataLoader

from core.datasets import STEDatasetForward
from core.util import configure_backbone_forward_phase, load_ava_val_video_set

import core.config as exp_conf
import core.custom_transforms as ct


if __name__ == '__main__':
    cuda_device_number = str(sys.argv[1])
    clip_lenght = int(sys.argv[2])
    image_size = (160, 160)

    model_weights = '/home/alcazajl/Models/ASC2/ste/ste_res18_clip11/55.pth'
    target_directory = '/home/alcazajl/Forwards/ICCV/ste11_val/'
    io_config = exp_conf.STE_inputs_forward
    opt_config = exp_conf.STE_forward_params

    # cuda config
    backbone = configure_backbone_forward_phase(opt_config['backbone'], model_weights, clip_lenght)
    has_cuda = torch.cuda.is_available()
    device = torch.device('cuda:'+cuda_device_number if has_cuda else 'cpu')
    backbone = backbone.to(device)

    video_val_transform = transforms.Compose([transforms.Resize(image_size), ct.video_val])
    video_val_path = os.path.join(io_config['video_dir'], 'val')
    audio_val_path = os.path.join(io_config['audio_dir'], 'val')

    video_set = load_ava_val_video_set()
    csv_path = io_config['csv_val_full']

    for video_key in video_set:
        target_file = os.path.join(target_directory, video_key+'.csv')
        if os.path.exists(target_file):
            print('skip', target_file)
            continue

        print('forward video ', video_key)
        with open(target_directory+video_key+'.csv', mode='w') as vf:
            vf_writer = csv.writer(vf, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

            d_val = STEDatasetForward(video_key, audio_val_path, video_val_path,
                                            csv_path, clip_lenght,
                                            image_size, video_val_transform,
                                            do_video_augment=False)

            dl_val = DataLoader(d_val, batch_size=opt_config['batch_size'],
                                shuffle=False, num_workers=opt_config['threads'])

            for idx, dl in enumerate(dl_val):
                if idx%10==0:
                    print(' \t Forward iter ', idx, '/', len(dl_val), end='\r')
                audio_data, video_data, video_id, ts, entity_id, gt = dl
                video_data = video_data.to(device)
                audio_data = audio_data.to(device)

                with torch.set_grad_enabled(False):
                    preds, _, _, feats_a, feats_v = backbone(audio_data, video_data)
                    feats_a = feats_a.cpu().numpy()[0]
                    feats_v = feats_v.cpu().numpy()[0]
                    vf_writer.writerow([video_id[0], ts[0], entity_id[0], float(gt[0]), float(preds[0][0]), float(preds[0][1]), list(feats_a), list(feats_v)])
