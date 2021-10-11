import sys
import torch

import core.config as exp_conf
import core.models_maas as maas

from torch.optim import lr_scheduler
from torch_geometric.data import DataLoader

from core.datasets import MAASDataset
from core.util import set_up_log_and_ws_out
from core.optimization import optimize_maas


if __name__ == '__main__':
    # experiment Reproducibility
    torch.manual_seed(11)
    torch.cuda.manual_seed(22)
    torch.backends.cudnn.deterministic = True

    time_length = int(sys.argv[1])
    stride = int(sys.argv[2])
    speakers = int(sys.argv[3])
    hidden_size = int(sys.argv[4])
    cuda_device_number = int(sys.argv[5])
    clusters = int(sys.argv[6])

    io_config = exp_conf.MAAS_inputs
    opt_config = exp_conf.MAAS_optimization_params

    # io config
    model_name = 'MAAS_'+str(time_length)+'len_'+str(stride)+'stride_'+str(speakers)+'speakers_'+str(hidden_size)+'filters_'+str(clusters)+'k'
    log, target_models = set_up_log_and_ws_out(io_config['models_out'],
                                               opt_config, model_name)

    # cuda config
    model = maas.MAAS(512, hidden_size, clusters)

    # GPU stuff
    has_cuda = torch.cuda.is_available()
    print('has_cuda', has_cuda)
    device = torch.device("cuda:"+str(cuda_device_number) if has_cuda else "cpu")
    model.to(device)

    criterion = opt_config['criterion']
    optimizer = opt_config['optimizer'](model.parameters(),
                                        lr=opt_config['learning_rate'])
    scheduler = lr_scheduler.StepLR(optimizer, step_size=opt_config['step_size'],
                                    gamma=opt_config['gamma'])

    dataset_train = MAASDataset(io_config['features_train_full'], time_length,
                                stride, noise=0.2, candidate_speakers=speakers,
                                limit_videos=None)
    dataset_val = MAASDataset(io_config['features_val_full'], time_length,
                              stride, candidate_speakers=speakers,
                              limit_videos=None)

    dl_train = DataLoader(dataset_train, batch_size=opt_config['batch_size'],
                          shuffle=True, num_workers=opt_config['threads'])
    dl_val = DataLoader(dataset_val, batch_size=opt_config['batch_size'],
                        shuffle=False, num_workers=opt_config['threads'])

    optimize_maas(model, (1, speakers, time_length), dl_train, dl_val, device,
                             criterion, optimizer, scheduler,
                             num_epochs=opt_config['epochs'],
                             models_out=target_models, log=log)
