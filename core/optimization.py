import os
import torch

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

##################
###### STE #######
##################

def optimize_ste_losses(model, dataloader_train, data_loader_val, device,
                       criterion, optimizer, scheduler, num_epochs,
                       models_out=None, log=None):

    for epoch in range(num_epochs):
        print()
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        outs_train = _train_model_ste_losses(model, dataloader_train, optimizer,
                                             criterion, device)
        outs_val = _test_model_ste_losses(model, data_loader_val, criterion,
                                          device)
        scheduler.step()

        train_loss, train_loss_a, train_loss_v, train_auc, train_ap = outs_train
        val_loss, val_loss_a, val_loss_v, val_auc, val_ap = outs_val

        if models_out is not None:
            model_target = os.path.join(models_out, str(epoch+1)+'.pth')
            print('save model to ', model_target)
            torch.save(model.state_dict(), model_target)

        if log is not None:
            log.writeDataLog([epoch+1, train_loss, train_loss_a, train_loss_v, train_ap, val_loss, val_loss_a, val_loss_v, val_ap])

    return model


def _train_model_ste_losses(model, dataloader, optimizer, criterion, device):
    softmax_layer = torch.nn.Softmax(dim=1)

    model.train()
    pred_lst = []
    label_lst = []

    running_loss_av = 0.0
    running_loss_a = 0.0
    running_loss_v = 0.0

    # Iterate over data
    for idx, dl in enumerate(dataloader):
        print('\t Train iter ', idx, '/', len(dataloader), end='\r')
        (audio_data, video_data), av_label, audio_label = dl

        video_data = video_data.to(device, non_blocking=True)
        audio_data = audio_data.to(device, non_blocking=True)
        av_label = av_label.to(device, non_blocking=True)
        audio_label = audio_label.to(device, non_blocking=True)

        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            av_out, a_out, v_out, _, _ = model(audio_data, video_data)
            _, preds = torch.max(av_out, 1)

            loss_av = criterion(av_out, av_label)
            loss_a = criterion(a_out, audio_label)
            loss_v = criterion(v_out, av_label)
            loss = loss_av + loss_a + loss_v

            loss.backward()
            optimizer.step()

        with torch.set_grad_enabled(False):
            label_lst.extend(av_label.cpu().numpy().tolist())
            pred_lst.extend(softmax_layer(av_out).cpu().numpy()[:, 1].tolist())

        # statistics
        running_loss_av += loss_av.item()
        running_loss_a += loss_a.item()
        running_loss_v += loss_v.item()

    epoch_loss_av = running_loss_av / len(dataloader)
    epoch_loss_a = running_loss_a / len(dataloader)
    epoch_loss_v = running_loss_v / len(dataloader)

    epoch_auc = roc_auc_score(label_lst, pred_lst)
    epoch_ap = average_precision_score(label_lst, pred_lst)
    print('AV Loss: {:.4f} A Loss: {:.4f}  V Loss: {:.4f}  AUC: {:.4f} AP: {:.4f}'.format(
          epoch_loss_av, epoch_loss_a, epoch_loss_v, epoch_auc, epoch_ap))
    return epoch_loss_av, epoch_loss_a, epoch_loss_v, epoch_auc, epoch_ap


def _test_model_ste_losses(model, dataloader, criterion, device):
    softmax_layer = torch.nn.Softmax(dim=1)

    model.eval()   # Set model to evaluate mode
    pred_lst = []
    label_lst = []

    running_loss_av = 0.0
    running_loss_a = 0.0
    running_loss_v = 0.0

    # Iterate over data.
    for idx, dl in enumerate(dataloader):
        print('\t Val iter ', idx, '/', len(dataloader), end='\r')
        (audio_data, video_data), av_label, audio_label = dl

        video_data = video_data.to(device)
        audio_data = audio_data.to(device)
        av_label = av_label.to(device)
        audio_label = audio_label.to(device)

        # forward
        with torch.set_grad_enabled(False):
            av_out, a_out, v_out, _, _ = model(audio_data, video_data)
            _, preds = torch.max(av_out, 1)
            loss_av = criterion(av_out, av_label)
            loss_a = criterion(a_out, audio_label)
            loss_v = criterion(v_out, av_label)

            label_lst.extend(av_label.cpu().numpy().tolist())
            pred_lst.extend(softmax_layer(av_out).cpu().numpy()[:, 1].tolist())

        # statistics
        running_loss_av += loss_av.item()
        running_loss_a += loss_a.item()
        running_loss_v += loss_v.item()

    epoch_loss_av = running_loss_av / len(dataloader)
    epoch_loss_a = running_loss_a / len(dataloader)
    epoch_loss_v = running_loss_v / len(dataloader)

    epoch_auc = roc_auc_score(label_lst, pred_lst)
    epoch_ap = average_precision_score(label_lst, pred_lst)
    print('AV Loss: {:.4f}  A Loss: {:.4f}  V Loss: {:.4f} auROC: {:.4f} AP: {:.4f}'.format(
          epoch_loss_av, epoch_loss_a, epoch_loss_v, epoch_auc, epoch_ap))

    return epoch_loss_av, epoch_loss_a, epoch_loss_v, epoch_auc, epoch_ap



##################
###### MAAS ######
##################

def optimize_maas(model, space_conf, dataloader_train,
                  data_loader_val, device, criterion, optimizer,
                  scheduler, num_epochs, models_out=None, log=None):

    for epoch in range(num_epochs):
        print()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        loss, ap = _train_maas(model, dataloader_train, optimizer, criterion, device)
        val_loss, val_ap, val_ap_same, val_ap_mid = _test_maas(model, space_conf, data_loader_val, criterion, device)
        scheduler.step()

        if models_out is not None:
            model_target = os.path.join(models_out, str(epoch+1)+'.pth')
            print('save model to ', model_target)
            torch.save(model.state_dict(), model_target)

        if log is not None:
            log.writeDataLog([epoch+1, loss, ap, ' ', val_loss, val_ap, val_ap_same, val_ap_mid])

    return model


def _train_maas(model, dataloader, optimizer, criterion, device):
    model.train()

    # Stats vars
    softmax_layer = torch.nn.Softmax(dim=-1)
    running_loss = 0.0
    pred_lst = []
    label_lst = []

    # Iterate over data
    for idx, dl in enumerate(dataloader):
        print('\t Train iter {:d}/{:d} {:.4f}'.format(idx, len(dataloader), running_loss/(idx+1)) , end='\r')

        graph_data = dl
        graph_data = graph_data.to(device)
        targets = graph_data.y

        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            outputs = model(graph_data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        with torch.set_grad_enabled(False):
            label_lst.extend(targets.cpu().numpy().tolist())
            pred_lst.extend(softmax_layer(outputs).cpu().numpy()[:, 1].tolist())

        # statistics
        running_loss += loss.item()
        if idx == len(dataloader)-2:
            break

    epoch_loss = running_loss / len(dataloader)
    epoch_ap = average_precision_score(label_lst, pred_lst)
    print('Train Loss: {:.4f} mAP: {:.4f}'.format(epoch_loss, epoch_ap))

    return epoch_loss, epoch_ap


def _test_maas(model, space_conf, dataloader, criterion, device):
    model.eval()  # Set model to evaluate mode

    offset, speakers, time_l = space_conf
    mid = offset + (speakers+1)*int(time_l/2)

    # Stats vars
    softmax_layer = torch.nn.Softmax(dim=-1)
    running_loss = 0.0
    pred_lst = []
    label_lst = []

    pred_lst_same = []
    label_lst_same = []

    pred_lst_mid = []
    label_lst_mid = []

    # Iterate over data.
    for idx, dl in enumerate(dataloader):
        print('\t Val iter {:d}/{:d}'.format(idx, len(dataloader)) , end='\r')
        graph_data = dl
        graph_data = graph_data.to(device)
        targets = torch.flatten(graph_data.y)

        # forward
        with torch.set_grad_enabled(False):
            outputs = model(graph_data)
            loss = criterion(outputs, targets)

            label_lst.extend(targets.cpu().numpy().tolist())
            pred_lst.extend(softmax_layer(outputs).cpu().numpy()[:, 1].tolist())

            label_lst_same.extend(targets[offset::speakers+1].cpu().numpy().tolist())
            pred_lst_same.extend(softmax_layer(outputs)[offset::speakers+1].cpu().numpy()[:, 1].tolist())

            label_lst_mid.extend(targets[mid::(speakers+1)*time_l].cpu().numpy().tolist())
            pred_lst_mid.extend(softmax_layer(outputs)[mid::(speakers+1)*time_l].cpu().numpy()[:, 1].tolist())

        # statistics
        running_loss += loss.item()
        if idx == len(dataloader)-2:
            break

    epoch_loss = running_loss / len(dataloader)
    epoch_ap = average_precision_score(label_lst, pred_lst)
    epoch_ap_same = average_precision_score(label_lst_same, pred_lst_same)
    epoch_ap_mid = average_precision_score(label_lst_mid, pred_lst_mid)
    print('Val Loss: {:.4f} mAP: {:.4f} same_mAP{:.4f} mid_mAP{:.4f} '.format(epoch_loss, epoch_ap, epoch_ap_same, epoch_ap_mid))

    return epoch_loss, epoch_ap, epoch_ap_same, epoch_ap_mid
