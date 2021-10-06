import torch
from sklearn.metrics import average_precision_score

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
