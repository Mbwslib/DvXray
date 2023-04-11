import torch
from model_ResNet import *

from dataset import data_loader, DvX_dataset_collate
from loss_func import BCELoss

from torch.utils.data import DataLoader
from utils import adjust_learning_rate, clip_gradient, confidence_weighted_view_fusion
from get_ap import AveragePrecisionMeter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Training parameters
start_epoch = 0
epochs = 30
input_shape = [224, 224]
learning_rate = 0.01
batch_size = 32
grad_clip = 5.
print_freq = 100
train_annotation_path = 'DvXray_train.txt'

checkpoint = None


def save_checkpoint(epoch, model, optimizer):

    state = {'epoch': epoch,
             'model': model,
             'optimizer': optimizer,
             }

    filename = 'ep%03d_ResNet_checkpoint.pth.tar' % (epoch + 1)
    torch.save(state, './checkpoint/' + filename)

def main():

    '''
    Training and Validation
    '''

    global checkpoint, start_epoch

    if checkpoint is None:

        model = AHCR(num_classes=15)

        optimizer = torch.optim.SGD(params=filter(lambda p: p.requires_grad, model.parameters()),
                                     lr=learning_rate)

    else:

        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']

    model = model.to(device)

    criterion = BCELoss().to(device)

    with open(train_annotation_path) as f:
        train_lines = f.readlines()

    train_loader = DataLoader(data_loader(train_lines, input_shape), batch_size=batch_size, shuffle=True,
                               drop_last=True, collate_fn=DvX_dataset_collate)

    for epoch in range(start_epoch, epochs):

        if epoch is not 0 and epoch % 10 == 0:
            adjust_learning_rate(optimizer, 0.1)

        train(train_loader, model, criterion, optimizer, epoch)

        save_checkpoint(epoch, model, optimizer)


def train(train_loader, model, criterion, optimizer, epoch):

    model.train()

    Loss = []

    ap_meter = AveragePrecisionMeter()
    ap_meter.reset()

    for i, (img_ols, img_sds, gt_s) in enumerate(train_loader):

        img_ols = img_ols.to(device)
        img_sds = img_sds.to(device)
        gt_s = gt_s.to(device)

        ol_output, sd_output = model(img_ols, img_sds)

        loss = criterion(ol_output, sd_output, gt_s)

        optimizer.zero_grad()
        loss.backward()

        Loss.append(loss.item())

        if grad_clip is not None:
            clip_gradient(optimizer, grad_clip)

        optimizer.step()

        if i is not 0 and i % print_freq == 0:
            print('Epoch: [{}]/[{}/{}]\t'
                  'Loss: {:.3f}'.format((epoch + 1), i, len(train_loader),
                                        sum(Loss)/len(Loss),))

        prediction = confidence_weighted_view_fusion(torch.sigmoid(ol_output), torch.sigmoid(sd_output))

        ap_meter.add(prediction.data, gt_s)

    each_ap = ap_meter.value()
    map = 100 * each_ap.mean()

    print(each_ap)
    print('mAP: {:.3f}'.format(map))


if __name__ == '__main__':
    main()