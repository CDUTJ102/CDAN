import torch.backends.cudnn as cudnn
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
import os
from dataset import *
from math import log10
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from CDAN import *
from FSRCNN import *
from LapSRN import *
from SRResNet import *
from VDSR import *
from FENet import *
from RFDN import *
from ESRT import *


if __name__ == '__main__':

    train_path = r'/'
    val_path = r'/'
    batch_size = 30
    crop_size_train = 128
    crop_size_val = 128
    net_scale = 4
    model = 'CDAN'
    checkpoint = rf'trained_model/{model}_x{net_scale}.pth'
    start_epoch = 1
    epochs = 300
    workers = 32
    ngpu = 1
    cudnn.benchmark = True
    lr = 1e-3
    lr_gamma = 0.95
    lr_step = 5
    log_dir = rf'log/{model}_x{net_scale}'

    if os.path.isdir(log_dir):
        pass
    else:
        os.mkdir(log_dir)

    train_dataloader = DataLoader(Train_dataset(train_path, crop_size_train, net_scale), batch_size=batch_size,
                                  num_workers=workers, shuffle=True, drop_last=True)

    val_dataloader = DataLoader(Train_dataset(val_path, crop_size_val, net_scale), batch_size=1, num_workers=workers)

    pre_psnr = 0

    device = torch.device("cuda:0")
    print(device)

    writer = SummaryWriter(log_dir)

    if model == 'CDAN':
        net = CDAN(scale=net_scale)

    elif model == 'FSRCNN':
        net = FSRCNN(scale=net_scale)

    elif model == 'LAPSRN':
        net = LAPSRN(scale=net_scale)

    elif model == 'VDSR':
        net = VDSR()

    elif model == 'SRResNet':
        net = SRResNet(scale=net_scale)

    elif model == 'VDSR':
        net = VDSR()

    elif model == 'ESRT':
        net = ESRT(scale=net_scale)

    elif model == 'FENet':
        net = FENet(scale=net_scale)

    elif model == 'RFDN':
        net = RFDN(scale=net_scale)

    optimizer = torch.optim.Adam(params=net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=lr_gamma)

    print('Current model: %s' % model)
    net = net.to(device)

    criterion = nn.MSELoss().to(device)

    for epoch in range(start_epoch, epochs + 1):

        net.train()
        train_loss = 0
        n_iter_train = len(train_dataloader)
        train_psnr = 0

        with tqdm(total=(len(Train_dataset(train_path, 256, 4)) -
                         len(Train_dataset(train_path, 256, 4)) % 5)) as t:

            t.set_description('epoch: {}/{}'.format(epoch, epochs))

            for lr_imgs, hr_imgs in train_dataloader:

                lr_imgs = lr_imgs.to(torch.float32)
                hr_imgs = hr_imgs.to(torch.float32)

                lr_imgs = lr_imgs.to(device)
                hr_imgs = hr_imgs.to(device)

                sr_imgs = net(lr_imgs)
                loss = criterion(sr_imgs, hr_imgs)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                psnr = 10 * log10(1 / loss.item())
                train_psnr += psnr
                t.set_postfix(loss='{:.6f}'.format(loss))
                t.update(len(lr_imgs))

            epoch_loss_train = train_loss / n_iter_train
            train_psnr = train_psnr / n_iter_train

        print(f"Epoch {epoch}. Training loss: {epoch_loss_train} Train psnr {train_psnr}DB")

        net.eval()
        test_loss = 0
        all_psnr = 0
        n_iter_test = len(val_dataloader)

        with torch.no_grad():
            for i, (lr_imgs, hr_imgs) in enumerate(val_dataloader):

                lr_imgs = lr_imgs.to(torch.float32)
                hr_imgs = hr_imgs.to(torch.float32)

                lr_imgs = lr_imgs.to(device)
                hr_imgs = hr_imgs.to(device)

                sr_imgs = net(lr_imgs)
                loss = criterion(sr_imgs, hr_imgs)

                if loss.item() != 0:
                    psnr = 10 * log10(1 / loss.item())

                all_psnr += psnr
                test_loss += loss.item()

                if i == n_iter_test - 30:
                    writer.add_image('CDAN/epoch_' + str(epoch) + '_lr',
                                     make_grid(lr_imgs[:4, :3, :, :].cpu(), nrow=4, normalize=True), epoch)
                    writer.add_image('CDAN/epoch_' + str(epoch) + '_sr',
                                     make_grid(sr_imgs[:4, :3, :, :].cpu(), nrow=4, normalize=True), epoch)
                    writer.add_image('CDAN/epoch_' + str(epoch) + '_hr',
                                     make_grid(hr_imgs[:4, :3, :, :].cpu(), nrow=4, normalize=True), epoch)

        epoch_loss_test = test_loss / n_iter_test
        epoch_psnr = all_psnr / n_iter_test

        # tensorboard
        writer.add_scalars('CDAN/Loss', {
            'train_loss': epoch_loss_train,
            'test_loss': epoch_loss_test,
        }, epoch)

        writer.add_scalars('CDAN/PSNR', {
            'train_psnr': train_psnr,
            'test_psnr': epoch_psnr,
        }, epoch)

        print(f"Epoch {epoch}. Testing loss: {epoch_loss_test} Test psnr: {epoch_psnr} dB")

        if epoch_psnr > pre_psnr:
            torch.save(net.state_dict(), checkpoint)
            pre_psnr = epoch_psnr
            print('save done')

        print('Current learning rate:', end=' ')
        print(optimizer.state_dict()['param_groups'][0]['lr'])
        print('*' * 50)

        scheduler.step()

    writer.close()
