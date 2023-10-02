import torch.backends.cudnn as cudnn
from dataset import *
from torch.utils.data.dataloader import DataLoader
from FSRCNN import *
from LapSRN import *
from SRResNet import *
from VDSR import *
from FENet import *
from RFDN import *
from ESRT import *
from CDAN import *
import torch
from torch import nn
from torchvision.utils import save_image
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim


if __name__ == '__main__':

    model = 'CDAN'
    weights = rf'params/CDAN_x4.pth'
    test_data_path = r'data/StructSeg2019.npy'
    net_scale = 4
    save_path = 'pic'
    batch_size = 1

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    test_dataloader = DataLoader(Test_dataset(test_data_path, scale=net_scale),
                                 batch_size=batch_size, shuffle=False)

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

    print('Current model: %s' % model)

    net = net.to(device)
    net.load_state_dict(torch.load(weights, map_location=device))

    criterion = nn.MSELoss().to(device)

    net.eval()

    i = 0
    all_psnr = 0.0
    all_ssim = 0.0
    n_iter_test = len(test_dataloader)

    for data in test_dataloader:
        inputs, labels = data

        inputs = inputs.to(torch.float32)
        labels = labels.to(torch.float32)
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            output_test = net(inputs)
            output_test = output_test.clamp(0.0, 1.0)
            loss = criterion(output_test, labels)

            output_test_np = output_test.cpu().numpy()
            output_test_np = np.squeeze(output_test_np, axis=(0, 1))

            label_np = labels.cpu().numpy()
            label_np = np.squeeze(label_np, axis=(0, 1))

            psnr = compare_psnr(output_test_np, label_np)
            ssim = compare_ssim(output_test_np, label_np)

            print("PSNR:%f, SSIM:%f" % (psnr, ssim))

            all_psnr = psnr + all_psnr
            all_ssim = ssim + all_ssim

        if i % 20 == 0:

            out_img = output_test[0]
            la_img = labels[0]
            img = torch.stack([la_img, out_img], dim=0)

            save_image(img, f'{save_path}/{i}.png')

        i = i + 1

    print('*' * 30)
    print('Number of pictures:')
    print(i)
    print("total_PSNR:%f, total_SSIM:%f" % (all_psnr / i, all_ssim / i))
