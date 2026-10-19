import numpy as np
import sys

import torch
sys.path.append('modules')
import scipy.io as sio
from torch.optim import Adam
import matlab.engine
import time
import cv2
import torch.utils.data as data
from skimage.measure import block_reduce
import utils
from unet import UNet



def setup_seed(seed=0):
    import torch
    import os
    import numpy as np
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main():
    setup_seed(seed=0)

    lambdai = 1e-5
    miu = 1e-5
    n = 20

    start_time = time.time()

    training_size = 32
    stride = 1
    downsample_factor = 4
    BATCH_SIZE = 64
    EPOCH1 = 2000
    EPOCH2 = 5000
    Iters = 50

    Z = sio.loadmat('data/flower.mat')['S'].astype(np.float32).transpose([2, 0, 1])
    Z = block_reduce(Z, block_size=(1, 2, 2), func=np.mean)
    Z_tensor = torch.FloatTensor(Z).cuda()

    def create_F():
        F = np.array([
            [2,  1,  1,  1,  1,  1,  0,  0,  30,  0,  0,  0,  0,  0,  0,  0,  2,  6, 11, 17, 21, 22, 21, 20, 20, 19, 19, 18, 18, 17, 17],
            [1,  1,  1,  1,  1,  1,  2,  4,  6,  8, 11, 16, 19, 21, 20, 18, 16, 14, 11,  7,  5,  3,  2,  2,  1,  1,  2,  2,  2,  2,  2],
            [7, 10, 15, 19, 25, 29, 30, 29, 27, 22, 16,  9,  2,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1]
        ], dtype=np.float32)
        for band in range(3):
            row_sum = np.sum(F[band, :])
            F[band, :] = F[band, :] / row_sum
        return F
    R_numpy = create_F()
    R = torch.FloatTensor(R_numpy)
    R = R.cuda()

    Y = torch.tensordot(R, Z_tensor, dims=([1], [0])).cuda()
    Y_tensor = Y.unsqueeze(0).cuda()

    fft_B, bw = utils.para_setting('gaussian_blur', 8, [Z.shape[1], Z.shape[2]], 3)
    fft_B = torch.cat(
        (torch.Tensor(np.real(fft_B)).unsqueeze(2), torch.Tensor(np.imag(fft_B)).unsqueeze(2)), 2).cuda()
    X = utils.H_z(Z_tensor, downsample_factor, fft_B).cuda()
    HSI0 = X.clone().detach().cpu().numpy()
    X_matrix = utils.hyperConvert2D(X.cpu().numpy().transpose([1, 2, 0]))

    eng = matlab.engine.start_matlab()
    eng.cd(r'.', nargout=0)
    X_matlab = matlab.double(X_matrix.tolist())
    par = eng.struct()
    par['K'] = n
    E_matlab = eng.Nonnegative_DL(X_matlab, par)
    E = np.array(E_matlab, dtype=np.float32)
    eng.quit()
    E_tensor = torch.from_numpy(E).cuda()


    augument = [0]
    HSI_aug = []
    HSI_aug.append(HSI0)

    train_lrhs = []
    train_hrhs = []

    for j in augument:
        HSI = cv2.flip(HSI0, j)
        HSI_aug.append(HSI)

    Gnet = UNet(num_input_channels=Y.shape[0], num_output_channels=n).cuda()
    Opt = Adam(list(Gnet.parameters()), lr=1e-3)

    print(f"mu: {miu} lambda: {lambdai}  r: {n}")

    for j in range(len(HSI_aug)):
        HSI = HSI_aug[j]
        MSI_LR = np.tensordot(R_numpy, HSI, axes=([1], [0]))
        for j in range(0, HSI.shape[1] - training_size + 1, stride):
            for k in range(0, HSI.shape[2] - training_size + 1, stride):
                temp_lrhs = MSI_LR[:, j:j + training_size, k:k + training_size]
                temp_hrhs = HSI[:, j:j + training_size, k:k + training_size]

                train_lrhs.append(temp_lrhs)
                train_hrhs.append(temp_hrhs)

    train_lrhs = torch.Tensor(train_lrhs)
    train_hrhs = torch.Tensor(train_hrhs)

    train_data = utils.HSI_MSI_Data(train_hrhs, train_lrhs)
    train_loader = data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

    net_ini_time = time.time()
    for i in range(EPOCH1):
        for step1, (a1, a2) in enumerate(train_loader):

            f_theta_pre = Gnet(a2.cuda())

            E_f_theta_pre = torch.tensordot(f_theta_pre, E_tensor, dims=([1], [1]))
            E_f_theta_pre = torch.Tensor.permute(E_f_theta_pre, (0, 3, 1, 2))

            loss_pre = torch.sum((a1.cuda() - E_f_theta_pre) ** 2)

            Opt.zero_grad()
            loss_pre.backward()
            Opt.step()

        if (i + 1) % 1000 == 0:
            print(f"  epoch: {i + 1:d}   loss: {loss_pre.item():.4e}")

    net_ini_time = time.time() - net_ini_time
    print(f"  Initialization Time: {net_ini_time:.3f}")
    print('\n')

    torch.save({'model': Gnet.state_dict(), 'opt': Opt.state_dict()},
               "data/Gpre.pth")

    cks = torch.load("data/Gpre.pth")
    Gnet.load_state_dict(cks['model'])

    import gc
    del train_lrhs
    del train_hrhs
    del train_data
    del train_loader
    gc.collect()

    del Opt
    Opt = Adam(list(Gnet.parameters()), lr=1e-4)

    G = torch.zeros((n, Y.shape[1], Y.shape[2]), dtype=torch.float32).cuda()
    eta = torch.zeros((n, Y.shape[1], Y.shape[2]), dtype=torch.float32).cuda()

    f_theta_copy = None
    E_f_theta_copy = None


    for Iter in range(Iters):
        if (Iter + 1) == 10:
            EPOCH2 = 4000
        elif (Iter + 1) == 20:
            EPOCH2 = 3000
        elif (Iter + 1) == 30:
            EPOCH2 = 2000
        elif (Iter + 1) == 40:
            EPOCH2 = 1000

        net_time = time.time()
        for i in range(EPOCH2):
            f_theta = Gnet(Y_tensor)

            E_f_theta = torch.tensordot(f_theta, E_tensor, dims=([1], [1]))
            E_f_theta = torch.Tensor.permute(E_f_theta, (0, 3, 1, 2))

            if i == EPOCH2 - 1:
                f_theta_copy = f_theta.clone().detach().squeeze(0)
                E_f_theta_copy = E_f_theta.clone().detach().squeeze(0)

            X_hat = utils.H_z(E_f_theta.squeeze(0), downsample_factor, fft_B)
            Y_hat = torch.tensordot(R, E_f_theta.squeeze(0), dims=([1], [0]))

            loss = (torch.sum((X - X_hat) ** 2) + torch.sum((Y - Y_hat) ** 2) +
                        miu * torch.sum((G - f_theta.squeeze(0) + eta) ** 2))

            if (i + 1) % 1000 == 0:
                print(f"  epoch: {i + 1:d}   loss: {loss.item():.4e}")

            Opt.zero_grad()
            loss.backward()
            Opt.step()

        net_time = time.time() - net_time
        print(f"   Time: {net_time:.3f}")

        G = utils.softT(f_theta_copy - eta, lambdai / (2 * miu))

        eta = eta + G - f_theta_copy

        Z_hat = E_f_theta_copy.cpu().numpy()

        psnr = utils.psnr(Z_hat, Z, data_range=1.0)
        mse = utils.compute_mse(Z_hat, Z)
        ssim = utils.compute_ssim(Z_hat, Z)
        sam = utils.compute_sam(Z_hat, Z)
        ergas = utils.compute_ergas(mse, Z, r=downsample_factor)

        if (Iter + 1) % 1 == 0:
            print(f"      Iter: {Iter + 1:d}  PSNR: {psnr:.4f}  SSIM: {ssim: .4f}  SAM: {sam: .4f}  ERGAS: {ergas: .4f}")


    our_time = time.time() - start_time
    print(f"   All Time: {our_time:.3f}")
    print('\n')


if __name__ == "__main__":
    main()
