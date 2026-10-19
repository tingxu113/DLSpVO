import numpy as np
import pypher.pypher
import cv2
import torch
import scipy.io as sio
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as sk_cpt_ssim
import torch.utils.data as data


def para_setting(kernel_type, sf, sz, delta):
    if kernel_type == 'uniform_blur':
        psf = np.ones([sf,sf]) / (sf * sf)
    elif kernel_type == 'gaussian_blur':
        BW = cv2.getGaussianKernel(sf, delta)
        psf = np.multiply(BW, BW.T)
    BW1 = pypher.pypher.psf2otf(BW, [192, 1])
    fft_B = pypher.pypher.psf2otf(psf, sz)
    return fft_B, BW1

def H_z(z, factor, fft_B ):
    f = torch.fft.fft2(z, dim=(-2, -1))
    f = torch.stack((f.real, f.imag), -1)
    if len(z.shape) == 3:
        ch, h, w = z.shape
        fft_B = fft_B.unsqueeze(0).repeat(ch, 1, 1, 1)
        M = torch.cat(((f[:, :, :, 0]*fft_B[:, :, :, 0]-f[:, :, :, 1]*fft_B[:, :, :, 1]).unsqueeze(3), (
                f[:, :, :, 0]*fft_B[:, :, :, 1]+f[:, :, :, 1]*fft_B[:, :, :, 0]).unsqueeze(3)), 3)
        Hz = torch.fft.ifft2(torch.complex(M[..., 0], M[..., 1]), dim=(-2, -1))
        x = Hz[:, int(factor//2)::factor, int(factor//2)::factor].real
    elif len(z.shape) == 4:
        bs, ch, h, w = z.shape
        fft_B = fft_B.unsqueeze(0).unsqueeze(0).repeat(bs, ch, 1, 1, 1)
        M = torch.cat(((f[:, :, :, :, 0]*fft_B[:, :, :, :, 0]-f[:, :, :, :, 1]*fft_B[:, :, :, :, 1]).unsqueeze(4), (
                f[:, :, :, :, 0]*fft_B[:, :, :, :, 1]+f[:, :, :, :, 1]*fft_B[:, :, :, :, 0]).unsqueeze(4)), 4)
        Hz = torch.fft.ifft2(torch.complex(M[..., 0], M[..., 1]), dim=(-2, -1))
        x = Hz[:, :, int(factor//2)::factor, int(factor//2)::factor].real
    return x


def softT(x, tau):
    return torch.sign(x) * torch.maximum(torch.abs(x) - tau, torch.tensor(0.0))

def hyperConvert2D(Image3D):
    if Image3D.ndim == 2:
        numBands = 1
        h, w = Image3D.shape
    else:
        h, w, numBands = Image3D.shape
    Image2D = np.reshape(Image3D, (w * h, numBands), order='F').T
    return Image2D


class HSI_MSI_Data(data.Dataset):
    def __init__(self,train_hrhs_all,train_lrhs_all):
        self.train_hrhs_all = train_hrhs_all
        self.train_lrhs_all  = train_lrhs_all
    def __getitem__(self, index):
        train_hrhs = self.train_hrhs_all[index, :, :, :]
        train_lrhs = self.train_lrhs_all[index, :, :, :]
        return train_hrhs, train_lrhs

    def __len__(self):
        return self.train_hrhs_all.shape[0]

def psnr(img, imclean, data_range):
    img_cpu = img.astype(np.float32)
    imgclean = imclean.astype(np.float32)
    psnr = 0
    for i in range(img_cpu.shape[0]):
        psnr += compare_psnr(np.clip(imgclean[i, :, :],0,1), np.clip(img_cpu[i, :, :],0,1), data_range=data_range)
    return psnr / img_cpu.shape[0]

def compute_mse(img1, img2):
    w,h,c = img1.shape
    img1 = np.reshape(img1, (w*h,c))
    img2 = np.reshape(img2, (w*h,c))
    mse_sum  = (img1  - img2 )**2
    mse_loss = mse_sum.mean(0)
    mse_loss = mse_loss
    mse = mse_sum.mean()
    if mse < 1.0e-10:
        return 100
    return mse_loss

def compute_ssim(im1,im2):
    w,h,c = im1.shape
    im1 = np.reshape(im1,(w,h,c))
    im2 = np.reshape(im2,(w,h,c))
    n = im1.shape[2]
    ms_ssim = 0.0
    for i in range(n):
        single_ssim = sk_cpt_ssim(im1[:,:,i], im2[:,:,i],data_range=1.0)
        ms_ssim += single_ssim
    return ms_ssim/n

def compute_sam(im1, im2):
    w,h,c = im1.shape
    im1 = np.reshape(im1,(w*h,c))
    im2 = np.reshape(im2,(w*h,c))
    mole = np.sum(np.multiply(im1, im2), axis=1)
    im1_norm = np.sqrt(np.sum(np.square(im1), axis=1))
    im2_norm = np.sqrt(np.sum(np.square(im2), axis=1))
    deno = np.multiply(im1_norm, im2_norm)
    sam = np.rad2deg(np.arccos(((mole+10e-8)/(deno+10e-8)).clip(-1,1)))
    return np.mean(sam)

def compute_ergas(mse, out, r=8):
    w,h,c = out.shape
    out = np.reshape(out,(w*h,c))
    out_mean = np.mean(out, axis=0)
    mse = np.reshape(mse, (c, 1))
    out_mean = np.reshape(out_mean, (c, 1))
    ergas = 100/r*np.sqrt(np.mean(mse/out_mean**2))
    return ergas