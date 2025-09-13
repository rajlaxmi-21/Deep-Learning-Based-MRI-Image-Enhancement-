import torch
import torch.nn.functional as F

def mixge_loss(pred, target, lambda_grad=0.1, eps=1e-6):
    
    mse = F.mse_loss(pred, target)

    gx = torch.tensor([[-1., -2., -1.],
                       [ 0.,  0.,  0.],
                       [ 1.,  2.,  1.]],
                       dtype=pred.dtype, device=pred.device).unsqueeze(0).unsqueeze(0)

    gy = torch.tensor([[-1.,  0.,  1.],
                       [-2.,  0.,  2.],
                       [-1.,  0.,  1.]],
                       dtype=pred.dtype, device=pred.device).unsqueeze(0).unsqueeze(0)

    pred_gx = F.conv2d(pred, gx, padding=1)
    pred_gy = F.conv2d(pred, gy, padding=1)
    targ_gx = F.conv2d(target, gx, padding=1)
    targ_gy = F.conv2d(target, gy, padding=1)

    pred_grad = torch.sqrt(pred_gx**2 + pred_gy**2 + eps)
    targ_grad = torch.sqrt(targ_gx**2 + targ_gy**2 + eps)

    mge = F.mse_loss(pred_grad, targ_grad)

    return mse + lambda_grad * mge

import torch
import torch.nn.functional as F

# Peak Signla to Noise Ratio

def psnr(pred, target, max_val=1.0):
    mse = F.mse_loss(pred, target, reduction='mean')
    if mse == 0:
        return torch.tensor(100.0)  
    return 20 * torch.log10(max_val / torch.sqrt(mse))

# Structural Similarity Index 

def ssim(pred, target, eps=1e-6):
    mu_x = pred.mean()
    mu_y = target.mean()
    sigma_x = pred.var(unbiased=False)
    sigma_y = target.var(unbiased=False)
    sigma_xy = ((pred - mu_x) * (target - mu_y)).mean()

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_val = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / \
               ((mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2) + eps)
    return ssim_val
