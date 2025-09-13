import torch
import torch.nn.functional as F
import torch.optim as optim


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


device = torch.device("cuda")
model = UnetSrplus(in_ch=1, out_ch=1, depth=5, base_channels=32).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for lq, hq in train_loader:
        lq, hq = lq.to(device), hq.to(device)

        pred = model(lq)
        loss = mixge_loss(pred, hq, lambda_grad=0.1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss+= loss.item() * lq.size(0)

    train_loss /= len(train_loader.dataset)
    #validation

    model.eval()
    val_loss, val_psnr, val_ssim = 0.0, 0.0, 0.0
    with torch.no_grad():
        for lq, hq, in val_loader:
            lq, hq = lq.to(device), hq.to(device)
            pred = model(lq)

            loss = mixge_loss(pred, hq, lambda_grad=0.1)
            val_loss+= loss.item() * lq.size(0)
            val_psnr += psnr(pred, hq).item() * lq.size(0)
            val_ssim += ssim(pred, hq).item() * lq.size(0)

    val_loss /= len(val_loader.dataset)
    val_psnr /= len(val_loader.dataset)
    val_ssim /= len(val_loader.dataset)

    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"Train Loss: {train_loss:.6f} "
          f"Val Loss: {val_loss:.6f} "
          f"Val PSNR: {val_psnr:.2f} dB "
          f"Val SSIM: {val_ssim:.4f}")
    
    if (epoch+1) % 5 == 0:
        torch.save(model.state_dict(), f"unetsrplus_epoch{epoch+1}.pth")