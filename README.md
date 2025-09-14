# Deep-Learning-Based-MRI-Image-Enhancement

MRI Image Enhancement using Deep Learning

This repository contains implementations of vanilla and modified U-Net architectures for enhancing low-field (0.4T) knee MRI images to approximate high-field (3T) image quality. The goal is to improve image clarity and reduce noise for potential clinical evaluation.

Features

Vanilla U-Net for MRI image enhancement.

Modified U-Net with improved performance over the vanilla model.

Quantitative evaluation using PSNR and SSIM metrics.

Comparison with existing super-resolution and fat-suppression MRI methods (Inaoka et al., 2024).

| Model          | PSNR (dB) | SSIM          |
| -------------- | --------- | -----------   |
| Vanilla U-Net  | 32–33     | 0.9102–0.9699 |
| Modified U-Net | 33–33.6   | 0.9788–0.9799 |


Tools & Libraries

Python 3.x

PyTorch

h5py

scikit-image

NumPy
