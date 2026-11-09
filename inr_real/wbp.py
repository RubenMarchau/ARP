"""Real-space weighted back-projection of (even + odd) / 2 on the same grid as the INR volumes.

Usage: python inr_real/wbp.py even.mrc odd.mrc tilts.tlt thickness_nm out.mrc
Ramp filter along x (perpendicular to the tilt axis y), no CTF correction. The sign is flipped so that density is
bright, as in the INR (the raw images have protein dark). Detector x of a voxel is x cos - z sin, the INR convention.
"""
import sys

import mrcfile
import numpy as np
import torch
import torch.nn.functional as F

even_path, odd_path, tlt_path, thickness_nm, out_path = sys.argv[1:6]
with mrcfile.open(even_path) as mrc:
    pixel = float(mrc.voxel_size.x)
    even = torch.tensor(mrc.data).float()
odd = torch.tensor(mrcfile.read(odd_path)).float()
tilts = torch.tensor(np.loadtxt(tlt_path), dtype=torch.float32)
images = (even + odd) / 2
images = images - images.mean(dim=(-2, -1), keepdim=True)
n_tilts, ny, nx = images.shape

pad = 2 * nx
filtered = torch.fft.irfft(torch.fft.rfft(images, n=pad, dim=-1) * torch.fft.rfftfreq(pad), n=pad, dim=-1)[..., :nx]

nz = round(float(thickness_nm) * 10 / pixel)
z = torch.arange(nz) - nz / 2 + 0.5
x = (torch.arange(nx) - nx // 2).float()
y = (torch.arange(ny) - ny // 2).float()
gy = (y + ny // 2) / (ny - 1) * 2 - 1
volume = torch.zeros(nz, ny, nx)
for t in range(n_tilts):
    angle = torch.deg2rad(tilts[t])
    detector_x = x[None, :] * torch.cos(angle) - z[:, None] * torch.sin(angle)  # (z, x)
    gx = (detector_x + nx // 2) / (nx - 1) * 2 - 1
    grid = torch.stack([gx[:, None, :].expand(-1, ny, -1), gy[None, :, None].expand(nz, -1, nx)], -1)
    volume += F.grid_sample(filtered[t][None, None].expand(nz, 1, ny, nx), grid, align_corners=True)[:, 0]

with mrcfile.new(out_path, overwrite=True) as mrc:
    mrc.set_data(-volume.numpy().astype(np.float32))
    mrc.voxel_size = pixel
