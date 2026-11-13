"""Noise2Noise denoiser between an even/odd pair of reconstructions (as in cryo-CARE / Noise2Inverse), small 3D U-Net.

Usage: python inr_real/n2n_denoise.py even.mrc odd.mrc out_prefix [steps=4000 patch=64 batch=4 width=16 lr=3e-4]
Trains on both directions (even -> odd, odd -> even). A block of rows along y is held out to pick the checkpoint with the
lowest loss. Writes out_prefix_{even,odd,average}.mrc and out_prefix.pt.
The odd half is the training target here, so it no longer gives an independent validation of the denoised volumes.
Only flips that keep the wedge are used: y, and x with z together (a 180 degree rotation about the tilt axis y).
"""
import copy
import sys
import time

import mrcfile
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

even_path, odd_path, prefix = sys.argv[1:4]
args = dict(a.split("=", 1) for a in sys.argv[4:])
STEPS, PATCH, BATCH, WIDTH = (int(args.get(k, d)) for k, d in [("steps", 4000), ("patch", 64), ("batch", 4), ("width", 16)])
LR, EVAL_EVERY = float(args.get("lr", 3e-4)), int(args.get("eval_every", 200))
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

with mrcfile.open(even_path) as mrc:
    pixel = float(mrc.voxel_size.x)
    even = torch.tensor(mrc.data, dtype=torch.float32)
odd = torch.tensor(mrcfile.read(odd_path), dtype=torch.float32)
mean, std = ((even + odd) / 2).mean(), ((even + odd) / 2).std()
halves = torch.stack([(even - mean) / std, (odd - mean) / std]).to(device)
nz, ny, nx = even.shape
train_ny = ny - max(ny // 5, PATCH)  # rows from train_ny on are held out


class Block(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.net = nn.Sequential(nn.Conv3d(n_in, n_out, 3, padding=1), nn.LeakyReLU(0.1), nn.Conv3d(n_out, n_out, 3, padding=1), nn.LeakyReLU(0.1))

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    """Three downsamplings: the receptive field (about 70 voxels) is well beyond the 6-10 voxel grain."""

    def __init__(self, width):
        super().__init__()
        self.down = nn.ModuleList([Block(1, width), Block(width, 2 * width), Block(2 * width, 4 * width)])
        self.middle = Block(4 * width, 8 * width)
        self.up = nn.ModuleList([Block(12 * width, 4 * width), Block(6 * width, 2 * width), Block(3 * width, width)])
        self.out = nn.Conv3d(width, 1, 1)
        nn.init.zeros_(self.out.weight), nn.init.zeros_(self.out.bias)  # start at the identity

    def forward(self, x):
        skips, y = [], x
        for block in self.down:
            y = block(y)
            skips.append(y)
            y = F.avg_pool3d(y, 2)
        y = self.middle(y)
        for block, skip in zip(self.up, reversed(skips)):
            y = y.repeat_interleave(2, 2).repeat_interleave(2, 3).repeat_interleave(2, 4)
            y = block(torch.cat([y, skip], 1))
        return x + self.out(y)


def crop(direction, z, y, x):
    return (halves[direction, z:z + PATCH, y:y + PATCH, x:x + PATCH],
            halves[1 - direction, z:z + PATCH, y:y + PATCH, x:x + PATCH])


def training_batch():
    inputs, targets = [], []
    for _ in range(BATCH):
        a, b = crop(torch.randint(2, (1,)).item(), torch.randint(nz - PATCH + 1, (1,)).item(),
                    torch.randint(train_ny - PATCH + 1, (1,)).item(), torch.randint(nx - PATCH + 1, (1,)).item())
        if torch.rand(1) < 0.5:
            a, b = a.flip(1), b.flip(1)
        if torch.rand(1) < 0.5:
            a, b = a.flip(0, 2), b.flip(0, 2)
        inputs.append(a), targets.append(b)
    return torch.stack(inputs)[:, None], torch.stack(targets)[:, None]


generator = torch.Generator().manual_seed(0)
held_out = [crop(d, torch.randint(nz - PATCH + 1, (1,), generator=generator).item(),
                 torch.randint(train_ny, ny - PATCH + 1, (1,), generator=generator).item(),
                 torch.randint(nx - PATCH + 1, (1,), generator=generator).item()) for d in (0, 1) for _ in range(16)]
held_in, held_target = (torch.stack(v)[:, None] for v in zip(*held_out))


@torch.no_grad()
def held_out_loss(model):
    return float(np.mean([F.mse_loss(model(held_in[i:i + BATCH]), held_target[i:i + BATCH]).item() for i in range(0, len(held_in), BATCH)]))


@torch.no_grad()
def denoise(model, volume, tile=96, overlap=32):
    """Tiled inference with linear blending (no tile seams, bounded memory)."""
    ramp = torch.minimum(torch.arange(1, tile + 1), torch.arange(tile, 0, -1)).float().clamp(max=overlap) / overlap
    weight = (ramp[:, None, None] * ramp[None, :, None] * ramp[None, None, :]).to(device)
    out, total = torch.zeros_like(volume), torch.zeros_like(volume)
    starts = [sorted({*range(0, n - tile, tile - overlap), n - tile}) for n in volume.shape]
    for z in starts[0]:
        for y in starts[1]:
            for x in starts[2]:
                out[z:z + tile, y:y + tile, x:x + tile] += model(volume[z:z + tile, y:y + tile, x:x + tile][None, None])[0, 0] * weight
                total[z:z + tile, y:y + tile, x:x + tile] += weight
    return out / total


if __name__ == "__main__":
    torch.manual_seed(0)
    model = UNet(WIDTH).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    print("parameters", sum(p.numel() for p in model.parameters()), "held-out rows from", train_ny, flush=True)
    print(f"identity (input as prediction) held-out loss {held_out_loss(model):.5f}", flush=True)
    best = {"val": float("inf"), "step": 0}
    start = time.time()
    for step in range(1, STEPS + 1):
        inputs, targets = training_batch()
        loss = F.mse_loss(model(inputs), targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % EVAL_EVERY == 0 or step == STEPS:
            val = held_out_loss(model)
            if val < best["val"]:
                best = {"val": val, "step": step, "state": copy.deepcopy(model.state_dict())}
                torch.save({"state": best["state"], "step": step, "val": val, "mean": mean, "std": std}, prefix + ".pt")
            print(f"step {step} train {loss.item():.5f} val {val:.5f} ({time.time() - start:.0f} s)", flush=True)
            if device == "mps":
                torch.mps.empty_cache()
    model.load_state_dict(best["state"])
    print("best step", best["step"], "val", best["val"], flush=True)
    torch.save({"state": model.state_dict(), "step": best["step"], "val": best["val"], "mean": mean, "std": std}, prefix + ".pt")
    den = [denoise(model, halves[h]) * std.to(device) + mean.to(device) for h in (0, 1)]
    for name, volume in [("even", den[0]), ("odd", den[1]), ("average", (den[0] + den[1]) / 2)]:
        with mrcfile.new(f"{prefix}_{name}.mrc", overwrite=True) as mrc:
            mrc.set_data(volume.cpu().numpy().astype(np.float32))
            mrc.voxel_size = pixel
