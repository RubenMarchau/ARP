"""Average an even/odd pair of INR volumes, compute the half-map FSC, and write an FSC-weighted average.

Usage: python inr_real/combine_halves.py even.mrc odd.mrc out_prefix
Writes out_prefix_average.mrc, out_prefix_average_fsc_weighted.mrc and out_prefix_fsc.png.
The FSC weight per shell is sqrt(2 FSC / (1 + FSC)). Note: the half-map FSC of INR pairs is inflated by the priors
both halves share (positivity, TV), so treat it as an upper bound.
"""
import sys

import matplotlib
import mrcfile
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def frequency_grid(shape, pixel):
    """|k| in 1/A on the rfftn grid of a volume."""
    axes = [torch.fft.fftfreq(shape[0]), torch.fft.fftfreq(shape[1]), torch.fft.rfftfreq(shape[2])]
    return torch.sqrt(sum(f ** 2 for f in torch.meshgrid(*axes, indexing="ij"))) / pixel


def half_map_fsc(a, b, pixel, n_shells):
    """FSC in n_shells equal-width shells up to Nyquist, with a soft cosine edge (8 voxels) against box-edge leakage."""
    window = torch.ones_like(a)
    for dim, n in enumerate(a.shape):
        w = torch.ones(n, dtype=a.dtype)
        r = min(8, n // 4)
        w[:r] = 0.5 - 0.5 * torch.cos(torch.pi * torch.arange(r) / r)
        w[-r:] = w[:r].flip(0)
        window = window * w.reshape([-1 if d == dim else 1 for d in range(a.ndim)])
    fa, fb = torch.fft.rfftn(a * window), torch.fft.rfftn(b * window)
    edges = torch.linspace(0, 0.5 / pixel, n_shells + 1)
    shell = torch.bucketize(frequency_grid(a.shape, pixel), edges[1:-1]).flatten()

    def shell_sum(values):
        return torch.zeros(n_shells, dtype=torch.float64).index_add_(0, shell, values.flatten())

    fsc = shell_sum((fa * fb.conj()).real) / torch.sqrt(shell_sum(fa.abs() ** 2) * shell_sum(fb.abs() ** 2))
    return ((edges[:-1] + edges[1:]) / 2).numpy(), fsc.numpy()


if __name__ == "__main__":
    even_path, odd_path, prefix = sys.argv[1:4]
    with mrcfile.open(even_path) as mrc:
        pixel = float(mrc.voxel_size.x)
        even = torch.tensor(mrc.data).double()
    odd = torch.tensor(mrcfile.read(odd_path)).double()
    average = (even + odd) / 2

    k, fsc = half_map_fsc(even, odd, pixel, n_shells=min(even.shape) // 2)
    weight = np.sqrt(2 * np.clip(fsc, 0, None) / (1 + np.clip(fsc, 0, None)))
    voxel_weight = torch.tensor(np.interp(frequency_grid(even.shape, pixel).numpy(), k, weight))  # smooth in |k|
    weighted = torch.fft.irfftn(torch.fft.rfftn(average) * voxel_weight, s=average.shape)

    for name, volume in [("average", average), ("average_fsc_weighted", weighted)]:
        with mrcfile.new(f"{prefix}_{name}.mrc", overwrite=True) as mrc:
            mrc.set_data(volume.float().numpy())
            mrc.voxel_size = pixel

    below = np.where(fsc[1:] < 0.143)[0]
    print("half-map FSC 0.143 at", f"{1 / k[1:][below[0]]:.0f} A" if len(below) else f"beyond Nyquist ({2 * pixel:.0f} A)")
    print("even/odd volume cc:", np.corrcoef(even.flatten().numpy(), odd.flatten().numpy())[0, 1])
    plt.plot(k, fsc)
    plt.axhline(0.143, color="gray", ls=":")
    plt.axhline(0.5, color="gray", ls=":")
    plt.xlabel("spatial frequency (1/A)")
    plt.ylabel("half-map FSC")
    plt.savefig(f"{prefix}_fsc.png", dpi=80)
