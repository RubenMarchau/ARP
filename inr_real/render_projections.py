"""Render CTF-filtered full-image projections (and optionally the volume) of a trained model and compare with the data.
Usage: python inr_real/render_projections.py <same model/data args as training> ckpt=run_even.pt out=prefix [volume=vol.mrc]"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
args = dict(a.split("=") for a in sys.argv[1:])
sys.argv = [sys.argv[0]] + [f"{k}={v}" for k, v in args.items() if k not in ("ckpt", "out", "volume")]
import mrcfile, numpy as np, torch, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import inr_real as r

model = r.GridINR().to(r.device)
ckpt = torch.load(args["ckpt"], map_location=r.device)
model.load_state_dict(ckpt.get("state", ckpt), strict=False)  # older checkpoints have no tilt_log_gain
if "volume" in args:
    with mrcfile.new(args["volume"], overwrite=True) as mrc:
        mrc.set_data(r.render_volume(model).numpy().astype(np.float32)); mrc.voxel_size = r.PIXEL_SPACING

det_y, det_x = torch.meshgrid((torch.arange(r.NY, device=r.device) - r.NY // 2) * r.DETECTOR_SCALE,
                              (torch.arange(r.NX, device=r.device) - r.NX // 2) * r.DETECTOR_SCALE, indexing="ij")
det_y, det_x = det_y.flatten().float(), det_x.flatten().float()
predictions = []
with torch.no_grad():
    for tilt in range(r.N_TILTS):
        n = r.n_samples_for(tilt, 1.0)
        line = torch.cat([
            (lambda p: model(p[0]).sum(-1) * p[1])(r.ray_samples(r.m_projection[tilt], det_y[i:i + 8192], det_x[i:i + 8192], n, False))
            for i in range(0, len(det_x), 8192)
        ]).reshape(r.NY, r.NX)
        image = torch.fft.irfftn(torch.fft.rfftn(line) * r.ctf_full[tilt], s=(r.NY, r.NX)) * model.tilt_gain()[tilt] + model.tilt_offset[tilt]
        predictions.append(image.cpu())
predictions = torch.stack(predictions)
data = r.stacks.cpu()
valid = r.valid

with mrcfile.new(args["out"] + ".mrc", overwrite=True) as mrc:
    mrc.set_data((predictions * r.scale).numpy().astype(np.float32)); mrc.voxel_size = r.PIXEL_SPACING

print(f"{'tilt':>6} {'cc even (train)':>16} {'cc odd (val)':>13} {'cc even-odd':>12}")
for t in range(r.N_TILTS):
    m = valid[t]
    cc = lambda a, b: np.corrcoef(a[m].numpy(), b[m].numpy())[0, 1]
    print(f"{r.tilt_angles[t].item():6.0f} {cc(predictions[t], data[0, t]):16.3f} {cc(predictions[t], data[1, t]):13.3f} {cc(data[0, t], data[1, t]):12.3f}")

tilts = [0, 19, 34]
fig, ax = plt.subplots(len(tilts), 4, figsize=(22, 5.5 * len(tilts)))
for row, t in enumerate(tilts):
    lim = dict(vmin=np.percentile(data[1, t], 0.5), vmax=np.percentile(data[1, t], 99.5))
    for col, (name, img) in enumerate([("prediction (-CTF x dose)", predictions[t]), ("even (train)", data[0, t]),
                                        ("odd (validation)", data[1, t]), ("odd - prediction", data[1, t] - predictions[t])]):
        ax[row, col].imshow(img, cmap="gray", **lim)
        ax[row, col].set_title(f"{name}, tilt {r.tilt_angles[t].item():+.0f}")
        ax[row, col].set_axis_off()
plt.tight_layout()
plt.savefig(args["out"] + ".png", dpi=50)
