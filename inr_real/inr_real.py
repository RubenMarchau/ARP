"""Fit a grid INR (positivity + second-order TV) to the even or odd half of a real, aligned tilt-series.

Forward model per tilt: line integral along parallel rays -> -CTF (raw, not phase flipped; per-tilt defocus from the
Warp XML) x dose attenuation x cos(tilt) x learned gain + learned offset. Training renders random patches, applies the
CTF per patch, and scores the inner region (MARGIN from the patch edge) against the data with alignment-fill masks.
The checkpoint with the lowest loss on fixed validation patches of the *other* half is kept.

Model coordinates are in units of MODEL_VOXEL (20 A) at any pixel size, with the box defined by the 20 A images, so a
20 A model can be warm-started at 10 A (resume=..., optionally with extra finer grid levels via cells=...).

Arguments are key=value, see README.md. Example (20 A, even half):
    python inr_real/inr_real.py steps=1500 thickness_nm=300 grid_lr=1e-2 mlp_lr=1e-3 tilt_gain=1 halves=0 out=runs/real20
"""
import copy, math, sys, time, xml.etree.ElementTree as ET
from pathlib import Path

import mrcfile, numpy as np, torch, torch.nn.functional as F
from scipy.ndimage import binary_dilation, uniform_filter
from torch_ctf import calculate_ctf_2d
from torch_fourier_filter.dose_weight import _apply_voltage_correction, _critical_exposure_grid
from torch_grid_utils import fftfreq_grid
from torch_affine_utils.transforms_3d import Ry

args = dict(a.split("=") for a in sys.argv[1:])
N_STEPS = int(args.get("steps", 200))
TV_WEIGHT = float(args.get("tv", 3.0))
OUT = args.get("out", "")
HALVES = [int(h) for h in args.get("halves", "0").split(",")]

DATA = Path(args.get("data", "data/real"))
NAME = args.get("name", "tomo200528_110")
PIXEL_SPACING = float(args.get("pixel", 20.0))
MODEL_VOXEL = 20.0  # model coordinates are in units of the 20 A runs
DETECTOR_SCALE = PIXEL_SPACING / MODEL_VOXEL  # detector pixel -> model units
REF_NY, REF_NX = 320, 330  # 20 A image size that defines the model box
VOLTAGE = 200
THICKNESS = float(args.get("thickness_nm", 200)) * 10 / MODEL_VOXEL  # in model voxels
PATCH, MARGIN = 96, 8
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

even = torch.tensor(mrcfile.read(DATA / f"{NAME}_{PIXEL_SPACING:.1f}Apx_even.mrc")).float()
odd = torch.tensor(mrcfile.read(DATA / f"{NAME}_{PIXEL_SPACING:.1f}Apx_odd.mrc")).float()
tilt_angles = torch.tensor(np.loadtxt(DATA / f"{NAME}_20.0Apx.tlt"), dtype=torch.float32)
N_TILTS, NY, NX = even.shape

root = ET.parse(DATA / f"{NAME}.xml").getroot()
grid_values = lambda tag: torch.tensor([float(n.get("Value")) for n in root.find(tag)])
defocus = grid_values("GridCTF")
astigmatism = grid_values("GridCTFDefocusDelta")
astigmatism_angle = grid_values("GridCTFDefocusAngle")
dose = torch.tensor([float(v) for v in root.find("Dose").text.split()])
assert torch.allclose(torch.tensor([float(v) for v in root.find("Angles").text.split()]), tilt_angles, atol=0.02)

stack = even.double().numpy()
# alignment fill: low local variance (the fill carries a little noise after binning), dilated by MARGIN to also cover
# CTF delocalisation; plus the outer 2 columns (binning wrap-around). Border rows stay: y is the tilt axis, so volume
# row y is only seen by detector row y and masking it would leave that volume slice unconstrained
local_var = uniform_filter(stack ** 2, (1, 5, 5), mode="nearest") - uniform_filter(stack, (1, 5, 5), mode="nearest") ** 2
fill = local_var < 0.2 * np.median(local_var, axis=(1, 2), keepdims=True)
fill = binary_dilation(fill, structure=np.ones((1, 2 * MARGIN + 1, 2 * MARGIN + 1)))
fill[:, :, :2], fill[:, :, -2:] = True, True
valid = torch.tensor(~fill)
print("valid fraction", valid.float().mean().item())

# normalise: subtract mean of valid pixels per tilt, divide by global std
stacks = torch.stack([even, odd])
means = torch.stack([torch.stack([s[i][valid[i]].mean() for i in range(N_TILTS)]) for s in stacks])
stacks = stacks - means[..., None, None]
scale = float(args["scale"]) if "scale" in args else stacks[:, valid].std().item()  # pass the 20 A scale to keep density units
stacks = (stacks / scale).to(device)
print("data scale", scale)
valid_d = valid.to(device)


def microscope_filter(shape):
    """-CTF (raw, not phase flipped) x dose weighting x cos(tilt), per tilt, rfft layout."""
    ctf = -calculate_ctf_2d(
        defocus=defocus, astigmatism=astigmatism,
        astigmatism_angle=astigmatism_angle, voltage=VOLTAGE, spherical_aberration=2.7,
        amplitude_contrast=0.07, phase_shift=0.0, pixel_size=PIXEL_SPACING, image_shape=shape, rfft=True, fftshift=False,
    )
    ctf[..., fftfreq_grid(image_shape=shape, rfft=True, fftshift=False, norm=True) > 0.5] = 0
    # per-tilt Grant & Grigorieff attenuation exp(-dose / 2Ne); dose_weight_2d also normalises over the stack, which we don't want
    critical = _critical_exposure_grid(image_shape=shape, pixel_size=PIXEL_SPACING, rfft=True, fftshift=False)
    ctf = ctf * torch.exp(-_apply_voltage_correction(dose, VOLTAGE)[:, None, None] / (2 * critical))
    ctf = ctf * torch.cos(torch.deg2rad(tilt_angles))[:, None, None]
    return ctf.float().to(device)


ctf_patch = microscope_filter((PATCH, PATCH))
ctf_full = microscope_filter((NY, NX))
m_projection = Ry(tilt_angles)[..., :3, :3].float().to(device)

# model box: z slab of THICKNESS, y the image, x wide enough that every ray of every tilt enters and exits via z faces
max_tilt = math.radians(tilt_angles.abs().max().item())
half_x = math.ceil((REF_NX / 2 + THICKNESS / 2 * math.sin(max_tilt)) / math.cos(max_tilt))
BOX_LOW = torch.tensor([-half_x, -(REF_NY // 2) - 0.5, -THICKNESS / 2], device=device)
BOX_HIGH = torch.tensor([half_x, REF_NY - REF_NY // 2 - 0.5, THICKNESS / 2], device=device)
BOX_CENTER, BOX_HALF = (BOX_LOW + BOX_HIGH) / 2, (BOX_HIGH - BOX_LOW) / 2
print("model box (x, y, z) voxels:", (2 * BOX_HALF).tolist())


def ray_samples(rotation, y, x, n_samples, jitter):
    origin = rotation[:, 0] * x[:, None] + rotation[:, 1] * y[:, None]
    direction = rotation[:, 2]
    safe = torch.where(direction.abs() < 1e-6, torch.full_like(direction, 1e-6), direction)
    t1, t2 = (BOX_LOW - origin) / safe, (BOX_HIGH - origin) / safe
    t_near = torch.minimum(t1, t2).amax(-1)
    length = (torch.maximum(t1, t2).amin(-1) - t_near).clamp(min=0)
    u = ((torch.arange(n_samples, device=device) + 0.5) / n_samples).expand(len(x), n_samples)
    if jitter:
        u = u + (torch.rand_like(u) - 0.5) / n_samples
    t = t_near[:, None] + u * length[:, None]
    return origin[:, None] + t[..., None] * direction, length / n_samples


def n_samples_for(tilt, density):
    return max(8, math.ceil(density * THICKNESS / math.cos(math.radians(tilt_angles[tilt].item()))))


def smoothstep_coordinates(u, n_nodes):
    g = (u + 1) / 2 * (n_nodes - 1)
    cell = g.floor()
    t = g - cell
    return (cell + t * t * (3 - 2 * t)) / (n_nodes - 1) * 2 - 1


class GridINR(torch.nn.Module):
    def __init__(self, cell_sizes=tuple(float(c) for c in args.get("cells", "12,8,6,4,3,2").split(",")), n_features=2, width=64):
        super().__init__()
        size = (2 * BOX_HALF).cpu()
        nodes = [torch.ceil(size / c).long() + 1 for c in cell_sizes]  # (x, y, z) nodes per level
        self.grids = torch.nn.ParameterList([
            torch.nn.Parameter(torch.empty(1, n_features, n[2] + 2, n[1] + 2, n[0] + 2).uniform_(-1e-4, 1e-4)) for n in nodes
        ])
        cell = torch.stack([2 / (n - 1).float() for n in nodes])  # (levels, 3) in [-1, 1] units
        self.register_buffer("extent", 1 + cell)
        self.register_buffer("offsets", (torch.rand(len(nodes), 3) - 0.5) * cell)
        self.register_buffer("n_nodes", torch.stack(nodes).float() + 2)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(n_features * len(nodes), width), torch.nn.ReLU(),
            torch.nn.Linear(width, width), torch.nn.ReLU(), torch.nn.Linear(width, 1),
        )
        torch.nn.init.constant_(self.mlp[-1].bias, -0.2)
        self.tilt_offset = torch.nn.Parameter(torch.zeros(N_TILTS))  # background level per image
        self.tilt_log_gain = torch.nn.Parameter(torch.zeros(N_TILTS), requires_grad=TILT_GAIN)  # contrast per image

    def tilt_gain(self):
        """Per-tilt gain with geometric mean 1, so it cannot trade off against the overall density scale."""
        return torch.exp(self.tilt_log_gain - self.tilt_log_gain.mean())

    def forward(self, xyz):
        box = ((xyz - BOX_CENTER) / BOX_HALF).reshape(1, 1, 1, -1, 3)
        features = torch.cat([
            F.grid_sample(level, smoothstep_coordinates((box + offset) / extent, n), align_corners=True,
                          padding_mode="border").reshape(level.shape[1], -1)
            for level, offset, extent, n in zip(self.grids, self.offsets, self.extent, self.n_nodes)
        ])
        return F.softplus(self.mlp(features.T).reshape(xyz.shape[:-1]), beta=20)


TV_STEP = float(args.get("tv_step", 1.0))  # finite-difference step in model voxels (0.5 = 10 A)


def second_order_tv(model, n_points=65536):
    """Mean Frobenius norm of the finite-difference Hessian (in units of model voxels squared, so the step size only
    changes which scales it sees, not the overall strength) at random points anywhere in the box."""
    points = torch.rand(n_points, 3, device=device) * (BOX_HIGH - BOX_LOW) + BOX_LOW
    steps = torch.eye(3, device=device) * TV_STEP
    values = model(points)
    hessian = []
    for i in range(3):
        hessian.append(model(points + steps[i]) - 2 * values + model(points - steps[i]))
        for j in range(i + 1, 3):
            mixed = (model(points + steps[i] + steps[j]) - model(points + steps[i] - steps[j])
                     - model(points - steps[i] + steps[j]) + model(points - steps[i] - steps[j])) / 4
            hessian += [mixed, mixed]
    return torch.sqrt(torch.stack(hessian).pow(2).sum(0) + 1e-12).mean() / TV_STEP ** 2


patch_1d = torch.arange(PATCH, device=device)


SUPERSAMPLE = int(args.get("supersample", 1))  # render at PIXEL_SPACING / SUPERSAMPLE, then Fourier-crop
PIXEL_JITTER = bool(int(args.get("pixel_jitter", 0)))  # random sub-pixel ray positions during training (slower at 1500 steps)
TILT_GAIN = bool(int(args.get("tilt_gain", 0)))  # learn a contrast gain per tilt on top of cos(tilt)
FULL_COVERAGE = bool(int(args.get("full_coverage", 1)))  # patches may extend past the image; mask rays missing the volume


def fourier_crop(spectrum, n):
    """Keep the n x (n // 2 + 1) lowest frequencies of an rfft2 spectrum (n even)."""
    return torch.cat([spectrum[..., : n // 2, : n // 2 + 1], spectrum[..., -(n // 2):, : n // 2 + 1]], dim=-2)


def render_patch(model, tilt, y0, x0, density, jitter):
    """CTF-filtered projection of a PATCH x PATCH detector region with top-left pixel (y0, x0)."""
    fine = PATCH * SUPERSAMPLE
    offsets = torch.arange(fine, device=device) / SUPERSAMPLE  # fine sample SUPERSAMPLE * k sits on pixel centre k
    py, px = torch.meshgrid((offsets + y0 - NY // 2) * DETECTOR_SCALE, (offsets + x0 - NX // 2) * DETECTOR_SCALE, indexing="ij")
    py, px = py.flatten().float(), px.flatten().float()
    if jitter and PIXEL_JITTER:  # random sub-pixel ray position: a box-filtered pixel on average
        py = py + (torch.rand_like(py) - 0.5) / SUPERSAMPLE * DETECTOR_SCALE
        px = px + (torch.rand_like(px) - 0.5) / SUPERSAMPLE * DETECTOR_SCALE
    points, dt = ray_samples(m_projection[tilt], py, px, n_samples_for(tilt, density), jitter)
    image = (model(points).sum(-1) * dt).reshape(fine, fine)
    spectrum = fourier_crop(torch.fft.rfftn(image, norm="forward"), PATCH)
    image = torch.fft.irfftn(spectrum * ctf_patch[tilt], s=(PATCH, PATCH), norm="forward")
    return image * model.tilt_gain()[tilt] + model.tilt_offset[tilt]


# detector pixels whose rays cross the saved volume (|x| <= NX/2, |z| <= THICKNESS/2)
detector_x = (torch.arange(NX) - NX // 2).abs().float() * DETECTOR_SCALE
reach = REF_NX / 2 * torch.cos(torch.deg2rad(tilt_angles)) + THICKNESS / 2 * torch.sin(torch.deg2rad(tilt_angles.abs()))
sees_volume = (detector_x[None, :] <= reach[:, None] + 1)[:, None, :].expand(N_TILTS, NY, NX)
train_valid = valid & sees_volume
print("training pixels dropped by the volume mask:", (valid & ~sees_volume).sum().item() / valid.sum().item())

# pad by PATCH so that patches may extend past the image; padded pixels are masked out
pad = (PATCH, PATCH, PATCH, PATCH)
stacks_padded = F.pad(stacks, pad)
masks_padded = {"train": F.pad(train_valid.float(), pad).to(device), "validation": F.pad(valid.float(), pad).to(device)}


def patch_loss(model, half, patches, density, jitter, mask_name="validation"):
    """Masked MSE over the inner region (MARGIN from the patch edge) of each patch."""
    total, count = 0.0, 0
    for tilt, y0, x0 in patches:
        prediction = render_patch(model, tilt, y0, x0, density, jitter)[MARGIN:-MARGIN, MARGIN:-MARGIN]
        rows = slice(PATCH + y0 + MARGIN, PATCH + y0 + PATCH - MARGIN)
        cols = slice(PATCH + x0 + MARGIN, PATCH + x0 + PATCH - MARGIN)
        target = stacks_padded[half, tilt, rows, cols]
        mask = masks_padded[mask_name][tilt, rows, cols]
        total = total + ((prediction - target) ** 2 * mask).sum()
        count = count + mask.sum()
    return total / count.clamp(min=1)


def random_patches(n, generator=None):
    return [(torch.randint(N_TILTS, (1,), generator=generator).item(),
             torch.randint(NY - PATCH + 1, (1,), generator=generator).item(),
             torch.randint(NX - PATCH + 1, (1,), generator=generator).item()) for _ in range(n)]


def training_patches(n):
    """With FULL_COVERAGE every image pixel is equally likely to fall in a patch's inner region."""
    if not FULL_COVERAGE:
        return random_patches(n)
    low = -(PATCH - MARGIN) + 1
    return [(torch.randint(N_TILTS, (1,)).item(),
             torch.randint(low, NY - MARGIN, (1,)).item(),
             torch.randint(low, NX - MARGIN, (1,)).item()) for _ in range(n)]


validation_patches = random_patches(3 * N_TILTS, torch.Generator().manual_seed(0))  # unchanged, for comparability


DENSITY = float(args.get("density", 0.5))  # ray samples per model voxel (20 A) once training is under way
BATCH = int(args.get("batch", 4))  # patches per step


def load_expanded(model, state):
    """Load a state dict with fewer grid levels: old levels copied, new levels keep their init, and the MLP weights
    that read the new levels start at zero, so the warm-started model initially equals the old one."""
    own = model.state_dict()
    n_old = sum(1 for k in state if k.startswith("grids."))
    for key, value in state.items():
        if key in ("offsets", "extent", "n_nodes"):
            own[key][:n_old] = value
        elif key == "mlp.0.weight":
            own[key].zero_()
            own[key][:, : value.shape[1]] = value
        else:
            own[key] = value
    model.load_state_dict(own)
    return n_old


def train(train_half, val_half, n_steps, batch_size=BATCH, lr=1e-4, tv_weight=TV_WEIGHT, warmup_steps=100,
          density_schedule=((0, 0.25), (200, 0.5)), eval_every=100, seed=1):
    torch.manual_seed(seed)
    model = GridINR().to(device)
    n_old = len(model.grids)  # levels trained at the old learning rate; new (finer) levels may use fine_lr
    if "resume" in args:  # warm start (optimizer state is not saved, so Adam restarts)
        resume = args["resume"].replace("HALF", ["even", "odd"][train_half])
        state = torch.load(resume, map_location=device)
        n_old = load_expanded(model, state.get("state", state))
        density_schedule, warmup_steps = ((0, DENSITY),), 100
        print("warm start from", resume)
    print("parameters", sum(p.numel() for p in model.parameters()))
    optimizer = torch.optim.Adam([
        {"params": list(model.grids.parameters())[:n_old], "lr": float(args.get("grid_lr", lr))},
        {"params": list(model.grids.parameters())[n_old:], "lr": float(args.get("fine_lr", args.get("grid_lr", lr)))},
        {"params": model.mlp.parameters(), "lr": float(args.get("mlp_lr", lr))},
        {"params": [model.tilt_offset, model.tilt_log_gain], "lr": 1e-2},
    ])
    decay_steps = int(args.get("decay_steps", 0))  # cosine decay to 1/10 of every learning rate over decay_steps
    decay_start = int(args.get("decay_start", 0))  # constant learning rate before this step

    def lr_factor(s):
        factor = min(1.0, (s + 1) / warmup_steps)
        if decay_steps:
            progress = min(max(s - decay_start, 0), decay_steps) / decay_steps
            factor *= 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))
        return factor

    warmup = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    history = {"step": [], "val": []}
    best = {"val": float("inf"), "step": 0}
    start = time.time()
    for step in range(1, n_steps + 1):
        density = [d for s, d in density_schedule if step >= s][-1]
        loss = patch_loss(model, train_half, training_patches(batch_size), density, jitter=True,
                          mask_name="train" if FULL_COVERAGE else "validation")
        optimizer.zero_grad()
        (loss + tv_weight * second_order_tv(model) if tv_weight > 0 else loss).backward()
        optimizer.step()
        warmup.step()
        if step % eval_every == 0 or step == n_steps:
            with torch.no_grad():
                val = patch_loss(model, val_half, validation_patches, DENSITY, jitter=False).item()
            history["step"].append(step); history["val"].append(val)
            if val < best["val"]:
                best = {"val": val, "step": step, "state": copy.deepcopy(model.state_dict())}
            if OUT:
                torch.save(model.state_dict(), f"{OUT}_{['even', 'odd'][train_half]}_latest.pt")
            if device == "mps":
                torch.mps.empty_cache()  # release cached blocks so memory does not creep up over a run
            print(f"step {step} train {loss.item():.4f} val {val:.5f} ({time.time() - start:.0f} s)", flush=True)
    model.load_state_dict(best["state"])
    return model, history, best["step"]


def render_volume(model):
    """Voxel centres [z, y, x] at PIXEL_SPACING over the field of view of the 0 degree image."""
    nz = round(THICKNESS / DETECTOR_SCALE)
    z = ((torch.arange(nz, device=device) + 0.5) * DETECTOR_SCALE - THICKNESS / 2).float()
    y = ((torch.arange(NY, device=device) - NY // 2) * DETECTOR_SCALE).float()
    x = ((torch.arange(NX, device=device) - NX // 2) * DETECTOR_SCALE).float()
    with torch.no_grad():
        out = []
        for zi in z:
            yy, xx = torch.meshgrid(y, x, indexing="ij")
            xyz = torch.stack([xx, yy, torch.full_like(xx, zi)], -1).reshape(-1, 3).float()
            out.append(model(xyz).reshape(len(y), len(x)).cpu())
    return torch.stack(out) * scale


if __name__ == "__main__":
    for half in HALVES:
        model, history, best_step = train(half, 1 - half, N_STEPS, seed=1 + half)
        if OUT:
            vol = render_volume(model)
            name = ["even", "odd"][half]
            with mrcfile.new(f"{OUT}_{name}.mrc", overwrite=True) as mrc:
                mrc.set_data(vol.numpy().astype(np.float32)); mrc.voxel_size = PIXEL_SPACING
            torch.save({"history": history, "best_step": best_step, "state": model.state_dict()}, f"{OUT}_{name}.pt")
