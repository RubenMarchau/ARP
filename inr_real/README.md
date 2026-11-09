# INR reconstruction of real even/odd tilt-series

Scripts to fit a grid INR (positivity + second-order TV) to each half of a real, aligned tilt-series, following the
simulation work in `INR_even_odd.ipynb`. Developed on `tomo200528_110` (Warp, 200 kV, 35 tilts, +57° to −45°).

## Input

In `data/real/` (override with `data=...` and `name=...`):

- `{name}_{pixel}Apx_even.mrc`, `{name}_{pixel}Apx_odd.mrc`: frame-split half stacks, aligned, tilt axis along y
- `{name}_20.0Apx.tlt`: tilt angles
- `{name}.xml`: Warp metadata; per-tilt defocus (`GridCTF`), astigmatism and accumulated dose are read from it

## Forward model

For tilt i, the predicted image is

    gain_i * cos(tilt_i) * [ -CTF_i * dose_i ] ∗ ∫ V ds  +  offset_i

- `∫ V ds`: line integral along parallel rays through the model box (sum of samples times the step length)
- `-CTF_i`: raw CTF (the data are not phase flipped), sign-flipped for the teamtomo convention
- `dose_i`: Grant & Grigorieff attenuation exp(-dose / 2Ne). Not `dose_weight_2d`, which normalises over the stack.
- `gain_i`, `offset_i`: learned per tilt; the gains are normalised to geometric mean 1

Training renders random 96 px patches and applies the CTF per patch. The loss is the masked MSE on the inner region,
8 px from the patch edge. Masks:

- alignment fill: low local variance, dilated by 8 px
- the outer 2 image columns
- high-tilt pixels whose rays miss the saved volume

Patches may extend past the image, so every pixel is trained equally often.

## Model

- Coordinates are in units of 20 Å at any pixel size. The box is the 20 Å field of view in y, `thickness_nm` in z,
  and is widened in x so every ray exits through the z faces.
- Dense multi-resolution grids, with cells of 12, 8, 6, 4, 3 and 2 model voxels and 2 features per node:
  - smoothstep interpolation
  - a random sub-cell offset per model
  - border padding
- Small ReLU MLP, with softplus(β=20) output for positivity.
- Second-order TV (L1 norm of the finite-difference Hessian) at random points over the whole box.
  `tv_step` sets the finite-difference step in model voxels.
- Model selection: the checkpoint with the lowest loss on fixed validation patches of the other half.

## Usage

Output paths are prefixes; create the directory first (`mkdir -p runs`).

20 Å, one half (run with `halves=0` and `halves=1`; `seed` differs per half automatically):

    python inr_real/inr_real.py steps=1500 thickness_nm=300 tv=3 grid_lr=1e-2 mlp_lr=1e-3 tilt_gain=1 halves=0 out=runs/real20

Writes `runs/real20_even.mrc` (the volume on the 20 Å grid of the images, `thickness_nm` deep) and `runs/real20_even.pt`.

Optional cosine decay, which continues a run for 1500 more steps down to 1/10 of the learning rate (−6% validation
loss in our tests):

    python inr_real/inr_real.py steps=1500 ... resume=runs/real20_even.pt decay_steps=1500 out=runs/real20_decay

10 Å, warm-started from the 20 Å models (`HALF` is replaced by even/odd), with one extra grid level of 1 model voxel
(20 Å cells):

    python inr_real/inr_real.py steps=1500 pixel=10 thickness_nm=300 tv=3 grid_lr=1e-3 fine_lr=1e-2 mlp_lr=3e-4 \
        tilt_gain=1 scale=<20 A data scale> cells=12,8,6,4,3,2,1 batch=2 density=0.5 \
        resume=runs/real20_HALF.pt halves=0 out=runs/real10

`scale` keeps the density units of the 20 Å model. It is printed by the 20 Å run as `data scale`; it was 0.7725 for the
current data.

Then:

    python inr_real/combine_halves.py runs/real20_even.mrc runs/real20_odd.mrc runs/real20   # average, FSC, FSC-weighted
    python inr_real/render_projections.py thickness_nm=300 tilt_gain=1 ckpt=runs/real20_even.pt out=runs/real20_even_proj
    python inr_real/wbp.py data/real/{name}_20.0Apx_even.mrc data/real/{name}_20.0Apx_odd.mrc data/real/{name}_20.0Apx.tlt 300 runs/wbp.mrc

## Main arguments

| argument | default | meaning |
|---|---|---|
| `pixel` | 20 | pixel size of the stacks to fit (Å) |
| `thickness_nm` | 200 | slab thickness of the model box |
| `steps` | 200 | training steps |
| `tv` / `tv_step` | 3 / 1.0 | TV₂ weight / finite-difference step (model voxels) |
| `grid_lr`, `fine_lr`, `mlp_lr` | 1e-4 | learning rates; `fine_lr` applies to grid levels added on top of a warm start |
| `tilt_gain` | 0 | learn a per-tilt gain |
| `full_coverage` | 1 | patches may extend past the image, plus the volume mask |
| `pixel_jitter` | 0 | random sub-pixel ray positions during training (slower convergence at 1500 steps) |
| `batch`, `density` | 4, 0.5 | patches per step, ray samples per model voxel |
| `cells` | 12,8,6,4,3,2 | grid cell sizes in model voxels |
| `resume`, `decay_start`, `decay_steps` | | warm start and cosine decay of all learning rates to 1/10 |

On a 24 GB Apple M5 Pro, 1500 steps at 20 Å take about 8 minutes per half. At 10 Å (density 0.5, batch 2) they take
about 10 minutes per half. Higher ray densities or supersampling run out of memory and start swapping.
