from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _as_numpy(volume: Any) -> np.ndarray:
    if hasattr(volume, "detach"):
        volume = volume.detach().cpu().numpy()
    return np.asarray(volume)


def _within_bounds(shape: Sequence[int], xyz: Sequence[int], radius: int) -> bool:
    z_size, y_size, x_size = shape
    x, y, z = xyz
    return (
        radius <= x < x_size - radius
        and radius <= y < y_size - radius
        and radius <= z < z_size - radius
    )


def _crop(volume: np.ndarray, xyz: Sequence[int], radius: int) -> np.ndarray:
    x, y, z = xyz
    return volume[
        z - radius : z + radius + 1,
        y - radius : y + radius + 1,
        x - radius : x + radius + 1,
    ].copy()


def _soft_spherical_mask(shape: Sequence[int], radius: int, edge_width: int = 3) -> np.ndarray:
    center = (np.asarray(shape) - 1) / 2
    z, y, x = np.ogrid[: shape[0], : shape[1], : shape[2]]
    distance = np.sqrt((z - center[0]) ** 2 + (y - center[1]) ** 2 + (x - center[2]) ** 2)
    inner_radius = max(0, radius - edge_width)
    mask = np.ones(shape, dtype=np.float32)
    mask[distance >= radius] = 0
    transition = (distance > inner_radius) & (distance < radius)
    mask[transition] = 0.5 * (
        1 + np.cos(np.pi * (distance[transition] - inner_radius) / edge_width)
    )
    return mask


def _fsc_curve(
    first_map: np.ndarray,
    second_map: np.ndarray,
    pixel_spacing: float,
    threshold: float = 0.143,
) -> tuple[np.ndarray, np.ndarray, float | None]:
    if first_map.shape != second_map.shape or len(first_map.shape) != 3:
        raise ValueError("FSC maps must have the same three-dimensional shape.")
    if len(set(first_map.shape)) != 1:
        raise ValueError("FSC requires cubic maps with isotropic voxels.")

    size = first_map.shape[0]
    nyquist = 1 / (2 * pixel_spacing)
    frequency_step = 1 / (size * pixel_spacing)
    max_shell = int(np.floor(nyquist / frequency_step))
    fz, fy, fx = np.meshgrid(
        np.fft.fftfreq(size, d=pixel_spacing),
        np.fft.fftfreq(size, d=pixel_spacing),
        np.fft.fftfreq(size, d=pixel_spacing),
        indexing="ij",
    )
    radial_frequency = np.sqrt(fz**2 + fy**2 + fx**2)
    shell = np.rint(radial_frequency / frequency_step).astype(np.int32)
    within_nyquist = (radial_frequency <= nyquist) & (shell <= max_shell)

    first_ft = np.fft.fftn(first_map)
    second_ft = np.fft.fftn(second_map)
    first_power = np.abs(first_ft) ** 2
    second_power = np.abs(second_ft) ** 2
    cross_power = (first_ft * np.conj(second_ft)).real
    shell_ids = shell[within_nyquist]

    cross_sum = np.bincount(
        shell_ids, weights=cross_power[within_nyquist], minlength=max_shell + 1
    )
    first_sum = np.bincount(
        shell_ids, weights=first_power[within_nyquist], minlength=max_shell + 1
    )
    second_sum = np.bincount(
        shell_ids, weights=second_power[within_nyquist], minlength=max_shell + 1
    )
    denominator = np.sqrt(first_sum * second_sum)
    fsc = np.full(max_shell + 1, np.nan, dtype=np.float64)
    valid_shells = denominator > 0
    fsc[valid_shells] = cross_sum[valid_shells] / denominator[valid_shells]
    frequencies = np.arange(max_shell + 1, dtype=np.float64) * frequency_step
    fsc[0] = np.nan

    crossing_frequency = None
    for shell_index in range(2, len(fsc)):
        previous, current = fsc[shell_index - 1], fsc[shell_index]
        if np.isfinite(previous) and np.isfinite(current) and previous > threshold >= current:
            fraction = (previous - threshold) / (previous - current)
            crossing_frequency = (
                frequencies[shell_index - 1]
                + fraction * (frequencies[shell_index] - frequencies[shell_index - 1])
            )
            break
    resolution = 1 / crossing_frequency if crossing_frequency else None
    return frequencies, fsc, resolution


def analyze_protein_resolution(
    *,
    protein_data: pd.DataFrame,
    fbp_particle_lists: Mapping[str, Sequence[tuple[int, Any]]],
    fbp_tomogram_shapes: Mapping[int, Sequence[int]],
    tomograms_by_method: Mapping[str, Mapping[int, Any]],
    template_volumes: Mapping[str, Any],
    radii: Mapping[str, int],
    rotation_parser: Callable[[Any], np.ndarray],
    unrotate_fn: Callable[..., np.ndarray],
    center_offset_fn: Callable[[Any], np.ndarray],
    pixel_spacing: float,
) -> tuple[
    dict[str, dict[str, list[dict[str, Any]]]],
    dict[str, dict[str, np.ndarray]],
    pd.DataFrame,
    dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray]],
]:
    if pixel_spacing <= 0:
        raise ValueError("Pixel spacing must be positive.")
    method_names = ["Ground truth", "FBP", "DeepDeWedge", "ICecream"]
    required_methods = {"Ground truth", "DeepDeWedge", "ICecream"}
    if not required_methods.issubset(tomograms_by_method):
        raise ValueError(f"Missing tomogram mappings for {required_methods - set(tomograms_by_method)}.")

    proteins = ("ribosome", "cohesin")
    volumes = {
        method: {
            int(tomogram_id): _as_numpy(volume)
            for tomogram_id, volume in mapping.items()
        }
        for method, mapping in tomograms_by_method.items()
    }

    fbp_by_tomogram: dict[str, dict[int, list[np.ndarray]]] = {}
    for protein in proteins:
        fbp_by_tomogram[protein] = defaultdict(list)
        for tomogram_id, particle in fbp_particle_lists[protein]:
            fbp_by_tomogram[protein][int(tomogram_id)].append(_as_numpy(particle))

    aligned_subvolumes: dict[str, dict[str, list[dict[str, Any]]]] = {
        protein: {method: [] for method in method_names} for protein in proteins
    }
    center_offsets = {
        protein: center_offset_fn(template_volumes[protein]) for protein in proteins
    }
    fbp_cursors: dict[str, dict[int, int]] = {
        protein: defaultdict(int) for protein in proteins
    }

    for row_position, (_, row) in enumerate(protein_data.iterrows()):
        protein = str(row["protein_name"])
        if protein not in proteins:
            continue

        tomogram_id = int(row["subtomo_id"])
        coordinate_text = str(row["sub_loc"]).replace("np.int64", "")
        xyz = [int(value) for value in re.findall(r"-?\d+", coordinate_text)]
        if len(xyz) != 3:
            raise ValueError(f"Expected three particle coordinates in row {row_position}.")
        radius = int(radii[protein])
        fbp_valid = _within_bounds(fbp_tomogram_shapes[tomogram_id], xyz, radius)
        fbp_crop = None
        if fbp_valid:
            cursor = fbp_cursors[protein][tomogram_id]
            particles = fbp_by_tomogram[protein].get(tomogram_id, [])
            if cursor >= len(particles):
                raise ValueError(
                    f"FBP particle list is missing an entry for {protein}, "
                    f"tomogram {tomogram_id}, metadata row {row_position}."
                )
            fbp_crop = particles[cursor]
            fbp_cursors[protein][tomogram_id] += 1

        if not fbp_valid:
            continue
        if fbp_crop.shape != (2 * radius + 1,) * 3:
            raise ValueError(
                f"Unexpected FBP crop shape for {protein}, tomogram {tomogram_id}: "
                f"{fbp_crop.shape}."
            )

        if any(
            tomogram_id not in volumes[method]
            or not _within_bounds(volumes[method][tomogram_id].shape, xyz, radius)
            for method in required_methods
        ):
            continue

        rotation = rotation_parser(row["rotation"])
        method_crops = {
            "Ground truth": _crop(volumes["Ground truth"][tomogram_id], xyz, radius),
            "FBP": fbp_crop,
            "DeepDeWedge": _crop(volumes["DeepDeWedge"][tomogram_id], xyz, radius),
            "ICecream": _crop(volumes["ICecream"][tomogram_id], xyz, radius),
        }
        mask = _soft_spherical_mask((2 * radius + 1,) * 3, radius)

        for method, crop in method_crops.items():
            aligned = unrotate_fn(
                crop,
                rotation,
                center_offset_zyx=center_offsets[protein],
            )
            aligned_subvolumes[protein][method].append(
                {
                    "row_position": row_position,
                    "subtomo_id": tomogram_id,
                    "volume": np.asarray(aligned, dtype=np.float32) * mask,
                }
            )

    for protein in proteins:
        for tomogram_id, particles in fbp_by_tomogram[protein].items():
            used = fbp_cursors[protein][tomogram_id]
            if used != len(particles):
                raise ValueError(
                    f"FBP particle list has {len(particles) - used} unmatched entries "
                    f"for {protein}, tomogram {tomogram_id}."
                )
        counts = {
            len(aligned_subvolumes[protein][method]) for method in method_names
        }
        if len(counts) != 1 or not counts or next(iter(counts)) == 0:
            raise ValueError(
                f"Expected matching, non-empty particle sets for {protein}; "
                f"got {dict((m, len(aligned_subvolumes[protein][m])) for m in method_names)}."
            )

    average_maps: dict[str, dict[str, np.ndarray]] = {}
    fsc_curves: dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray]] = {}
    summary_rows = []
    for protein in proteins:
        average_maps[protein] = {}
        ground_truth_particles = aligned_subvolumes[protein]["Ground truth"]
        ground_truth_average = np.mean(
            [particle["volume"] for particle in ground_truth_particles], axis=0
        )
        for method in method_names:
            particles = aligned_subvolumes[protein][method]
            average_map = np.mean([particle["volume"] for particle in particles], axis=0)
            average_maps[protein][method] = average_map

            truth_resolution = None
            if method != "Ground truth":
                frequencies, fsc, truth_resolution = _fsc_curve(
                    average_map, ground_truth_average, pixel_spacing
                )
                fsc_curves[(protein, "Ground truth agreement", method)] = (frequencies, fsc)

            even_particles = [
                particle["volume"] for particle in particles if particle["subtomo_id"] % 2 == 0
            ]
            odd_particles = [
                particle["volume"] for particle in particles if particle["subtomo_id"] % 2 == 1
            ]
            half_resolution = None
            if even_particles and odd_particles:
                even_average = np.mean(even_particles, axis=0)
                odd_average = np.mean(odd_particles, axis=0)
                frequencies, fsc, half_resolution = _fsc_curve(
                    even_average, odd_average, pixel_spacing
                )
                fsc_curves[(protein, "Even/odd tomogram-ID half-sets", method)] = (
                    frequencies,
                    fsc,
                )

            summary_rows.append(
                {
                    "protein": protein,
                    "method": method,
                    "particle_count": len(particles),
                    "truth_FSC_resolution_A": truth_resolution,
                    "even_odd_half_set_FSC_resolution_A": half_resolution,
                }
            )

    resolution_summary = pd.DataFrame(summary_rows)
    _plot_fsc_curves(fsc_curves, pixel_spacing)
    return aligned_subvolumes, average_maps, resolution_summary, fsc_curves


def _plot_fsc_curves(
    fsc_curves: Mapping[tuple[str, str, str], tuple[np.ndarray, np.ndarray]],
    pixel_spacing: float,
) -> None:
    nyquist = 1 / (2 * pixel_spacing)
    for protein in ("ribosome", "cohesin"):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
        for ax, metric in zip(
            axes,
            ("Ground truth agreement", "Even/odd tomogram-ID half-sets"),
        ):
            for method in ("FBP", "DeepDeWedge", "ICecream"):
                curve = fsc_curves.get((protein, metric, method))
                if curve is not None:
                    frequencies, fsc = curve
                    ax.plot(frequencies, fsc, label=method)
            if metric == "Even/odd tomogram-ID half-sets":
                curve = fsc_curves.get((protein, metric, "Ground truth"))
                if curve is not None:
                    frequencies, fsc = curve
                    ax.plot(frequencies, fsc, label="Ground truth")
            ax.axhline(0.143, color="black", linestyle="--", linewidth=1)
            ax.axvline(nyquist, color="gray", linestyle=":", linewidth=1)
            ax.set_xlim(0, nyquist)
            ax.set_ylim(-0.2, 1)
            ax.set_xlabel("Spatial frequency (1/A)")
            ax.set_title(metric)
            ax.grid(alpha=0.2)
        axes[0].set_ylabel("Fourier shell correlation")
        axes[1].legend()
        fig.suptitle(f"{protein.capitalize()} particle-average FSC")
        fig.tight_layout()
        plt.show()
