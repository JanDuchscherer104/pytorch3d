# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Benchmark deterministic point/mesh kernels across PyTorch3D backends."""

import argparse
import hashlib
import io
import json
import os
import platform
import statistics
import subprocess
import sys
import time
import zipfile
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from pytorch3d import _mojo_ops


def _fixture_arrays():
    point_counts = np.array([128, 160, 96, 128], dtype=np.int64)
    face_counts = np.array([64, 80, 48, 64], dtype=np.int64)
    p = np.arange(point_counts.sum(), dtype=np.float32)
    f = np.arange(face_counts.sum(), dtype=np.float32)
    points = np.stack((np.sin(p * 0.17), np.cos(p * 0.11), np.sin(p * 0.07)), axis=1)
    centers = np.stack((np.sin(f * 0.13), np.cos(f * 0.19), np.sin(f * 0.05)), axis=1)
    offsets = np.array(
        [[-0.09, -0.06, 0.02], [0.10, -0.04, -0.01], [-0.02, 0.11, 0.03]],
        dtype=np.float32,
    )
    raster_faces = []
    for index in range(48):
        x = -0.875 + (index % 8) * 0.25
        y = -0.75 + (index // 8) * 0.30
        z = 0.2 + index * 0.02
        raster_faces.append(
            ((x - 0.09, y - 0.08, z), (x + 0.09, y - 0.08, z), (x, y + 0.10, z))
        )
    return {
        "points": points.astype(np.float32),
        "points_first_idx": np.r_[0, np.cumsum(point_counts)[:-1]],
        "tris": (centers[:, None, :] + offsets[None, :, :]).astype(np.float32),
        "tris_first_idx": np.r_[0, np.cumsum(face_counts)[:-1]],
        "max_points": np.array(point_counts.max(), dtype=np.int64),
        "max_tris": np.array(face_counts.max(), dtype=np.int64),
        "min_triangle_area": np.array(5e-3, dtype=np.float64),
        "face_verts": np.asarray(raster_faces, dtype=np.float32),
        "mesh_to_face_first_idx": np.array([0, 24], dtype=np.int64),
        "num_faces_per_mesh": np.array([24, 24], dtype=np.int64),
        "clipped_faces_neighbor_idx": np.full(48, -1, dtype=np.int64),
        "image_size": np.array([48, 64], dtype=np.int64),
    }


def _write_npz(path, arrays):
    """Write byte-for-byte reproducible, uncompressed NPZ files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        for name in sorted(arrays):
            data = io.BytesIO()
            np.lib.format.write_array(
                data, np.asarray(arrays[name]), allow_pickle=False
            )
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.external_attr = 0o644 << 16
            archive.writestr(info, data.getvalue())


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _percentile(values, percent):
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _timings(call, warmups, trials, synchronize):
    for _ in range(warmups):
        call()
    synchronize()
    raw = []
    for _ in range(trials):
        synchronize()
        start = time.perf_counter_ns()
        call()
        synchronize()
        raw.append((time.perf_counter_ns() - start) / 1e6)
    q25, q75 = _percentile(raw, 25), _percentile(raw, 75)
    return {
        "raw_ms": raw,
        "median_ms": statistics.median(raw),
        "p5_ms": _percentile(raw, 5),
        "p95_ms": _percentile(raw, 95),
        "iqr_ms": q75 - q25,
    }


def _command_output(*command):
    try:
        return subprocess.run(
            command, check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _package_version(package):
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def _metadata(backend):
    commit = _command_output("git", "rev-parse", "HEAD")
    dirty = _command_output("git", "status", "--porcelain")
    return {
        "backend": backend,
        "machine": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_implementation": platform.python_implementation(),
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "mojo": _package_version("mojo") or _command_output("mojo", "--version"),
        },
        "git": {"commit": commit, "dirty": None if dirty is None else bool(dirty)},
    }


def _load_fixture(path, device):
    required = {
        "points",
        "points_first_idx",
        "tris",
        "tris_first_idx",
        "max_points",
        "max_tris",
        "min_triangle_area",
        "face_verts",
        "mesh_to_face_first_idx",
        "num_faces_per_mesh",
        "clipped_faces_neighbor_idx",
        "image_size",
    }
    with np.load(path, allow_pickle=False) as source:
        missing = required - set(source.files)
        if missing:
            raise ValueError(f"fixture is missing arrays: {', '.join(sorted(missing))}")
        values = {name: source[name] for name in required}
    tensors = {
        name: torch.from_numpy(value).to(device=device).contiguous()
        for name, value in values.items()
        if value.ndim > 0 and name != "image_size"
    }
    tensors.update(
        max_points=int(values["max_points"]),
        max_tris=int(values["max_tris"]),
        min_triangle_area=float(values["min_triangle_area"]),
        image_size=tuple(int(x) for x in values["image_size"]),
    )
    return tensors


def _to_numpy(outputs):
    return {
        name: tensor.detach().cpu().numpy()
        for operation, tensors in outputs.items()
        for name, tensor in zip(operation, tensors)
    }


def run(args):
    if args.backend == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("backend cuda requested, but CUDA is unavailable")
    if args.backend == "mojo" and not _mojo_ops.has_mojo():
        raise RuntimeError("backend mojo requested, but pytorch3d._mojo is unavailable")
    if args.generate_fixture:
        _write_npz(args.fixture, _fixture_arrays())
    if not args.fixture.is_file():
        raise FileNotFoundError(f"fixture does not exist: {args.fixture}")

    os.environ["PYTORCH3D_BACKEND"] = args.backend
    device = torch.device("cuda" if args.backend == "cuda" else "cpu")
    fixture = _load_fixture(args.fixture, device)
    synchronize = torch.cuda.synchronize if device.type == "cuda" else lambda: None
    calls = {
        "point_face_dist_forward": (
            ("point_face_dists", "point_face_idxs"),
            lambda: _mojo_ops.point_face_dist_forward(
                fixture["points"],
                fixture["points_first_idx"],
                fixture["tris"],
                fixture["tris_first_idx"],
                fixture["max_points"],
                fixture["min_triangle_area"],
            ),
        ),
        "face_point_dist_forward": (
            ("face_point_dists", "face_point_idxs"),
            lambda: _mojo_ops.face_point_dist_forward(
                fixture["points"],
                fixture["points_first_idx"],
                fixture["tris"],
                fixture["tris_first_idx"],
                fixture["max_tris"],
                fixture["min_triangle_area"],
            ),
        ),
        "rasterize_meshes_forward": (
            ("pix_to_face", "zbuf", "barycentric_coords", "pixel_dists"),
            lambda: _mojo_ops.rasterize_meshes_forward(
                fixture["face_verts"],
                fixture["mesh_to_face_first_idx"],
                fixture["num_faces_per_mesh"],
                fixture["clipped_faces_neighbor_idx"],
                fixture["image_size"],
                0.0,
                1,
                0,
                10_000,
                True,
                False,
                False,
            ),
        ),
    }
    _mojo_ops.reset_stats()
    with torch.inference_mode():
        outputs = {names: call() for names, call in calls.values()}
        timings = {
            operation: _timings(call, args.warmups, args.trials, synchronize)
            for operation, (_, call) in calls.items()
        }
    if args.backend == "mojo" and (
        _mojo_ops.point_face_calls() == 0
        or _mojo_ops.face_point_calls() == 0
        or _mojo_ops.rasterize_calls() == 0
    ):
        raise RuntimeError(
            "Mojo was requested but at least one operation did not use it"
        )

    arrays_path = args.arrays or args.output.with_suffix(".outputs.npz")
    _write_npz(arrays_path, _to_numpy(outputs))
    result = _metadata(args.backend)
    result.update(
        fixture={"path": str(args.fixture.resolve()), "sha256": _sha256(args.fixture)},
        outputs={"path": str(arrays_path.resolve()), "sha256": _sha256(arrays_path)},
        benchmark={"warmups": args.warmups, "trials": args.trials, "timings": timings},
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cpu", "cuda", "mojo"), required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--generate-fixture", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arrays", type=Path)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--trials", type=int, default=20)
    args = parser.parse_args()
    if args.warmups < 0 or args.trials < 1:
        parser.error("--warmups must be non-negative and --trials must be positive")
    try:
        result = run(args)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
