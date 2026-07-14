# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import os

import torch

from . import _C

_mojo_import_error = None
try:
    from . import _mojo
except ImportError as error:
    _mojo = None
    _mojo_import_error = error


_point_face_calls = 0
_face_point_calls = 0
_rasterize_calls = 0


def _backend():
    backend = os.getenv("PYTORCH3D_BACKEND", "auto").lower()
    if backend not in {"auto", "cpu", "cuda", "mojo"}:
        raise RuntimeError("PYTORCH3D_BACKEND must be one of: auto, cpu, cuda, mojo")
    return backend


def _use_mojo(backend, eligible, operation):
    if backend == "mojo" and not eligible:
        message = f"{operation} requires the supported Mojo tensor contract"
        if _mojo_import_error is not None:
            message += f": {_mojo_import_error}"
        raise RuntimeError(message) from _mojo_import_error
    return eligible and backend in {"auto", "mojo"}


def _require_reference_device(backend, tensors):
    if backend not in {"cpu", "cuda"}:
        return
    if backend == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("PYTORCH3D_BACKEND=cuda requires available CUDA")
    if any(tensor.device.type != backend for tensor in tensors):
        raise RuntimeError(
            f"PYTORCH3D_BACKEND={backend} requires {backend.upper()} tensors"
        )


def _point_mesh_eligible(points, points_first_idx, tris, tris_first_idx):
    return (
        _mojo is not None
        and points.device.type == "cpu"
        and tris.device.type == "cpu"
        and points_first_idx.device.type == "cpu"
        and tris_first_idx.device.type == "cpu"
        and points.dtype == torch.float32
        and tris.dtype == torch.float32
        and points_first_idx.dtype == torch.int64
        and tris_first_idx.dtype == torch.int64
        and points.shape[1:] == (3,)
        and tris.shape[1:] == (3, 3)
        and points_first_idx.ndim == 1
        and tris_first_idx.ndim == 1
        and points_first_idx.shape == tris_first_idx.shape
        and points.is_contiguous()
        and tris.is_contiguous()
        and points_first_idx.is_contiguous()
        and tris_first_idx.is_contiguous()
        and not points.requires_grad
        and not tris.requires_grad
    )


def _point_mesh_forward(name, *args):
    global _point_face_calls, _face_point_calls
    backend = _backend()
    eligible = _point_mesh_eligible(*args[:4])
    if _use_mojo(backend, eligible, name):
        try:
            result = getattr(_mojo, name)(*args)
        except Exception as error:
            raise RuntimeError(str(error)) from error
        if name == "point_face_dist_forward":
            _point_face_calls += 1
        else:
            _face_point_calls += 1
        return result
    _require_reference_device(backend, args[:4])
    return getattr(_C, name)(*args)


def point_face_dist_forward(*args):
    return _point_mesh_forward("point_face_dist_forward", *args)


def face_point_dist_forward(*args):
    return _point_mesh_forward("face_point_dist_forward", *args)


def _rasterize_eligible(
    face_verts,
    mesh_to_face_first_idx,
    num_faces_per_mesh,
    clipped_faces_neighbor_idx,
    image_size,
    blur_radius,
    faces_per_pixel,
    bin_size,
    perspective_correct,
    clip_barycentric_coords,
    cull_backfaces,
):
    tensors = (
        face_verts,
        mesh_to_face_first_idx,
        num_faces_per_mesh,
        clipped_faces_neighbor_idx,
    )
    if _mojo is None or not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
        return False
    if not isinstance(image_size, (list, tuple)) or len(image_size) != 2:
        return False
    height, width = image_size
    return (
        all(tensor.device.type == "cpu" for tensor in tensors)
        and face_verts.dtype == torch.float32
        and all(tensor.dtype == torch.int64 for tensor in tensors[1:])
        and face_verts.ndim == 3
        and face_verts.shape[1:] == (3, 3)
        and all(tensor.ndim == 1 for tensor in tensors[1:])
        and mesh_to_face_first_idx.shape == num_faces_per_mesh.shape
        and clipped_faces_neighbor_idx.shape == (face_verts.shape[0],)
        and all(tensor.is_contiguous() for tensor in tensors)
        and not face_verts.requires_grad
        and isinstance(height, int)
        and isinstance(width, int)
        and height > 0
        and width > 0
        and blur_radius == 0.0
        and faces_per_pixel == 1
        and bin_size == 0
        and perspective_correct is True
        and clip_barycentric_coords is False
        and cull_backfaces is False
    )


def rasterize_meshes_forward(
    face_verts,
    mesh_to_face_first_idx,
    num_faces_per_mesh,
    clipped_faces_neighbor_idx,
    image_size,
    blur_radius,
    faces_per_pixel,
    bin_size,
    max_faces_per_bin,
    perspective_correct,
    clip_barycentric_coords,
    cull_backfaces,
):
    global _rasterize_calls
    backend = _backend()
    eligible = _rasterize_eligible(
        face_verts,
        mesh_to_face_first_idx,
        num_faces_per_mesh,
        clipped_faces_neighbor_idx,
        image_size,
        blur_radius,
        faces_per_pixel,
        bin_size,
        perspective_correct,
        clip_barycentric_coords,
        cull_backfaces,
    )
    if _use_mojo(backend, eligible, "rasterize_meshes_forward"):
        try:
            result = _mojo.rasterize_meshes_forward(
                face_verts,
                mesh_to_face_first_idx,
                num_faces_per_mesh,
                clipped_faces_neighbor_idx,
                image_size[0],
                image_size[1],
            )
        except Exception as error:
            raise RuntimeError(str(error)) from error
        _rasterize_calls += 1
        return result
    _require_reference_device(
        backend,
        (
            face_verts,
            mesh_to_face_first_idx,
            num_faces_per_mesh,
            clipped_faces_neighbor_idx,
        ),
    )
    return _C.rasterize_meshes(
        face_verts,
        mesh_to_face_first_idx,
        num_faces_per_mesh,
        clipped_faces_neighbor_idx,
        image_size,
        blur_radius,
        faces_per_pixel,
        bin_size,
        max_faces_per_bin,
        perspective_correct,
        clip_barycentric_coords,
        cull_backfaces,
    )


def has_mojo():
    return _mojo is not None


def point_face_calls():
    return _point_face_calls


def face_point_calls():
    return _face_point_calls


def rasterize_calls():
    return _rasterize_calls


def reset_stats():
    global _point_face_calls, _face_point_calls, _rasterize_calls
    _point_face_calls = 0
    _face_point_calls = 0
    _rasterize_calls = 0
