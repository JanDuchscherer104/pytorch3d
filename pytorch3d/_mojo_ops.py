# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import os

import torch

_mojo_import_error = None
try:
    from . import _mojo
except ImportError as error:
    _mojo = None
    _mojo_import_error = error


_point_face_calls = 0
_face_point_calls = 0
_rasterize_calls = 0


def _env_flag(name):
    return os.getenv(name) == "1"


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
    eligible = _point_mesh_eligible(*args[:4])
    if eligible and not _env_flag("PYTORCH3D_DISABLE_MOJO"):
        try:
            result = getattr(_mojo, name)(*args)
        except Exception as error:
            raise RuntimeError(str(error)) from error
        if name == "point_face_dist_forward":
            _point_face_calls += 1
        else:
            _face_point_calls += 1
        return result
    if _env_flag("PYTORCH3D_REQUIRE_MOJO"):
        message = f"{name} required Mojo but fell back to CPU"
        if _mojo_import_error is not None:
            message += f": {_mojo_import_error}"
        raise RuntimeError(message) from _mojo_import_error
    return None


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
    del max_faces_per_bin
    global _rasterize_calls
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
    if eligible and not _env_flag("PYTORCH3D_DISABLE_MOJO"):
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
    if _env_flag("PYTORCH3D_REQUIRE_MOJO"):
        message = "rasterize_meshes_forward required Mojo but fell back to CPU"
        if _mojo_import_error is not None:
            message += f": {_mojo_import_error}"
        raise RuntimeError(message) from _mojo_import_error
    return None


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
