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


def has_mojo():
    return _mojo is not None


def point_face_calls():
    return _point_face_calls


def face_point_calls():
    return _face_point_calls


def reset_stats():
    global _point_face_calls, _face_point_calls
    _point_face_calls = 0
    _face_point_calls = 0
