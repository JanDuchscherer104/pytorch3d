# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import os
import unittest

import torch
from pytorch3d import _C, _mojo_ops
from pytorch3d.renderer.mesh.rasterize_meshes import rasterize_meshes
from pytorch3d.structures import Meshes


@contextlib.contextmanager
def _backend(*, mojo, required=False):
    names = ("PYTORCH3D_DISABLE_MOJO", "PYTORCH3D_REQUIRE_MOJO")
    previous = {name: os.environ.get(name) for name in names}
    try:
        if mojo:
            os.environ.pop("PYTORCH3D_DISABLE_MOJO", None)
        else:
            os.environ["PYTORCH3D_DISABLE_MOJO"] = "1"
        if required:
            os.environ["PYTORCH3D_REQUIRE_MOJO"] = "1"
        else:
            os.environ.pop("PYTORCH3D_REQUIRE_MOJO", None)
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _triangle(z, x_scale=0.9):
    return torch.tensor(
        [[-x_scale, -0.9, z], [0.0, 0.9, z], [x_scale, -0.9, z]],
        dtype=torch.float32,
    )


def _starts(counts):
    return torch.tensor(
        [sum(counts[:index]) for index in range(len(counts))], dtype=torch.int64
    )


def _args(face_verts, counts, image_size, neighbors=None):
    counts_tensor = torch.tensor(counts, dtype=torch.int64)
    if neighbors is None:
        neighbors = torch.full((face_verts.shape[0],), -1, dtype=torch.int64)
    return (
        face_verts,
        _starts(counts),
        counts_tensor,
        neighbors,
        image_size,
        0.0,
        1,
        0,
        10_000,
        True,
        False,
        False,
    )


@unittest.skipUnless(_mojo_ops.has_mojo(), "PyTorch3D was built without Mojo")
class TestRasterizeMeshesMojo(unittest.TestCase):
    def setUp(self):
        _mojo_ops.reset_stats()

    def _call(self, args, *, mojo, required=None):
        if required is None:
            required = mojo
        with _backend(mojo=mojo, required=required):
            if mojo:
                result = _mojo_ops.rasterize_meshes_forward(*args)
                self.assertIsNotNone(result)
                return result
            return _C.rasterize_meshes(*args)

    def assert_parity(self, args):
        expected = self._call(args, mojo=False)
        actual = self._call(args, mojo=True)
        self.assertTrue(torch.equal(actual[0], expected[0]))
        for result, reference in zip(actual[1:], expected[1:]):
            torch.testing.assert_close(result, reference, rtol=2e-5, atol=2e-6)
        return actual

    def test_low_level_geometry_and_batch_parity(self):
        cases = {
            "square": _args(_triangle(1.0).unsqueeze(0), [1], (5, 5)),
            "rectangular_empty_batch": _args(
                torch.stack((_triangle(1.0), _triangle(2.0))),
                [1, 0, 1],
                (4, 7),
            ),
            "one_empty_mesh": _args(torch.empty((0, 3, 3)), [0], (3, 5)),
            "empty_batch": _args(torch.empty((0, 3, 3)), [], (3, 5)),
            "degenerate": _args(
                torch.stack((torch.zeros((3, 3)), _triangle(1.0))),
                [2],
                (5, 5),
            ),
            "depth_tie": _args(
                torch.stack((_triangle(1.0), _triangle(1.0))),
                [2],
                (5, 5),
            ),
            "nearer_face": _args(
                torch.stack((_triangle(2.0), _triangle(1.0))),
                [2],
                (5, 5),
            ),
        }
        for name, args in cases.items():
            with self.subTest(name=name):
                self.assert_parity(args)

        tied = self.assert_parity(cases["depth_tie"])
        self.assertEqual(tied[0][0, 2, 2, 0].item(), 0)
        nearer = self.assert_parity(cases["nearer_face"])
        self.assertEqual(nearer[0][0, 2, 2, 0].item(), 1)

    def test_clipped_sibling_strict_distance_replacement(self):
        faces = torch.stack((_triangle(1.0, 0.9), _triangle(5.0, 0.1)))
        args = _args(faces, [2], (5, 5), torch.tensor([1, 0]))
        actual = self.assert_parity(args)
        self.assertEqual(actual[0][0, 2, 2, 0].item(), 1)

        equal_args = _args(
            torch.stack((_triangle(1.0), _triangle(5.0))),
            [2],
            (5, 5),
            torch.tensor([1, 0]),
        )
        equal = self.assert_parity(equal_args)
        self.assertEqual(equal[0][0, 2, 2, 0].item(), 0)

    def test_randomized_holdout(self):
        for seed in range(20):
            torch.manual_seed(seed)
            counts = [1 + (seed * 3 + batch * 5) % 7 for batch in range(1 + seed % 4)]
            faces = torch.randn((sum(counts), 3, 3))
            faces[..., 2] = faces[..., 2].abs() + 0.05
            self.assert_parity(_args(faces, counts, (7 + seed % 4, 9 + seed % 5)))

    def test_z_epsilon_matches_cpu_double_boundary(self):
        epsilon = torch.tensor(1e-8, dtype=torch.float32)
        above = torch.nextafter(epsilon, torch.tensor(torch.inf, dtype=torch.float32))
        for z, expected_face in ((epsilon, -1), (above, 0)):
            face = _triangle(1.0).unsqueeze(0)
            face[..., 2] = z
            actual = self.assert_parity(_args(face, [1], (5, 5)))
            self.assertEqual(actual[0][0, 2, 2, 0].item(), expected_face)

    def test_public_near_plane_clipping_and_original_face_mapping(self):
        verts = torch.tensor(
            [[-0.9, -0.8, 0.05], [0.9, -0.8, 1.0], [0.0, 0.9, 1.0]],
            dtype=torch.float32,
        )
        mesh = Meshes(verts=[verts], faces=[torch.tensor([[0, 1, 2]])])
        kwargs = {
            "image_size": (9, 7),
            "blur_radius": 0.0,
            "faces_per_pixel": 1,
            "bin_size": 0,
            "perspective_correct": True,
            "clip_barycentric_coords": False,
            "cull_backfaces": False,
            "z_clip_value": 0.1,
            "cull_to_frustum": False,
        }
        with _backend(mojo=False):
            expected = rasterize_meshes(mesh, **kwargs)
        with _backend(mojo=True, required=True):
            actual = rasterize_meshes(mesh, **kwargs)
        self.assertTrue(torch.equal(actual[0], expected[0]))
        for result, reference in zip(actual[1:], expected[1:]):
            torch.testing.assert_close(result, reference, rtol=2e-5, atol=2e-6)
        hits = actual[0][actual[0] >= 0]
        self.assertGreater(hits.numel(), 0)
        self.assertTrue(torch.equal(hits.unique(), torch.tensor([0])))
        self.assertEqual(_mojo_ops.rasterize_calls(), 1)

    def test_unsupported_inputs_fall_back_and_required_rejects(self):
        base = list(_args(_triangle(1.0).unsqueeze(0), [1], (5, 5)))
        variants = []
        for index, value in (
            (0, base[0].double()),
            (0, base[0].clone().requires_grad_()),
            (4, (0, 5)),
            (5, 0.1),
            (6, 2),
            (7, 8),
            (9, False),
            (10, True),
            (11, True),
        ):
            variant = base.copy()
            variant[index] = value
            variants.append(variant)
        noncontiguous = base.copy()
        noncontiguous[0] = torch.randn((1, 3, 6), dtype=torch.float32)[:, :, ::2]
        variants.append(noncontiguous)

        for args in variants:
            with self.subTest(dtype=args[0].dtype, shape=args[0].shape, flags=args[4:]):
                with _backend(mojo=True):
                    self.assertIsNone(_mojo_ops.rasterize_meshes_forward(*args))
                with _backend(mojo=True, required=True):
                    with self.assertRaisesRegex(RuntimeError, "required Mojo"):
                        _mojo_ops.rasterize_meshes_forward(*args)
        self.assertEqual(_mojo_ops.rasterize_calls(), 0)

    def test_public_autograd_falls_back(self):
        verts = _triangle(1.0).requires_grad_()
        mesh = Meshes(verts=[verts], faces=[torch.tensor([[0, 1, 2]])])
        kwargs = {
            "image_size": 5,
            "blur_radius": 0.0,
            "faces_per_pixel": 1,
            "bin_size": 0,
            "perspective_correct": True,
            "clip_barycentric_coords": False,
            "cull_backfaces": False,
        }
        with _backend(mojo=True):
            _, zbuf, bary, dists = rasterize_meshes(mesh, **kwargs)
        (zbuf.sum() + bary.sum() + dists.sum()).backward()
        self.assertIsNotNone(verts.grad)
        self.assertEqual(_mojo_ops.rasterize_calls(), 0)

        required_verts = _triangle(1.0).requires_grad_()
        required_mesh = Meshes(
            verts=[required_verts], faces=[torch.tensor([[0, 1, 2]])]
        )
        with _backend(mojo=True, required=True):
            with self.assertRaisesRegex(RuntimeError, "required Mojo"):
                rasterize_meshes(required_mesh, **kwargs)

    def test_direct_binding_rejects_unsafe_inputs(self):
        mojo = _mojo_ops._mojo
        face = _triangle(1.0).unsqueeze(0)
        starts = torch.tensor([0])
        counts = torch.tensor([1])
        neighbors = torch.tensor([-1])
        calls = (
            (face.double(), starts, counts, neighbors, 5, 5),
            (face[:, :, :2], starts, counts, neighbors, 5, 5),
            (face, torch.tensor([1]), counts, neighbors, 5, 5),
            (face, starts, torch.tensor([2]), neighbors, 5, 5),
            (face, starts, counts, torch.tensor([1]), 5, 5),
            (face, starts, counts, neighbors, 0, 5),
        )
        for args in calls:
            with self.subTest(args=args[1:]):
                with self.assertRaises(Exception):
                    mojo.rasterize_meshes_forward(*args)

    def test_counter_and_disabled_required_error(self):
        args = _args(_triangle(1.0).unsqueeze(0), [1], (3, 3))
        self._call(args, mojo=True)
        self._call(args, mojo=True)
        self.assertEqual(_mojo_ops.rasterize_calls(), 2)
        with _backend(mojo=False, required=True):
            with self.assertRaisesRegex(RuntimeError, "required Mojo"):
                _mojo_ops.rasterize_meshes_forward(*args)


if __name__ == "__main__":
    unittest.main()
