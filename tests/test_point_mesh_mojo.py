# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import os
import unittest
from unittest import mock

import torch
from pytorch3d import _C, _mojo_ops
from pytorch3d.loss.point_mesh_distance import face_point_distance, point_face_distance


def _starts(counts, noncontiguous=False):
    starts = torch.tensor(
        [sum(counts[:idx]) for idx in range(len(counts))], dtype=torch.int64
    )
    if not noncontiguous:
        return starts
    storage = torch.empty(2 * len(counts), dtype=torch.int64)
    storage[::2] = starts
    return storage[::2]


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


@unittest.skipUnless(_mojo_ops.has_mojo(), "PyTorch3D was built without Mojo")
class TestPointMeshMojo(unittest.TestCase):
    min_triangle_area = 5e-3

    def setUp(self):
        torch.manual_seed(42)
        _mojo_ops.reset_stats()

    def _call(self, name, args, *, mojo, required=None):
        if required is None:
            required = mojo
        with _backend(mojo=mojo, required=required):
            if mojo:
                result = getattr(_mojo_ops, name)(*args)
                self.assertIsNotNone(result)
                return result
            return getattr(_C, name)(*args)

    def _forward(self, points, point_counts, tris, tri_counts, *, mojo):
        points_first = _starts(point_counts)
        tris_first = _starts(tri_counts)
        common = (points, points_first, tris, tris_first)
        return (
            self._call(
                "point_face_dist_forward",
                (*common, max(point_counts, default=0), self.min_triangle_area),
                mojo=mojo,
            ),
            self._call(
                "face_point_dist_forward",
                (*common, max(tri_counts, default=0), self.min_triangle_area),
                mojo=mojo,
            ),
        )

    def assert_forward_equal(self, points, point_counts, tris, tri_counts):
        cpu = self._forward(points, point_counts, tris, tri_counts, mojo=False)
        mojo = self._forward(points, point_counts, tris, tri_counts, mojo=True)
        for (cpu_dists, cpu_idxs), (mojo_dists, mojo_idxs) in zip(cpu, mojo):
            torch.testing.assert_close(mojo_dists, cpu_dists, rtol=2e-5, atol=2e-6)
            self.assertTrue(torch.equal(mojo_idxs, cpu_idxs))

    def test_ragged_forward_and_ties(self):
        points = torch.randn(7, 3)
        tris = torch.randn(7, 3, 3)
        self.assert_forward_equal(points, [4, 3], tris, [5, 2])

        tied_point = torch.tensor([[0.2, 0.2, 1.0]])
        tied_tris = torch.tensor(
            [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]
        ).repeat(2, 1, 1)
        self.assert_forward_equal(tied_point, [1], tied_tris, [2])
        _, idxs = self._call(
            "point_face_dist_forward",
            (
                tied_point,
                _starts([1]),
                tied_tris,
                _starts([2]),
                1,
                self.min_triangle_area,
            ),
            mojo=True,
        )
        self.assertEqual(idxs.item(), 1)

        tied_points = tied_point.repeat(2, 1)
        _, idxs = self._call(
            "face_point_dist_forward",
            (
                tied_points,
                _starts([2]),
                tied_tris[:1],
                _starts([1]),
                1,
                self.min_triangle_area,
            ),
            mojo=True,
        )
        self.assertEqual(idxs.item(), 1)

    def test_randomized_holdout(self):
        for seed in range(20):
            torch.manual_seed(seed)
            batch_size = 1 + seed % 4
            point_counts = [(seed * 3 + idx * 5) % 9 for idx in range(batch_size)]
            tri_counts = [(seed * 7 + idx * 2) % 8 for idx in range(batch_size)]
            if not any(point_counts):
                point_counts[-1] = 1
            if not any(tri_counts):
                tri_counts[-1] = 1
            self.assert_forward_equal(
                torch.randn(sum(point_counts), 3),
                point_counts,
                torch.randn(sum(tri_counts), 3, 3),
                tri_counts,
            )

    def test_degenerate_and_threshold_geometry(self):
        edge_points = torch.tensor(
            [[0.5, 0.0, 0.0], [0.0, 0.0, 0.0], [0.5, -1e-4, 0.0]]
        )
        unit_tri = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
        self.assert_forward_equal(edge_points, [3], unit_tri, [1])

        points = torch.tensor([[0.25, 0.25, 1.0], [2.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        tris = torch.tensor(
            [
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0]],
            ]
        )
        self.assert_forward_equal(points, [3], tris, [3])

        threshold_tri = tris[2:3]
        threshold_point = torch.tensor([[0.025, 0.025, 1.0]])
        area = (
            torch.linalg.cross(
                threshold_tri[0, 1] - threshold_tri[0, 0],
                threshold_tri[0, 2] - threshold_tri[0, 0],
            ).norm()
            / 2
        ).item()
        starts = torch.tensor([0])
        outputs = []
        for mojo in (False, True):
            outputs.append(
                self._call(
                    "point_face_dist_forward",
                    (
                        threshold_point,
                        starts,
                        threshold_tri,
                        starts,
                        1,
                        area + 1e-12,
                    ),
                    mojo=mojo,
                )
            )
        torch.testing.assert_close(outputs[1][0], outputs[0][0], rtol=0, atol=0)
        self.assertTrue(torch.equal(outputs[1][1], outputs[0][1]))

        rotated_point = torch.tensor([[0.02262224257, 0.18231746554, -0.81720376015]])
        rotated_tri = torch.tensor(
            [
                [
                    [0.17558883131, 0.28667378426, -0.13929519057],
                    [0.14069361985, 0.18612900376, -0.12446017563],
                    [0.07127179205, 0.26809445024, -0.12141314149],
                ]
            ]
        )
        for name in ("point_face_dist_forward", "face_point_dist_forward"):
            outputs = []
            for mojo in (False, True):
                outputs.append(
                    self._call(
                        name,
                        (
                            rotated_point,
                            starts,
                            rotated_tri,
                            starts,
                            1,
                            self.min_triangle_area,
                        ),
                        mojo=mojo,
                    )
                )
            torch.testing.assert_close(
                outputs[1][0], outputs[0][0], rtol=2e-5, atol=2e-6
            )
            self.assertTrue(torch.equal(outputs[1][1], outputs[0][1]))

    def test_empty_packed_batches(self):
        points = torch.tensor([[9.0, 0.0, 0.0], [20.0, 0.0, 0.0]])
        tris = torch.tensor(
            [
                [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                [[10.0, 0.0, 0.0], [10.0, 1.0, 0.0], [10.0, 0.0, 1.0]],
                [[20.0, 0.0, 0.0], [20.0, 1.0, 0.0], [20.0, 0.0, 1.0]],
            ]
        )
        self.assert_forward_equal(points, [0, 1, 1], tris, [1, 1, 1])
        self.assert_forward_equal(points, [1, 1, 0], tris[:2], [0, 1, 1])
        self.assert_forward_equal(torch.empty(0, 3), [0], torch.empty(0, 3, 3), [0])

    def test_public_autograd_falls_back_to_cpu(self):
        points_cpu = torch.randn(5, 3, requires_grad=True)
        tris_cpu = torch.randn(4, 3, 3, requires_grad=True)
        points_first = _starts([3, 2])
        tris_first = _starts([2, 2])
        with _backend(mojo=False):
            loss_cpu = (
                point_face_distance(
                    points_cpu, points_first, tris_cpu, tris_first, 3
                ).sum()
                + face_point_distance(
                    points_cpu, points_first, tris_cpu, tris_first, 2
                ).sum()
            )
        loss_cpu.backward()

        points_mojo = points_cpu.detach().clone().requires_grad_()
        tris_mojo = tris_cpu.detach().clone().requires_grad_()
        with _backend(mojo=True):
            loss_mojo = (
                point_face_distance(
                    points_mojo, points_first, tris_mojo, tris_first, 3
                ).sum()
                + face_point_distance(
                    points_mojo, points_first, tris_mojo, tris_first, 2
                ).sum()
            )
        loss_mojo.backward()

        torch.testing.assert_close(loss_mojo, loss_cpu, rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(
            points_mojo.grad, points_cpu.grad, rtol=3e-5, atol=3e-6
        )
        torch.testing.assert_close(tris_mojo.grad, tris_cpu.grad, rtol=3e-5, atol=3e-6)
        self.assertEqual(_mojo_ops.point_face_calls(), 0)
        self.assertEqual(_mojo_ops.face_point_calls(), 0)

    def test_public_forward_uses_mojo(self):
        points = torch.randn(5, 3)
        tris = torch.randn(4, 3, 3)
        points_first = _starts([3, 2])
        tris_first = _starts([2, 2])
        with _backend(mojo=False):
            cpu = (
                point_face_distance(points, points_first, tris, tris_first, 3),
                face_point_distance(points, points_first, tris, tris_first, 2),
            )
        with _backend(mojo=True, required=True):
            mojo = (
                point_face_distance(points, points_first, tris, tris_first, 3),
                face_point_distance(points, points_first, tris, tris_first, 2),
            )
        for actual, expected in zip(mojo, cpu):
            torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
        self.assertEqual(_mojo_ops.point_face_calls(), 1)
        self.assertEqual(_mojo_ops.face_point_calls(), 1)

    def test_unsupported_inputs_fall_back_or_raise(self):
        points = torch.randn(3, 6)[:, ::2]
        tris = torch.randn(2, 3, 6)[:, :, ::2]
        points_first = _starts([2, 1], noncontiguous=True)
        tris_first = _starts([1, 1], noncontiguous=True)
        self.assertFalse(points.is_contiguous())
        self.assertFalse(tris.is_contiguous())
        self.assertFalse(points_first.is_contiguous())

        expected = (
            _C.point_face_dist_forward(
                points,
                points_first,
                tris,
                tris_first,
                2,
                self.min_triangle_area,
            ),
            _C.face_point_dist_forward(
                points,
                points_first,
                tris,
                tris_first,
                1,
                self.min_triangle_area,
            ),
        )
        with _backend(mojo=True):
            actual = (
                _mojo_ops.point_face_dist_forward(
                    points,
                    points_first,
                    tris,
                    tris_first,
                    2,
                    self.min_triangle_area,
                ),
                _mojo_ops.face_point_dist_forward(
                    points,
                    points_first,
                    tris,
                    tris_first,
                    1,
                    self.min_triangle_area,
                ),
            )
        for (actual_dists, actual_idxs), (expected_dists, expected_idxs) in zip(
            actual, expected
        ):
            torch.testing.assert_close(actual_dists, expected_dists)
            self.assertTrue(torch.equal(actual_idxs, expected_idxs))
        self.assertEqual(_mojo_ops.point_face_calls(), 0)
        self.assertEqual(_mojo_ops.face_point_calls(), 0)

        unsupported = (
            (points.contiguous().requires_grad_(), tris.contiguous()),
            (points.contiguous(), tris.contiguous().requires_grad_()),
            (points.contiguous().double(), tris.contiguous()),
            (points.contiguous()[:, :2], tris.contiguous()),
        )
        for unsupported_points, unsupported_tris in unsupported:
            with self.subTest(
                points_shape=unsupported_points.shape,
                points_dtype=unsupported_points.dtype,
                points_grad=unsupported_points.requires_grad,
                tris_grad=unsupported_tris.requires_grad,
            ):
                with _backend(mojo=True, required=True):
                    with self.assertRaisesRegex(RuntimeError, "required Mojo"):
                        _mojo_ops.point_face_dist_forward(
                            unsupported_points,
                            _starts([2, 1]),
                            unsupported_tris,
                            _starts([1, 1]),
                            2,
                            self.min_triangle_area,
                        )

    def test_invalid_packed_offsets_and_max_outer_raise(self):
        points = torch.randn(3, 3)
        tris = torch.randn(3, 3, 3)
        valid = _starts([2, 1])
        invalid = (
            torch.tensor([1, 2]),
            torch.tensor([0, -1]),
            torch.tensor([0, 4]),
        )
        with _backend(mojo=True, required=True):
            for starts in invalid:
                with self.subTest(starts=starts.tolist()):
                    with self.assertRaisesRegex(RuntimeError, "points_first_idx"):
                        _mojo_ops.point_face_dist_forward(
                            points,
                            starts,
                            tris,
                            valid,
                            2,
                            self.min_triangle_area,
                        )
            with self.assertRaisesRegex(RuntimeError, "tris_first_idx"):
                _mojo_ops.face_point_dist_forward(
                    points,
                    valid,
                    tris,
                    torch.tensor([0, 4]),
                    2,
                    self.min_triangle_area,
                )
            with self.assertRaisesRegex(RuntimeError, "max_outer"):
                _mojo_ops.point_face_dist_forward(
                    points, valid, tris, valid, 1, self.min_triangle_area
                )
            with self.assertRaisesRegex(RuntimeError, "max_outer"):
                _mojo_ops.face_point_dist_forward(
                    points, valid, tris, valid, 1, self.min_triangle_area
                )

    def test_direct_module_rejects_unsafe_tensor_contracts(self):
        points = torch.randn(1, 3)
        tris = torch.randn(1, 3, 3)
        starts = torch.tensor([0])
        cases = (
            (points.double(), starts, tris, starts),
            (points[:, :2], starts, tris, starts),
            (points, torch.tensor([0, 0]), tris, starts),
            (points, starts.int(), tris, starts),
        )
        for args in cases:
            with self.subTest(args=[(tensor.shape, tensor.dtype) for tensor in args]):
                with self.assertRaises(Exception):
                    _mojo_ops._mojo.point_face_dist_forward(
                        *args, 1, self.min_triangle_area
                    )

    def test_provenance_and_required_fallback(self):
        points = torch.tensor([[0.2, 0.2, 1.0]])
        tris = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
        starts = torch.tensor([0])
        with _backend(mojo=True, required=True):
            _mojo_ops.point_face_dist_forward(
                points, starts, tris, starts, 1, self.min_triangle_area
            )
            _mojo_ops.face_point_dist_forward(
                points, starts, tris, starts, 1, self.min_triangle_area
            )
        self.assertEqual(_mojo_ops.point_face_calls(), 1)
        self.assertEqual(_mojo_ops.face_point_calls(), 1)

        with _backend(mojo=False, required=True):
            with self.assertRaisesRegex(RuntimeError, "required Mojo"):
                _mojo_ops.point_face_dist_forward(
                    points, starts, tris, starts, 1, self.min_triangle_area
                )


class TestPointMeshMojoRequirement(unittest.TestCase):
    def test_required_mode_reports_import_failure_without_skipping(self):
        points = torch.randn(1, 3)
        tris = torch.randn(1, 3, 3)
        starts = torch.tensor([0])
        error = ImportError("broken Mojo runtime")
        with mock.patch.object(_mojo_ops, "_mojo", None), mock.patch.object(
            _mojo_ops, "_mojo_import_error", error
        ), _backend(mojo=True, required=True):
            with self.assertRaisesRegex(RuntimeError, "broken Mojo runtime"):
                _mojo_ops.point_face_dist_forward(points, starts, tris, starts, 1, 5e-3)


if __name__ == "__main__":
    unittest.main()
