# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import json
import os
import unittest
from unittest import mock

from pytorch3d import _mojo_ops


class TestMojoBackendStatus(unittest.TestCase):
    def setUp(self):
        _mojo_ops.reset_stats()

    def test_status_is_json_serializable_and_reports_contract(self):
        with mock.patch.dict(os.environ, {"PYTORCH3D_BACKEND": "auto"}):
            status = _mojo_ops.backend_status()

        json.dumps(status)
        self.assertEqual(status["requested_backend"], "auto")
        self.assertEqual(status["dispatch_policy"], "eligible_cpu_contract_else_native")
        self.assertEqual(
            status["mojo_operations"],
            [
                "face_point_dist_forward",
                "point_face_dist_forward",
                "rasterize_meshes_forward",
            ],
        )
        self.assertEqual(
            status["counters"],
            {
                "point_face_calls": 0,
                "face_point_calls": 0,
                "rasterize_calls": 0,
            },
        )

    def test_status_exposes_import_failure_for_preflight(self):
        error = ImportError("missing Mojo runtime")
        with (
            mock.patch.object(_mojo_ops, "_mojo", None),
            mock.patch.object(_mojo_ops, "_mojo_import_error", error),
            mock.patch.dict(os.environ, {"PYTORCH3D_BACKEND": "mojo"}),
        ):
            status = _mojo_ops.backend_status()

        self.assertFalse(status["mojo_available"])
        self.assertEqual(status["mojo_import_error"], "missing Mojo runtime")

    def test_status_validates_requested_backend(self):
        with mock.patch.dict(os.environ, {"PYTORCH3D_BACKEND": "metal"}):
            with self.assertRaisesRegex(RuntimeError, "auto, cpu, cuda, mojo"):
                _mojo_ops.backend_status()


if __name__ == "__main__":
    unittest.main()
