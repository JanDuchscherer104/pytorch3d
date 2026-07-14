# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

from point_mesh_mojo import point_mesh_distance
from std.algorithm import parallelize
from std.os import abort
from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder


@export
def PyInit__mojo() abi("C") -> PythonObject:
    try:
        var module = PythonModuleBuilder("_mojo")
        module.def_function[point_face_dist_forward]("point_face_dist_forward")
        module.def_function[face_point_dist_forward]("face_point_dist_forward")
        return module.finalize()
    except error:
        abort(String("pytorch3d._mojo initialization failed: ", error))


def _max_packed_count(
    address: UInt64, total: Int, batches: Int, name: String
) raises -> Int:
    if batches == 0:
        if total != 0:
            raise Error(name + " cannot be empty when packed data is nonempty")
        return 0
    var starts = UnsafePointer[Int64, MutUntrackedOrigin](
        unsafe_from_address=Int(address)
    )
    if starts[0] != 0:
        raise Error(name + " must start at zero")
    var max_count = 0
    for batch in range(batches):
        var begin = Int(starts[batch])
        var end = total
        if batch + 1 < batches:
            end = Int(starts[batch + 1])
        if begin < 0 or begin > total or end < begin or end > total:
            raise Error(name + " must be monotonic and within packed tensor bounds")
        max_count = max(max_count, end - begin)
    return max_count


def _require_tensor(
    tensor: PythonObject,
    torch: PythonObject,
    dtype: PythonObject,
    rank: Int,
    name: String,
) raises:
    if not Bool(py=torch.is_tensor(tensor)):
        raise Error(name + " must be a Torch tensor")
    if String(py=tensor.device.type) != "cpu":
        raise Error(name + " must be on CPU")
    if String(py=tensor.dtype) != String(py=dtype):
        raise Error(name + " has an unsupported dtype")
    if Int(py=tensor.ndim) != rank:
        raise Error(name + " has an unsupported rank")
    if not Bool(py=tensor.is_contiguous()):
        raise Error(name + " must be contiguous")
    if Bool(py=tensor.requires_grad):
        raise Error(name + " must not require gradients")


def _validate_point_mesh_inputs(
    points: PythonObject,
    points_first_idx: PythonObject,
    tris: PythonObject,
    tris_first_idx: PythonObject,
    torch: PythonObject,
) raises:
    _require_tensor(points, torch, torch.float32, 2, "points")
    _require_tensor(tris, torch, torch.float32, 3, "tris")
    _require_tensor(points_first_idx, torch, torch.int64, 1, "points_first_idx")
    _require_tensor(tris_first_idx, torch, torch.int64, 1, "tris_first_idx")
    if Int(py=points.shape[1]) != 3:
        raise Error("points must have shape (P, 3)")
    if Int(py=tris.shape[1]) != 3 or Int(py=tris.shape[2]) != 3:
        raise Error("tris must have shape (T, 3, 3)")
    if Int(py=points_first_idx.shape[0]) != Int(py=tris_first_idx.shape[0]):
        raise Error("packed batch offsets must have equal lengths")


def _point_mesh_forward(
    points: PythonObject,
    points_first_idx: PythonObject,
    tris: PythonObject,
    tris_first_idx: PythonObject,
    max_outer_object: PythonObject,
    min_triangle_area_object: PythonObject,
    reverse: Bool,
) raises -> PythonObject:
    var torch = Python.import_module("torch")
    _validate_point_mesh_inputs(points, points_first_idx, tris, tris_first_idx, torch)
    var num_points = Int(py=points.shape[0])
    var num_tris = Int(py=tris.shape[0])
    var num_batches = Int(py=points_first_idx.shape[0])
    var max_outer = Int(py=max_outer_object)
    var min_triangle_area = Float64(py=min_triangle_area_object)
    var points_address = UInt64(py=points.data_ptr())
    var points_first_address = UInt64(py=points_first_idx.data_ptr())
    var tris_address = UInt64(py=tris.data_ptr())
    var tris_first_address = UInt64(py=tris_first_idx.data_ptr())
    var max_points = _max_packed_count(
        points_first_address, num_points, num_batches, "points_first_idx"
    )
    var max_tris = _max_packed_count(
        tris_first_address, num_tris, num_batches, "tris_first_idx"
    )
    var expected_max = max_tris if reverse else max_points
    if max_outer != expected_max:
        raise Error("max_outer must equal the largest packed batch")

    var outer = tris if reverse else points
    var outer_first = tris_first_idx if reverse else points_first_idx
    var outer_count = num_tris if reverse else num_points
    var dists = torch.empty(
        Python.tuple(outer_count), dtype=outer.dtype, device=outer.device
    )
    var idxs = torch.empty(
        Python.tuple(outer_count),
        dtype=outer_first.dtype,
        device=outer_first.device,
    )
    var dists_address = UInt64(py=dists.data_ptr())
    var idxs_address = UInt64(py=idxs.data_ptr())

    @parameter
    def work(work_idx: Int):
        point_mesh_distance(
            points_address,
            points_first_address,
            tris_address,
            tris_first_address,
            dists_address,
            idxs_address,
            num_points,
            num_tris,
            num_batches,
            max_outer,
            work_idx,
            reverse,
            min_triangle_area,
        )

    var total_work = num_batches * max_outer
    if total_work > 0:
        parallelize[work](total_work)
    return Python.tuple(dists, idxs)


def point_face_dist_forward(
    points: PythonObject,
    points_first_idx: PythonObject,
    tris: PythonObject,
    tris_first_idx: PythonObject,
    max_points: PythonObject,
    min_triangle_area: PythonObject,
) raises -> PythonObject:
    return _point_mesh_forward(
        points,
        points_first_idx,
        tris,
        tris_first_idx,
        max_points,
        min_triangle_area,
        False,
    )


def face_point_dist_forward(
    points: PythonObject,
    points_first_idx: PythonObject,
    tris: PythonObject,
    tris_first_idx: PythonObject,
    max_tris: PythonObject,
    min_triangle_area: PythonObject,
) raises -> PythonObject:
    return _point_mesh_forward(
        points,
        points_first_idx,
        tris,
        tris_first_idx,
        max_tris,
        min_triangle_area,
        True,
    )
