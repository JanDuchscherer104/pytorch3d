# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

from point_mesh_mojo import point_mesh_distance
from rasterize_meshes_mojo import rasterize_meshes
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
        module.def_function[rasterize_meshes_forward]("rasterize_meshes_forward")
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


def _validate_raster_packing(
    mesh_first_address: UInt64,
    counts_address: UInt64,
    neighbor_address: UInt64,
    num_faces: Int,
    num_batches: Int,
) raises:
    var mesh_first = UnsafePointer[Int64, MutUntrackedOrigin](
        unsafe_from_address=Int(mesh_first_address)
    )
    var counts = UnsafePointer[Int64, MutUntrackedOrigin](
        unsafe_from_address=Int(counts_address)
    )
    var neighbors = UnsafePointer[Int64, MutUntrackedOrigin](
        unsafe_from_address=Int(neighbor_address)
    )
    if num_batches == 0 and num_faces != 0:
        raise Error("empty mesh batches require empty face_verts")
    var expected_start = 0
    for batch in range(num_batches):
        var start = Int(mesh_first[batch])
        var count = Int(counts[batch])
        if start != expected_start:
            raise Error("mesh_to_face_first_idx must exactly pack face_verts")
        if count < 0 or count > num_faces - start:
            raise Error("num_faces_per_mesh is outside face_verts bounds")
        expected_start = start + count
    if expected_start != num_faces:
        raise Error("packed mesh counts must cover face_verts")
    for face in range(num_faces):
        var neighbor = Int(neighbors[face])
        if neighbor < -1 or neighbor >= num_faces:
            raise Error("clipped_faces_neighbor_idx is outside face_verts bounds")


def rasterize_meshes_forward(
    face_verts: PythonObject,
    mesh_to_face_first_idx: PythonObject,
    num_faces_per_mesh: PythonObject,
    clipped_faces_neighbor_idx: PythonObject,
    height_object: PythonObject,
    width_object: PythonObject,
) raises -> PythonObject:
    var torch = Python.import_module("torch")
    _require_tensor(face_verts, torch, torch.float32, 3, "face_verts")
    _require_tensor(
        mesh_to_face_first_idx,
        torch,
        torch.int64,
        1,
        "mesh_to_face_first_idx",
    )
    _require_tensor(num_faces_per_mesh, torch, torch.int64, 1, "num_faces_per_mesh")
    _require_tensor(
        clipped_faces_neighbor_idx,
        torch,
        torch.int64,
        1,
        "clipped_faces_neighbor_idx",
    )
    if Int(py=face_verts.shape[1]) != 3 or Int(py=face_verts.shape[2]) != 3:
        raise Error("face_verts must have shape (F, 3, 3)")
    var num_faces = Int(py=face_verts.shape[0])
    var num_batches = Int(py=mesh_to_face_first_idx.shape[0])
    if Int(py=num_faces_per_mesh.shape[0]) != num_batches:
        raise Error("packed mesh metadata must have equal lengths")
    if Int(py=clipped_faces_neighbor_idx.shape[0]) != num_faces:
        raise Error("clipped_faces_neighbor_idx must have one entry per face")
    var height = Int(py=height_object)
    var width = Int(py=width_object)
    if height <= 0 or width <= 0:
        raise Error("image height and width must be positive")

    var face_verts_address = UInt64(py=face_verts.data_ptr())
    var mesh_first_address = UInt64(py=mesh_to_face_first_idx.data_ptr())
    var counts_address = UInt64(py=num_faces_per_mesh.data_ptr())
    var neighbor_address = UInt64(py=clipped_faces_neighbor_idx.data_ptr())
    _validate_raster_packing(
        mesh_first_address,
        counts_address,
        neighbor_address,
        num_faces,
        num_batches,
    )

    var output_shape = Python.tuple(num_batches, height, width, 1)
    var face_idxs = torch.full(
        output_shape, -1, dtype=torch.int64, device=face_verts.device
    )
    var zbuf = torch.full(
        output_shape, -1, dtype=torch.float32, device=face_verts.device
    )
    var barycentric_coords = torch.full(
        Python.tuple(num_batches, height, width, 1, 3),
        -1,
        dtype=torch.float32,
        device=face_verts.device,
    )
    var dists = torch.full(
        output_shape, -1, dtype=torch.float32, device=face_verts.device
    )
    var face_idxs_address = UInt64(py=face_idxs.data_ptr())
    var zbuf_address = UInt64(py=zbuf.data_ptr())
    var bary_address = UInt64(py=barycentric_coords.data_ptr())
    var dists_address = UInt64(py=dists.data_ptr())

    @parameter
    def work(work_idx: Int):
        rasterize_meshes(
            face_verts_address,
            mesh_first_address,
            counts_address,
            neighbor_address,
            face_idxs_address,
            zbuf_address,
            bary_address,
            dists_address,
            num_faces,
            num_batches,
            height,
            width,
            work_idx,
        )

    var total_work = num_batches * height * width
    if total_work > 0:
        parallelize[work](total_work)
    return Python.tuple(face_idxs, zbuf, barycentric_coords, dists)


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
