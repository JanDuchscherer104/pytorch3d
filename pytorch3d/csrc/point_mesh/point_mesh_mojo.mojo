# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

from std.ffi import external_call
from std.math import sqrt


def _f32_ptr(address: UInt64) -> UnsafePointer[Float32, MutUntrackedOrigin]:
    return UnsafePointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(address))


def _i64_ptr(address: UInt64) -> UnsafePointer[Int64, MutUntrackedOrigin]:
    return UnsafePointer[Int64, MutUntrackedOrigin](unsafe_from_address=Int(address))


def _f32_mut_ptr(
    address: UInt64,
) -> UnsafePointer[mut=True, Float32, MutUntrackedOrigin]:
    return UnsafePointer[mut=True, Float32, MutUntrackedOrigin](
        unsafe_from_address=Int(address)
    )


def _i64_mut_ptr(
    address: UInt64,
) -> UnsafePointer[mut=True, Int64, MutUntrackedOrigin]:
    return UnsafePointer[mut=True, Int64, MutUntrackedOrigin](
        unsafe_from_address=Int(address)
    )


def _dot(
    ax: Float32, ay: Float32, az: Float32, bx: Float32, by: Float32, bz: Float32
) -> Float32:
    return ax * bx + ay * by + az * bz


def _line_distance_sq(
    px: Float32,
    py: Float32,
    pz: Float32,
    ax: Float32,
    ay: Float32,
    az: Float32,
    bx: Float32,
    by: Float32,
    bz: Float32,
) -> Float32:
    var dx = bx - ax
    var dy = by - ay
    var dz = bz - az
    var length_sq = _dot(dx, dy, dz, dx, dy, dz)
    if length_sq <= 1e-8:
        return _dot(px - bx, py - by, pz - bz, px - bx, py - by, pz - bz)
    var t = _dot(dx, dy, dz, px - ax, py - ay, pz - az) / length_sq
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    var qx = ax + t * dx
    var qy = ay + t * dy
    var qz = az + t * dz
    return _dot(px - qx, py - qy, pz - qz, px - qx, py - qy, pz - qz)


def _triangle_distance_sq(
    px: Float32,
    py: Float32,
    pz: Float32,
    ax: Float32,
    ay: Float32,
    az: Float32,
    bx: Float32,
    by: Float32,
    bz: Float32,
    cx: Float32,
    cy: Float32,
    cz: Float32,
    min_area: Float64,
) -> Float32:
    var abx = bx - ax
    var aby = by - ay
    var abz = bz - az
    var acx = cx - ax
    var acy = cy - ay
    var acz = cz - az
    var nx = acy * abz - acz * aby
    var ny = acz * abx - acx * abz
    var nz = acx * aby - acy * abx
    var normal_length = sqrt(nx * nx + ny * ny + nz * nz)
    var area_x = external_call["fmaf", Float32](aby, acz, -(abz * acy))
    var area_y = external_call["fmaf", Float32](abz, acx, -(abx * acz))
    var area_z = external_call["fmaf", Float32](abx, acy, -(aby * acx))
    var area_length = external_call["hypotf", Float32](
        area_x, external_call["hypotf", Float32](area_y, area_z)
    )
    var inverse_normal_length = 1.0 / (normal_length + 1e-8)
    nx *= inverse_normal_length
    ny *= inverse_normal_length
    nz *= inverse_normal_length

    var plane_distance = _dot(ax - px, ay - py, az - pz, nx, ny, nz)
    var qx = px + plane_distance * nx
    var qy = py + plane_distance * ny
    var qz = pz + plane_distance * nz

    var qax = qx - ax
    var qay = qy - ay
    var qaz = qz - az
    var d00 = _dot(abx, aby, abz, abx, aby, abz)
    var d01 = _dot(abx, aby, abz, acx, acy, acz)
    var d11 = _dot(acx, acy, acz, acx, acy, acz)
    var d20 = _dot(qax, qay, qaz, abx, aby, abz)
    var d21 = _dot(qax, qay, qaz, acx, acy, acz)
    var denominator = d00 * d11 - d01 * d01 + 1e-8
    var w1 = (d11 * d20 - d01 * d21) / denominator
    var w2 = (d00 * d21 - d01 * d20) / denominator
    var w0 = 1.0 - w1 - w2
    var inside = Float64(area_length) * 0.5 >= min_area
    inside = inside and w0 >= 0.0 and w0 <= 1.0
    inside = inside and w1 >= 0.0 and w1 <= 1.0
    inside = inside and w2 >= 0.0 and w2 <= 1.0
    if inside and normal_length > 1e-8:
        return plane_distance * plane_distance

    var e01 = _line_distance_sq(px, py, pz, ax, ay, az, bx, by, bz)
    var e02 = _line_distance_sq(px, py, pz, ax, ay, az, cx, cy, cz)
    var e12 = _line_distance_sq(px, py, pz, bx, by, bz, cx, cy, cz)
    var distance = e02 if e01 > e02 else e01
    return e12 if distance > e12 else distance


def point_mesh_distance(
    points_address: UInt64,
    points_first_idx_address: UInt64,
    tris_address: UInt64,
    tris_first_idx_address: UInt64,
    dists_address: UInt64,
    idxs_address: UInt64,
    num_points: Int,
    num_tris: Int,
    num_batches: Int,
    max_outer: Int,
    work_idx: Int,
    reverse: Bool,
    min_triangle_area: Float64,
) -> None:
    var points = _f32_ptr(points_address)
    var points_first_idx = _i64_ptr(points_first_idx_address)
    var tris = _f32_ptr(tris_address)
    var tris_first_idx = _i64_ptr(tris_first_idx_address)
    var dists = _f32_mut_ptr(dists_address)
    var idxs = _i64_mut_ptr(idxs_address)
    var batch_idx = work_idx // max_outer
    var point_start = Int(points_first_idx[batch_idx])
    var tri_start = Int(tris_first_idx[batch_idx])
    var point_end = num_points
    var tri_end = num_tris
    if batch_idx + 1 < num_batches:
        point_end = Int(points_first_idx[batch_idx + 1])
        tri_end = Int(tris_first_idx[batch_idx + 1])

    var outer_start = point_start
    var outer_end = point_end
    var target_start = tri_start
    var target_end = tri_end
    if reverse:
        outer_start = tri_start
        outer_end = tri_end
        target_start = point_start
        target_end = point_end
    var outer_idx = outer_start + work_idx % max_outer
    if outer_idx >= outer_end:
        return

    var best = Float32.MAX_FINITE
    var best_idx = Int64(0)
    if not reverse:
        var point_offset = outer_idx * 3
        var px = points[point_offset]
        var py = points[point_offset + 1]
        var pz = points[point_offset + 2]
        for tri_idx in range(target_start, target_end):
            var tri_offset = tri_idx * 9
            var distance = _triangle_distance_sq(
                px,
                py,
                pz,
                tris[tri_offset],
                tris[tri_offset + 1],
                tris[tri_offset + 2],
                tris[tri_offset + 3],
                tris[tri_offset + 4],
                tris[tri_offset + 5],
                tris[tri_offset + 6],
                tris[tri_offset + 7],
                tris[tri_offset + 8],
                min_triangle_area,
            )
            if distance <= best:
                best = distance
                best_idx = Int64(tri_idx)
    else:
        var tri_offset = outer_idx * 9
        var ax = tris[tri_offset]
        var ay = tris[tri_offset + 1]
        var az = tris[tri_offset + 2]
        var bx = tris[tri_offset + 3]
        var by = tris[tri_offset + 4]
        var bz = tris[tri_offset + 5]
        var cx = tris[tri_offset + 6]
        var cy = tris[tri_offset + 7]
        var cz = tris[tri_offset + 8]
        for point_idx in range(target_start, target_end):
            var point_offset = point_idx * 3
            var distance = _triangle_distance_sq(
                points[point_offset],
                points[point_offset + 1],
                points[point_offset + 2],
                ax,
                ay,
                az,
                bx,
                by,
                bz,
                cx,
                cy,
                cz,
                min_triangle_area,
            )
            if distance <= best:
                best = distance
                best_idx = Int64(point_idx)
    dists[outer_idx] = best
    idxs[outer_idx] = best_idx
