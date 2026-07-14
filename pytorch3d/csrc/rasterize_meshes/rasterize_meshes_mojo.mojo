# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.


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


def _edge(
    px: Float32,
    py: Float32,
    x0: Float32,
    y0: Float32,
    x1: Float32,
    y1: Float32,
) -> Float32:
    return (px - x0) * (y1 - y0) - (py - y0) * (x1 - x0)


def _line_distance_sq(
    px: Float32,
    py: Float32,
    x0: Float32,
    y0: Float32,
    x1: Float32,
    y1: Float32,
) -> Float32:
    var dx = x1 - x0
    var dy = y1 - y0
    var length_sq = dx * dx + dy * dy
    if length_sq <= 1e-8:
        var ex = px - x1
        var ey = py - y1
        return ex * ex + ey * ey
    var t = (dx * (px - x0) + dy * (py - y0)) / length_sq
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    var ex = px - (x0 + t * dx)
    var ey = py - (y0 + t * dy)
    return ex * ex + ey * ey


def _pixel_ndc(index: Int, size: Int, other_size: Int) -> Float32:
    var ndc_range = Float32(2.0)
    if size > other_size:
        ndc_range = Float32(size) * ndc_range / Float32(other_size)
    var offset = ndc_range / 2.0
    return -offset + (ndc_range * Float32(index) + offset) / Float32(size)


def rasterize_meshes(
    face_verts_address: UInt64,
    mesh_first_address: UInt64,
    counts_address: UInt64,
    neighbor_address: UInt64,
    face_idxs_address: UInt64,
    zbuf_address: UInt64,
    bary_address: UInt64,
    dists_address: UInt64,
    num_faces: Int,
    num_batches: Int,
    height: Int,
    width: Int,
    work_idx: Int,
) -> None:
    if height <= 0 or width <= 0 or work_idx < 0:
        return
    var pixels_per_batch = height * width
    var batch_idx = work_idx // pixels_per_batch
    if batch_idx >= num_batches:
        return

    var face_verts = _f32_ptr(face_verts_address)
    var mesh_first = _i64_ptr(mesh_first_address)
    var counts = _i64_ptr(counts_address)
    var neighbors = _i64_ptr(neighbor_address)
    var face_idxs = _i64_mut_ptr(face_idxs_address)
    var zbuf = _f32_mut_ptr(zbuf_address)
    var bary = _f32_mut_ptr(bary_address)
    var dists = _f32_mut_ptr(dists_address)

    var pixel_idx = work_idx % pixels_per_batch
    var yi = pixel_idx // width
    var xi = pixel_idx % width
    var yf = _pixel_ndc(height - 1 - yi, height, width)
    var xf = _pixel_ndc(width - 1 - xi, width, height)
    var face_start = Int(mesh_first[batch_idx])
    var face_stop = face_start + Int(counts[batch_idx])
    if face_start < 0 or face_stop < face_start or face_stop > num_faces:
        return

    var have_best = False
    var best_face = -1
    var best_z = Float32(0.0)
    var best_dist = Float32(0.0)
    var best_w0 = Float32(0.0)
    var best_w1 = Float32(0.0)
    var best_w2 = Float32(0.0)

    for face_idx in range(face_start, face_stop):
        var offset = face_idx * 9
        var x0 = face_verts[offset]
        var y0 = face_verts[offset + 1]
        var z0 = face_verts[offset + 2]
        var x1 = face_verts[offset + 3]
        var y1 = face_verts[offset + 4]
        var z1 = face_verts[offset + 5]
        var x2 = face_verts[offset + 6]
        var y2 = face_verts[offset + 7]
        var z2 = face_verts[offset + 8]

        var z_min = z1 if z1 < z0 else z0
        z_min = z2 if z2 < z_min else z_min
        if z_min <= 1e-8:
            continue
        var face_area = _edge(x0, y0, x1, y1, x2, y2)
        if face_area <= 1e-8 and face_area >= -1e-8:
            continue

        var x_min = x1 if x1 < x0 else x0
        x_min = x2 if x2 < x_min else x_min
        var x_max = x1 if x1 > x0 else x0
        x_max = x2 if x2 > x_max else x_max
        var y_min = y1 if y1 < y0 else y0
        y_min = y2 if y2 < y_min else y_min
        var y_max = y1 if y1 > y0 else y0
        y_max = y2 if y2 > y_max else y_max
        if xf > x_max or xf < x_min or yf > y_max or yf < y_min:
            continue

        var bary_area = _edge(x2, y2, x0, y0, x1, y1) + 1e-8
        var w0 = _edge(xf, yf, x1, y1, x2, y2) / bary_area
        var w1 = _edge(xf, yf, x2, y2, x0, y0) / bary_area
        var w2 = _edge(xf, yf, x0, y0, x1, y1) / bary_area
        var w0_top = w0 * z1 * z2
        var w1_top = w1 * z0 * z2
        var w2_top = w2 * z0 * z1
        var perspective_denominator = w0_top + w1_top + w2_top
        if perspective_denominator < 1e-8:
            perspective_denominator = 1e-8
        w0 = w0_top / perspective_denominator
        w1 = w1_top / perspective_denominator
        w2 = w2_top / perspective_denominator
        var pz = w0 * z0 + w1 * z1 + w2 * z2
        if pz < 0.0 or w0 <= 0.0 or w1 <= 0.0 or w2 <= 0.0:
            continue

        var d01 = _line_distance_sq(xf, yf, x0, y0, x1, y1)
        var d02 = _line_distance_sq(xf, yf, x0, y0, x2, y2)
        var d12 = _line_distance_sq(xf, yf, x1, y1, x2, y2)
        var distance = d02 if d02 < d01 else d01
        distance = d12 if d12 < distance else distance
        var replace = False
        if have_best and Int(neighbors[face_idx]) == best_face:
            replace = distance < -best_dist
        elif not have_best or pz < best_z or (pz == best_z and face_idx < best_face):
            replace = True
        if replace:
            have_best = True
            best_face = face_idx
            best_z = pz
            best_dist = -distance
            best_w0 = w0
            best_w1 = w1
            best_w2 = w2

    if have_best:
        face_idxs[work_idx] = Int64(best_face)
        zbuf[work_idx] = best_z
        dists[work_idx] = best_dist
        var bary_offset = work_idx * 3
        bary[bary_offset] = best_w0
        bary[bary_offset + 1] = best_w1
        bary[bary_offset + 2] = best_w2
