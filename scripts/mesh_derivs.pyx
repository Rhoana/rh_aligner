#cython: boundscheck=True, wraparound=False, cdivision=True
from __future__ import division
import numpy as np
cimport numpy
cimport openmp
from cython.parallel import parallel, threadid, prange
from libc.math cimport sqrt, fabs

ctypedef numpy.float64_t FLOAT_TYPE
ctypedef numpy.int32_t int32
ctypedef numpy.uint32_t uint32

npFLOAT_TYPE = np.float64

cdef:
    FLOAT_TYPE small_value = 0.0001

from libc.stdio cimport printf


##################################################
# HUBER LOSS FUNCTION
##################################################
cdef  inline FLOAT_TYPE c_huber(FLOAT_TYPE value,
                                FLOAT_TYPE target,
                                FLOAT_TYPE sigma,
                                FLOAT_TYPE d_value_dx,
                                FLOAT_TYPE d_value_dy,
                                FLOAT_TYPE *d_huber_dx,
                                FLOAT_TYPE *d_huber_dy) nogil:
    cdef:
        FLOAT_TYPE diff, a, b, l

    diff = value - target
    if fabs(diff) <= sigma:
        a = (diff ** 2) / 2
        d_huber_dx[0] = diff * d_value_dx
        d_huber_dy[0] = diff * d_value_dy
        return a
    else:
        b = sigma * (fabs(diff) - sigma / 2)
        d_huber_dx[0] = sigma * d_value_dx
        d_huber_dy[0] = sigma * d_value_dy
        return b

cpdef huber(value, target, sigma, dx, dy):
    cdef:
        FLOAT_TYPE dhx, dhy
    val = c_huber(<FLOAT_TYPE> value, <FLOAT_TYPE> target, <FLOAT_TYPE> sigma, <FLOAT_TYPE> dx, <FLOAT_TYPE> dy, &(dhx), &(dhy))
    return val, dhx, dhy

##################################################
# REGULARIZED LENGTH FUNCTION
##################################################

cdef  inline FLOAT_TYPE c_reglen(FLOAT_TYPE vx,
                                 FLOAT_TYPE vy,
                                 FLOAT_TYPE d_vx_dx,
                                 FLOAT_TYPE d_vy_dy,
                                 FLOAT_TYPE *d_reglen_dx,
                                 FLOAT_TYPE *d_reglen_dy) nogil:
    cdef:
        FLOAT_TYPE sq_len, sqrt_len

    sq_len = vx * vx + vy * vy + small_value
    sqrt_len = sqrt(sq_len)
    d_reglen_dx[0] = vx / sqrt_len
    d_reglen_dy[0] = vy / sqrt_len
    return sqrt_len

cpdef reglen(vx, vy):
    cdef:
        FLOAT_TYPE drx, dry
    val = c_reglen(<FLOAT_TYPE> vx, <FLOAT_TYPE> vy, <FLOAT_TYPE> 1.0, 1.0, &(drx), &(dry))
    return val, drx, dry

##################################################
# MESH CROSS-LINK DERIVS
##################################################
cpdef FLOAT_TYPE crosslink_mesh_derivs(FLOAT_TYPE[:, ::1] mesh1,
                                       FLOAT_TYPE[:, ::1] mesh2,
                                       FLOAT_TYPE[:, ::1] d_cost_d_mesh1,
                                       FLOAT_TYPE[:, ::1] d_cost_d_mesh2,
                                       uint32[:, ::1] triangles,
                                       uint32[::1] src_idx,
                                       uint32[::1] tri_indices,
                                       FLOAT_TYPE[:, ::1] tri_weights,
                                       FLOAT_TYPE all_weight,
                                       FLOAT_TYPE sigma) nogil:
    cdef:
        int i, ti0, ti1, ti2
        FLOAT_TYPE px, py, qx, qy
        FLOAT_TYPE w0, w1, w2
        FLOAT_TYPE r, h
        FLOAT_TYPE dr_dx, dr_dy, dh_dx, dh_dy
        FLOAT_TYPE cost

    cost = 0
    for i in range(mesh1.shape[0]):
        ti0 = triangles[tri_indices[i], 0]
        ti1 = triangles[tri_indices[i], 1]
        ti2 = triangles[tri_indices[i], 2]
        w0 = tri_weights[i, 0]
        w1 = tri_weights[i, 1]
        w2 = tri_weights[i, 2]

        px = mesh1[i, 0]
        py = mesh1[i, 1]

        qx = (mesh2[ti0, 0] * w0 +
              mesh2[ti1, 0] * w1 +
              mesh2[ti2, 0] * w2)
        qy = (mesh2[ti0, 1] * w0 +
              mesh2[ti1, 1] * w1 +
              mesh2[ti2, 1] * w2)
        r = c_reglen(px - qx, py - qy,
                     1, 1,
                     &(dr_dx), &(dr_dy))
        h = c_huber(r, 0, sigma,
                    dr_dx, dr_dy,
                    &(dh_dx), &(dh_dy))
        cost += h * all_weight
        dh_dx *= all_weight
        dh_dy *= all_weight

        # update derivs
        d_cost_d_mesh1[i, 0] += dh_dx
        d_cost_d_mesh1[i, 1] += dh_dy
        # opposite direction for other end of spring, and distributed according to weight
        d_cost_d_mesh2[ti0, 0] -= w0 * dh_dx
        d_cost_d_mesh2[ti1, 0] -= w1 * dh_dx
        d_cost_d_mesh2[ti2, 0] -= w2 * dh_dx
        d_cost_d_mesh2[ti0, 1] -= w0 * dh_dy
        d_cost_d_mesh2[ti1, 1] -= w1 * dh_dy
        d_cost_d_mesh2[ti2, 1] -= w2 * dh_dy
    return cost


##################################################
# MESH INTERNAL-LINK DERIVS
##################################################
cpdef FLOAT_TYPE internal_mesh_derivs(FLOAT_TYPE[:, ::1] mesh,
                                      FLOAT_TYPE[:, ::1] d_cost_d_mesh,
                                      uint32[:, ::1] edge_indices,
                                      FLOAT_TYPE[:] rest_lengths,
                                      FLOAT_TYPE all_weight,
                                      FLOAT_TYPE sigma) nogil:
    cdef:
        int i
        int idx1, idx2
        FLOAT_TYPE px, py, qx, qy
        FLOAT_TYPE r, h
        FLOAT_TYPE dr_dx, dr_dy, dh_dx, dh_dy
        FLOAT_TYPE cost

    cost = 0
    for i in range(edge_indices.shape[0]):
        idx1 = edge_indices[i, 0]
        idx2 = edge_indices[i, 1]

        px = mesh[idx1, 0]
        py = mesh[idx1, 1]
        qx = mesh[idx2, 0]
        qy = mesh[idx2, 1]

        r = c_reglen(px - qx, py - qy,
                     1, 1,
                     &(dr_dx), &(dr_dy))
        h = c_huber(r, rest_lengths[i], sigma,
                    dr_dx, dr_dy,
                    &(dh_dx), &(dh_dy))
        cost += h * all_weight
        dh_dx *= all_weight
        dh_dy *= all_weight

        # update derivs
        d_cost_d_mesh[idx1, 0] += dh_dx
        d_cost_d_mesh[idx1, 1] += dh_dy
        d_cost_d_mesh[idx2, 0] -= dh_dx
        d_cost_d_mesh[idx2, 1] -= dh_dy

    return cost

##################################################
# ALL DERIVS IN PARALLEL
##################################################

cpdef FLOAT_TYPE all_derivs(FLOAT_TYPE[:, :, ::1] meshes,
                            numpy.ndarray[FLOAT_TYPE, ndim=3] d_cost_d_meshes,
                            uint32[:, ::1] triangles,  # Nx3, same for all meshes
                            uint32[:, ::1] pairs,  # Nx2
                            FLOAT_TYPE[::1] between_mesh_weights,
                            uint32[::1] src_indices,
                            uint32[::1] tri_indices,
                            FLOAT_TYPE[:, ::1] tri_weights,
                            uint32[::1] pair_offsets,
                            uint32[:, ::1] internal_edge_indices,   # same for all meshes
                            FLOAT_TYPE[::1] internal_rest_lengths,  # same for all meshes
                            FLOAT_TYPE within_mesh_weight,
                            FLOAT_TYPE between_winsor,
                            FLOAT_TYPE within_winsor,
                            uint32 lo, uint32 hi,
                            int num_threads):
    cdef:
        int num_meshes, num_pairs, num_pts
        FLOAT_TYPE[:, :, :, ::1] d_cost_per_thread
        FLOAT_TYPE[:] costs
        int m1, m2, i, j, k, tid, m1d_idx, m2d_idx
        uint32 offset, num_matches

    num_meshes = meshes.shape[0]
    num_pts = meshes.shape[1]
    num_pairs = pairs.shape[0]

    # we allocate one extra block for derivatives outside [lo..hi)
    np_d_cost_per_thread = np.zeros((num_threads, hi - lo + 1, meshes.shape[1], 2), dtype=npFLOAT_TYPE)
    d_cost_per_thread = np_d_cost_per_thread

    np_costs = np.zeros(num_threads, dtype=npFLOAT_TYPE)
    costs = np_costs

    with nogil:
        # between mesh cost
        for i in prange(num_pairs, num_threads=num_threads, schedule='dynamic'):
            tid = threadid()

            m1 = pairs[i, 0]
            m2 = pairs[i, 1]
            # index into the d_cost_per_thread array
            m1d_idx = (m1 - lo) if (m1 >= lo) else (hi - lo)
            m2d_idx = (m2 - lo) if (m2 >= lo) else (hi - lo)

            if (m1 < lo) and (m2 < lo):  # ignore already-processed pairs, but keep ones that straddle
                continue
            if (m1 >= hi) or (m2 >= hi):  # ignore to-be-processed meshes completely
                continue

            offset = pair_offsets[i]
            num_matches = pair_offsets[i + 1] - offset

            costs[tid] += crosslink_mesh_derivs(meshes[m1, ...],
                                                meshes[m2, ...],
                                                d_cost_per_thread[tid, m1d_idx, ...],
                                                d_cost_per_thread[tid, m2d_idx, ...],
                                                triangles,
                                                src_indices[offset:(offset + num_matches)],
                                                tri_indices[offset:(offset + num_matches)],
                                                tri_weights[offset:(offset + num_matches), ...],
                                                between_mesh_weights[i],
                                                between_winsor)

        # within mesh cost
        for i in prange(num_meshes, num_threads=num_threads, schedule='dynamic'):
            tid = threadid()
            # compute interior costs and derivs, and sum into output derivs
            if (i < lo) or (i >= hi):  # ignore already-processed meshes
                continue
            costs[tid] += internal_mesh_derivs(meshes[i, ...],
                                               d_cost_per_thread[tid, i - lo, ...],
                                               internal_edge_indices,
                                               internal_rest_lengths,
                                               within_mesh_weight,
                                               within_winsor)

    # ignore last block of derivatives (see above)
    np_d_cost_per_thread[:, :-1, ...].sum(axis=0, out=d_cost_d_meshes)

    return np_costs.sum()

def compare(x, y, eps, restlen, sigma):
    l, dl_dx, dl_dy = reglen(x, y)
    h0, dh_dx, dh_dy = huber(l, restlen, sigma, dl_dx, dl_dy)
    hx = huber(reglen(x+eps, y)[0], restlen, sigma, dl_dx, dl_dy)[0]
    hy = huber(reglen(x, y+eps)[0], restlen, sigma, dl_dx, dl_dy)[0]
    print x, y, restlen, sigma, "->", (dh_dx, dh_dy), "vs", ((hx - h0) / eps, (hy - h0) / eps)

if __name__ == '__main__':
    eps = 0.00001
    compare(2.0, 10.0, eps, 5.0, 20.0)
    compare(2.0, 10.0, eps, 5.0, 2.0)
    compare(2.0, 10.0, eps, 3.0, 20.0)
    compare(2.0, 10.0, eps, 3.0, 2.0)
    compare(-2.0, 10.0, eps, 3.0, 2.0)

    compare(10.0, 2.0, eps, 5.0, 20.0)
    compare(10.0, 2.0, eps, 5.0, 2.0)
    compare(10.0, 2.0, eps, 3.0, 20.0)
    compare(10.0, 2.0, eps, 3.0, 2.0)
    compare(-10.0, -2.0, eps, 3.0, 2.0)

