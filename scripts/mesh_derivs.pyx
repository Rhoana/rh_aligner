#cython: boundscheck=True, wraparound=False
from __future__ import division
import numpy as np
cimport numpy
from cython.parallel import parallel, threadid
from libc.math cimport sin, cos, acos, exp, sqrt, fabs, M_PI

ctypedef numpy.float64_t FLOAT_TYPE
ctypedef numpy.int32_t int32
ctypedef numpy.uint32_t uint32

npFLOAT_TYPE = np.float64

cdef:
    FLOAT_TYPE small_value = 0.0001



##################################################
# HUBER LOSS FUNCTION
##################################################
cdef inline FLOAT_TYPE c_huber(FLOAT_TYPE value,
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

cdef inline FLOAT_TYPE c_reglen(FLOAT_TYPE vx,
                             FLOAT_TYPE vy,
                             FLOAT_TYPE d_vx_dx,
                             FLOAT_TYPE d_vy_dy,
                             FLOAT_TYPE *d_reglen_dx,
                             FLOAT_TYPE *d_reglen_dy) nogil:
    cdef:
        FLOAT_TYPE sq_len, sqrt_len

    if (d_vx_dx != 1) or (d_vy_dy != 1):
        with gil:
            assert d_vx_dx == 1
            assert d_vy_dy == 1

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
                                    int32[:, ::1] idx2,
                                    FLOAT_TYPE[:, ::1] weight2,
                                    FLOAT_TYPE all_weight,
                                    FLOAT_TYPE sigma) nogil:
    cdef:
        int i
        FLOAT_TYPE px, py, qx, qy
        FLOAT_TYPE r, h
        FLOAT_TYPE dr_dx, dr_dy, dh_dx, dh_dy
        FLOAT_TYPE cost

    cost = 0
    for i in range(mesh1.shape[0]):
        if idx2[i, 0] == -1:
            continue
        px = mesh1[i, 0]
        py = mesh1[i, 1]
        qx = (mesh2[idx2[i, 0], 0] * weight2[i, 0] +
              mesh2[idx2[i, 1], 0] * weight2[i, 1] +
              mesh2[idx2[i, 2], 0] * weight2[i, 2])
        qy = (mesh2[idx2[i, 0], 1] * weight2[i, 0] +
              mesh2[idx2[i, 1], 1] * weight2[i, 1] +
              mesh2[idx2[i, 2], 1] * weight2[i, 2])
        r = c_reglen(px - qx, py - qy,
                     1, 1,
                     &(dr_dx), &(dr_dy))
        h = c_huber(r, 0, sigma,
                    dr_dx, dr_dy,
                    &(dh_dx), &(dh_dy))
        cost += h * all_weight
        dh_dx *= all_weight
        dh_dy *= all_weight
        if i == 1500:
            with gil:
                print "     D:", r, "C:", cost, "HC:", h * all_weight, "DH:", dh_dx, dh_dy, "A:", all_weight, "P:", px, py, "Del:", px - qx, py - qy, "der:", dr_dx, dr_dy

        # update derivs
        d_cost_d_mesh1[i, 0] += dh_dx
        d_cost_d_mesh1[i, 1] += dh_dy
        # opposite direction for other end of spring, and distributed according to weight
        d_cost_d_mesh2[idx2[i, 0], 0] -= weight2[i, 0] * dh_dx
        d_cost_d_mesh2[idx2[i, 1], 0] -= weight2[i, 1] * dh_dx
        d_cost_d_mesh2[idx2[i, 2], 0] -= weight2[i, 2] * dh_dx
        d_cost_d_mesh2[idx2[i, 0], 1] -= weight2[i, 0] * dh_dy
        d_cost_d_mesh2[idx2[i, 1], 1] -= weight2[i, 1] * dh_dy
        d_cost_d_mesh2[idx2[i, 2], 1] -= weight2[i, 2] * dh_dy
    with gil:
        print "     CCC", cost
    return cost


##################################################
# MESH INTERNAL-LINK DERIVS
##################################################
cpdef FLOAT_TYPE internal_mesh_derivs(FLOAT_TYPE[:, ::1] mesh,
                                   FLOAT_TYPE[:, ::1] d_cost_d_mesh,
                                   uint32[:] idx,
                                   FLOAT_TYPE[:] rest_lengths,
                                   FLOAT_TYPE all_weight,
                                   FLOAT_TYPE sigma) nogil:
    cdef:
        int i
        FLOAT_TYPE px, py, qx, qy
        FLOAT_TYPE r, h
        FLOAT_TYPE dr_dx, dr_dy, dh_dx, dh_dy
        FLOAT_TYPE cost

    cost = 0
    for i in range(mesh.shape[0]):
        if idx[i] == i:
            cost += small_value / 2.0
            continue
        px = mesh[i, 0]
        py = mesh[i, 1]
        qx = mesh[idx[i], 0]
        qy = mesh[idx[i], 1]
        r = c_reglen(px - qx, py - qy,
                     1, 1,
                     &(dr_dx), &(dr_dy))
        h = c_huber(r, rest_lengths[i], sigma,
                    dr_dx, dr_dy,
                    &(dh_dx), &(dh_dy))
        cost += h * all_weight
        dh_dx *= all_weight
        dh_dy *= all_weight

        if i == 1500 or idx[i] == 1500:
            with gil:
                if i == 1500:
                    print h * all_weight, dh_dx, dh_dy, cost
                else:
                    print -h * all_weight, "D", -dh_dx, -dh_dy, "C", cost
                        

        # update derivs
        d_cost_d_mesh[i, 0] += dh_dx
        d_cost_d_mesh[i, 1] += dh_dy
        d_cost_d_mesh[idx[i], 0] -= dh_dx
        d_cost_d_mesh[idx[i], 1] -= dh_dy

    return cost

##################################################
# ALL DERIVS IN PARALLEL
##################################################

cpdef FLOAT_TYPE all_derivs(FLOAT_TYPE[:, :, ::1] meshes,
                         FLOAT_TYPE[:, :, ::1] d_cost_d_meshes,
                         uint32[:, ::1] internal_neighbor_idx,   # same for all meshes
                         FLOAT_TYPE[:, ::1] internal_rest_lengths,  # same for all meshes
                         int32[:, ::1] bary_indices,
                         FLOAT_TYPE[:, ::1] bary_weights,
                         FLOAT_TYPE[::1] between_mesh_weights,
                         FLOAT_TYPE within_mesh_weight,
                         FLOAT_TYPE between_winsor,
                         FLOAT_TYPE within_winsor,
                         uint32[:, ::1] pairs_and_offsets,
                         int num_threads):
    cdef:
        int num_meshes, num_pairs, num_internal_neighbors, num_pts
        FLOAT_TYPE[:, :, :, ::1] d_cost_per_thread
        FLOAT_TYPE[:] costs
        int m1, m2, i, j, k, tid
        uint32 boffset

    num_meshes = meshes.shape[0]
    num_pts = meshes.shape[1]
    num_pairs = pairs_and_offsets.shape[0]
    num_internal_neighbors = internal_neighbor_idx.shape[1]
    _scratch = np.zeros((num_threads, meshes.shape[0], meshes.shape[1], 2), dtype=npFLOAT_TYPE)
    d_cost_per_thread = _scratch
    _costs = np.zeros(num_threads, dtype=npFLOAT_TYPE)
    costs = _costs

    with nogil, parallel(num_threads=num_threads):
        tid = threadid()
        costs[tid] = 0.0
        for i from tid <= i < num_pairs by num_threads:
            m1 = pairs_and_offsets[i, 0]
            m2 = pairs_and_offsets[i, 1]
            boffset = pairs_and_offsets[i, 2]
            with gil:
                print "CROSS", m1, m2, costs[tid]
            costs[tid] += crosslink_mesh_derivs(meshes[m1, ...],
                                                meshes[m2, ...],
                                                d_cost_per_thread[tid, m1, ...],
                                                d_cost_per_thread[tid, m2, ...],
                                                bary_indices[boffset:(boffset + num_pts), ...],
                                                bary_weights[boffset:(boffset + num_pts), ...],
                                                between_mesh_weights[i],
                                                between_winsor)
            # swap and do the other direction
            m1, m2 = m2, m1
            boffset = pairs_and_offsets[i, 3]
            with gil:
                print "CROSS", m1, m2, costs[tid]
            costs[tid] += crosslink_mesh_derivs(meshes[m1, ...],
                                                meshes[m2, ...],
                                                d_cost_per_thread[tid, m1, ...],
                                                d_cost_per_thread[tid, m2, ...],
                                                bary_indices[boffset:(boffset + num_pts), ...],
                                                bary_weights[boffset:(boffset + num_pts), ...],
                                                between_mesh_weights[i],
                                                between_winsor)

        # compute interior costs and derivs, and sum into output derivs
        for i from tid <= i < num_meshes by num_threads:
            d_cost_d_meshes[i, ...] = 0
            with gil:
                print "INT", i
            for j from 0 <= j < num_internal_neighbors:
                costs[tid] += internal_mesh_derivs(meshes[i, ...],
                                                   d_cost_d_meshes[i, ...],
                                                   internal_neighbor_idx[:, j],
                                                   internal_rest_lengths[:, j],
                                                   within_mesh_weight,
                                                   within_winsor)
            for j from 0 <= j < num_threads:
                for k from 0 <= k < num_pts:
                    d_cost_d_meshes[i, k, 0] += d_cost_per_thread[j, i, k, 0]
                    d_cost_d_meshes[i, k, 1] += d_cost_per_thread[j, i, k, 1]

    return _costs.sum()

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
