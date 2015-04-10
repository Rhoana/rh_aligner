#cython: boundscheck=False, wraparound=False
from __future__ import division
import numpy as np
cimport numpy
from libc.math cimport sin, cos, acos, exp, sqrt, fabs, M_PI

__all__ = ['compute_mesh_derivs']

ctypedef numpy.float32_t float32
ctypedef numpy.uint32_t uint32

cdef:
    float32 small_value = 0.0001


##################################################
# HUBER LOSS FUNCTION
##################################################
cdef inline float32 c_huber(float32 value,
                            float32 target,
                            float32 sigma,
                            float32 d_value_dx,
                            float32 d_value_dy,
                            float32 *d_huber_dx,
                            float32 *d_huber_dy) nogil:
    cdef:
        float32 diff, a, b, l

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
        float32 dhx, dhy
    val = c_huber(<float32> value, <float32> target, <float32> sigma, <float32> dx, <float32> dy, &(dhx), &(dhy))
    return val, dhx, dhy

##################################################
# REGULARIZED LENGTH FUNCTION
##################################################

cdef inline float32 c_reglen(float32 vx,
                             float32 vy,
                             float32 d_vx_dx,
                             float32 d_vy_dy,
                             float32 *d_reglen_dx,
                             float32 *d_reglen_dy) nogil:
    cdef:
        float32 sq_len, sqrt_len

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
        float32 drx, dry
    val = c_reglen(<float32> vx, <float32> vy, <float32> 1.0, 1.0, &(drx), &(dry))
    return val, drx, dry

##################################################
# MESH CROSS-LINK DERIVS
##################################################
cpdef float32 crosslink_mesh_derivs(float32[:, ::1] mesh1,
                                    float32[:, ::1] mesh2,
                                    float32[:, ::1] d_cost_d_mesh1,
                                    float32[:, ::1] d_cost_d_mesh2,
                                    uint32[:] idx1,
                                    uint32[:, ::1] idx2,
                                    float32[:, ::1] weight2,
                                    float32 all_weight,
                                    float32 sigma):
    cdef:
        int i
        float32 px, py, qx, qy
        float32 r, h
        float32 dr_dx, dr_dy, dh_dx, dh_dy
        float32 cost

    cost = 0
    with nogil:
        for i in range(idx1.shape[0]):
            px = mesh1[idx1[i], 0]
            py = mesh1[idx1[i], 1]
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

            # update derivs
            d_cost_d_mesh1[idx1[i], 0] += dh_dx
            d_cost_d_mesh1[idx1[i], 1] += dh_dy
            # opposite direction for other end of spring, and distributed according to weight
            d_cost_d_mesh2[idx2[i, 0], 0] -= weight2[i, 0] * dh_dx
            d_cost_d_mesh2[idx2[i, 1], 0] -= weight2[i, 1] * dh_dx
            d_cost_d_mesh2[idx2[i, 2], 0] -= weight2[i, 2] * dh_dx
            d_cost_d_mesh2[idx2[i, 0], 1] -= weight2[i, 0] * dh_dy
            d_cost_d_mesh2[idx2[i, 1], 1] -= weight2[i, 1] * dh_dy
            d_cost_d_mesh2[idx2[i, 2], 1] -= weight2[i, 2] * dh_dy
    return cost


##################################################
# MESH INTERNAL-LINK DERIVS
##################################################
cpdef float32 internal_mesh_derivs(float32[:, ::1] mesh,
                                   float32[:, ::1] d_cost_d_mesh,
                                   uint32[:] idx,
                                   float32[:] rest_lengths,
                                   float32 all_weight,
                                   float32 sigma):
    cdef:
        int i
        float32 px, py, qx, qy
        float32 r, h
        float32 dr_dx, dr_dy, dh_dx, dh_dy
        float32 cost

    cost = 0
    with nogil:
        with gil:
            print "in"
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

            # update derivs
            d_cost_d_mesh[i, 0] += dh_dx
            d_cost_d_mesh[i, 1] += dh_dy
            d_cost_d_mesh[idx[i], 0] -= dh_dx
            d_cost_d_mesh[idx[i], 1] -= dh_dy

    print "out"
    return cost


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
