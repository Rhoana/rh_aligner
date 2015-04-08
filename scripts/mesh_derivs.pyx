from __future__ import division
import numpy as np
cimport numpy
from libc.math cimport sin, cos, acos, exp, sqrt, fabs, M_PI

ctypedef numpy.float64_t float32

cdef:
    float32 small_value = 0.001


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
