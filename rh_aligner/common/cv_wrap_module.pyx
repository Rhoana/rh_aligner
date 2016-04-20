#cython: boundscheck=False
#cython: wraparound=False


# cdefine the signature of our c++ function
cdef extern from "opencv2/core/core.hpp" namespace "cv":
    cpdef void setNumThreads(int nthreads)

cdef extern from "opencv2/core/core.hpp" namespace "cv":
    cpdef int getNumThreads()

def setThreads(int threads_num):
    setNumThreads(threads_num)

def getThreads():
    return getNumThreads()
