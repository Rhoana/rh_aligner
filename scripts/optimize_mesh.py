import sys
import json
import os
import os.path
import numpy as np
import cPickle
from scipy.spatial import Delaunay
import pylab
from matplotlib import collections as mc

import pyximport
pyximport.install()
import mesh_derivs

FLOAT_TYPE = np.float64


sys.setrecursionlimit(10000)  # for grad


class Mesh(object):
    def __init__(self, points):
        # load the mesh
        self.pts = np.array(points, dtype=FLOAT_TYPE).reshape((-1, 2)).copy()
        center = self.pts.mean(axis=0)
        self.pts -= center
        self.pts *= 1.1
        self.pts += center

        print("# points in base mesh {}".format(self.pts.shape[0]))

        # for neighbor searching and internal mesh
        self.triangulation = Delaunay(self.pts)

    def internal_structural_mesh(self):
        simplices = self.triangulation.simplices.astype(np.uint32)
        # find unique edges
        edge_indices = np.vstack((simplices[:, :2],
                                  simplices[:, 1:],
                                  simplices[:, [0, 2]]))
        edge_indices = np.vstack({tuple(sorted(tuple(row))) for row in edge_indices}).astype(np.uint32)
        # mesh.pts[edge_indices, :].shape =(#edges, #pts-per-edge, #values-per-pt)
        edge_lengths = np.sqrt((np.diff(self.pts[edge_indices], axis=1) ** 2).sum(axis=2)).ravel()
        triangles_as_pts = self.pts[simplices]
        triangle_areas = 0.5 * np.cross(triangles_as_pts[:, 2, :] - triangles_as_pts[:, 0, :],
                                        triangles_as_pts[:, 1, :] - triangles_as_pts[:, 0, :])
        return edge_indices, edge_lengths, simplices, triangle_areas

    def query_barycentrics(self, points):
        """Returns the mesh indices that surround a point, and the barycentric weights of those points"""
        p = points.copy()
        p[p < 0] = 0.01
        simplex_indices = self.triangulation.find_simplex(p)
        assert not np.any(simplex_indices == -1)

        # http://codereview.stackexchange.com/questions/41024/faster-computation-of-barycentric-coordinates-for-many-points
        X = self.triangulation.transform[simplex_indices, :2]
        Y = points - self.triangulation.transform[simplex_indices, 2]
        b = np.einsum('ijk,ik->ij', X, Y)
        pt_indices = self.triangulation.simplices[simplex_indices].astype(np.uint32)
        barys = np.c_[b, 1 - b.sum(axis=1)]

        pts_test = np.einsum('ijk,ij->ik', self.pts[pt_indices], barys)

        return self.triangulation.simplices[simplex_indices].astype(np.uint32), barys

def linearize_grad(positions, gradients):
    '''perform a least-squares fit, then return the values from that fit'''
    positions = np.hstack((positions, np.ones((positions.shape[0], 1))))
    XTX = np.dot(positions.T, positions)
    XTY = np.dot(positions.T, gradients)
    Beta = np.dot(np.linalg.inv(XTX), XTY)
    return np.dot(positions, Beta)

def blend(a, b, t):
    '''at t=0, return a, at t=1, return b'''
    return a + (b - a) * t


def mean_offsets(meshes, links, stop_at_ts, plot=False):
    means = []
    for (ts1, ts2), ((idx1, w1), (idx2, w2)) in links.iteritems():
        if ts1 > stop_at_ts or ts2 > stop_at_ts:
            continue
        pts1 = np.einsum('ijk,ij->ik', meshes[ts1].pts[idx1], w1)
        pts2 = np.einsum('ijk,ij->ik', meshes[ts2].pts[idx2], w2)
        if plot:
            lines = [[p1, p2] for p1, p2 in zip(pts1, pts2)]

            lc = mc.LineCollection(lines)
            pylab.figure()
            pylab.title(ts1 + ' ' + ts2)
            pylab.gca().add_collection(lc)
            pylab.scatter(pts1[:, 0], pts1[:, 1])
            pylab.gca().autoscale()
        lens = np.sqrt(((pts1 - pts2) ** 2).sum(axis=1))
        means.append(np.median(lens))
    return np.mean(means)

def align(pts1, pts2):
    # find R,T that bring pts1 to pts2

    # convert to column vectors
    pts1 = pts1.T
    pts2 = pts2.T
    
    m1 = pts1.mean(axis=1, keepdims=True)
    m2 = pts2.mean(axis=1, keepdims=True)
    U, S, VT = np.linalg.svd(np.dot((pts1 - m1), (pts2 - m2).T))
    R = np.dot(VT.T, U.T)
    T = - np.dot(R, m1) + m2

    # convert to form used for left-multiplying row vectors
    return R.T, T.T

#def affine_align(pts1, pts2, regularization=0):
#    # find R,T that bring pts1 to pts2
#
#    # convert to column vectors
#    pts1 = pts1.T
#    pts2 = pts2.T
#    
#    m1 = pts1.mean(axis=1, keepdims=True)
#    m2 = pts2.mean(axis=1, keepdims=True)
#    U, S, VT = np.linalg.svd(np.dot((pts1 - m1), (pts2 - m2).T) + np.dot(regularization, np.eye((pts1.shape[0], pts2.shape[0]))))
#    R = np.dot(VT.T, U.T)
#    T = - np.dot(R, m1) + m2
#
#    # convert to form used for left-multiplying row vectors
#    return R.T, T.T


def Haffine_from_points(fp, tp):
    """Find H, affine transformation, s.t. tp is affine transformation of fp.
       Taken from 'Programming Computer Vision with Python: Tools and algorithms for analyzing images'
    """

    assert(fp.shape == tp.shape)

    fp = fp.T
    tp = tp.T

    fp = np.vstack([fp[0], fp[1], np.ones(len(fp[0]))])
    tp = np.vstack([tp[0], tp[1], np.ones(len(tp[0]))])

    # condition points
    # --- from points ---
    m = fp.mean(axis=1, keepdims=True)
    maxstd = np.max(np.std(fp, axis=1)) + 1e-9
    C1 = np.diag([1.0/maxstd, 1.0/maxstd, 1.0])
    C1[0][2] = -m[0] / maxstd
    C1[1][2] = -m[1] / maxstd
    fp_cond = np.dot(C1, fp)

    # --- to points ---
    m = tp.mean(axis=1, keepdims=True)
    C2 = C1.copy() # must use same scaling for both point sets
    C2[0][2] = -m[0] / maxstd
    C2[1][2] = -m[1] / maxstd
    tp_cond = np.dot(C2, tp)

    # conditioned points have mean zero, so translation is zero
    A = np.concatenate((fp_cond[:2], tp_cond[:2]), axis=0)
    U,S,V = np.linalg.svd(A.T)

    # create B and C matrices as Hartley-Zisserman (2:nd ed) p 130.
    tmp = V[:2].T
    B = tmp[:2]
    C = tmp[2:4]

    tmp2 = np.concatenate((np.dot(C, np.linalg.pinv(B)), np.zeros((2, 1))), axis=1)
    H = np.vstack((tmp2, [0, 0, 1]))

    # decondition
    H = np.dot(np.linalg.inv(C2), np.dot(H, C1))

    return H / H[2][2]


def optimize_meshes(meshes, links, conf_dict={}):
    # set default values
    cross_slice_weight = conf_dict.get("cross_slice_weight", 1.0)
    cross_slice_winsor = conf_dict.get("cross_slice_winsor", 20)
    intra_slice_weight = conf_dict.get("intra_slice_weight", 1.0)
    intra_slice_winsor = conf_dict.get("intra_slice_winsor", 200)
    intra_slice_weight = 3.0

    # block_size = conf_dict.get("block_size", 35)
    # block_step = conf_dict.get("block_step", 25)
    rigid_iterations = conf_dict.get("rigid_iterations", 5000)
    # min_iterations = conf_dict.get("min_iterations", 200)
    #max_iterations = conf_dict.get("max_iterations", 500000)
    max_iterations = conf_dict.get("max_iterations", 5000)
    # mean_offset_threshold = conf_dict.get("mean_offset_threshold", 5)
    # num_threads = conf_dict.get("optimization_threads", 8)

    # Build internal structural mesh
    # (edge_indices, edge_lengths, face_indices, face_areas)
    structural_meshes = {ts: mesh.internal_structural_mesh() for ts, mesh in meshes.iteritems()}

    # pre-optimization: rigid alignment of slices
    sorted_slices = sorted(meshes.keys())
    for active_ts in sorted_slices[1:]:
        print("Before: {}".format(mean_offsets(meshes, links, active_ts, plot=False)))
        # rot = 0
        # trans = 0
        count = 0
        all_H = np.zeros((3,3))
        for (ts1, ts2), ((idx1, w1), (idx2, w2)) in links.iteritems():
            if active_ts in (ts1, ts2) and (ts1 <= active_ts) and (ts2 <= active_ts):
                pts1 = np.einsum('ijk,ij->ik', meshes[ts1].pts[idx1], w1)
                pts2 = np.einsum('ijk,ij->ik', meshes[ts2].pts[idx2], w2)
                if ts1 == active_ts:
                    H = Haffine_from_points(pts1, pts2)
                else:
                    H = Haffine_from_points(pts2, pts1)
                all_H += H
                count += 1
        #meshes[active_ts].pts = np.dot(meshes[active_ts].pts, rot / count) + (trans / count)
        # normalize the affine transformation
        all_H = all_H * (1.0 / count)
        # transform the points
        print all_H[:2,:2]
        print all_H[:2,2]
        print all_H[:2]
        rot = all_H[:2, :2]
        trans = all_H[:2, 2]
        meshes[active_ts].pts = np.dot(meshes[active_ts].pts, rot.T) + trans
        # meshes[active_ts].pts = np.array([np.dot(all_H[:2], pt) for pt in meshes[active_ts].pts])
        print("After: {}\n".format(mean_offsets(meshes, links, active_ts, plot=False)))

#    for active_ts in sorted_slices[1:]:
#        print "\nPreopt", active_ts
#        prev_cost = np.inf
#        stepsize = 1e-7
#        for iter in range(300):
#            active_gradient = np.zeros_like(meshes[active_ts].pts)
#            cost = mesh_derivs.internal_grad(meshes[active_ts].pts, active_gradient,
#                                             *((structural_meshes[active_ts]) +
#                                               (intra_slice_weight, intra_slice_winsor)))
#            for (ts1, ts2), ((idx1, w1), (idx2, w2)) in links.iteritems():
#                if active_ts in (ts1, ts2) and (ts1 <= active_ts) and (ts2 <= active_ts):
#                    if ts1 == active_ts:
#                        ts1_gradient = active_gradient
#                        ts2_gradient = np.empty_like(meshes[ts2].pts)
#                    else:
#                        ts1_gradient = np.empty_like(meshes[ts1].pts)
#                        ts2_gradient = active_gradient
#
#                    pts1 = np.einsum('ijk,ij->ik', meshes[ts1].pts[idx1], w1)
#                    pts2 = np.einsum('ijk,ij->ik', meshes[ts2].pts[idx2], w2)
#
#                    D = mesh_derivs.external_grad(meshes[ts1].pts, meshes[ts2].pts,
#                                                  ts1_gradient, ts2_gradient,
#                                                  idx1, w1,
#                                                  idx2, w2,
#                                                  cross_slice_weight, cross_slice_winsor)
#                    cost += D
#
#            if cost < prev_cost:
#                prev_cost = cost
#                stepsize *= 1.1
#                g = linearize_grad(meshes[active_ts].pts, active_gradient)
#                prev_pts = meshes[active_ts].pts.copy()
#                meshes[active_ts].pts -= stepsize * g
#            else:
#                stepsize *= 0.5
#                meshes[active_ts].pts = prev_pts
#            print "    ", iter, cost, stepsize, active_ts, mean_offsets(meshes, links, active_ts)
#    print("Preoptimization done")
    print("After preopt: {}\n".format(mean_offsets(meshes, links, sorted_slices[-1], plot=False)))

    stepsize = 1e-8 #0.0001
    momentum = 0.5
    prev_cost = np.inf
    gradients_with_momentum = {ts: 0.0 for ts in meshes}
    old_pts = None


    for iter in range(max_iterations):
        cost = 0.0

        gradients = {ts: np.zeros_like(mesh.pts) for ts, mesh in meshes.iteritems()}

        internal_cost = 0.0
        external_cost = 0.0
        for ts in meshes:
            internal_cost += mesh_derivs.internal_grad(meshes[ts].pts, gradients[ts],
                                              *((structural_meshes[ts]) +
                                                (intra_slice_weight, intra_slice_winsor)))
        for (ts1, ts2), ((idx1, w1), (idx2, w2)) in links.iteritems():
            external_cost += mesh_derivs.external_grad(meshes[ts1].pts, meshes[ts2].pts,
                                              gradients[ts1], gradients[ts2],
                                              idx1, w1,
                                              idx2, w2,
                                              cross_slice_weight, cross_slice_winsor)
        if iter % 100 == 0:
            print("Internal cost: {}, External cost: {}".format(internal_cost, external_cost)) 
        cost = internal_cost + external_cost

        if cost < prev_cost and not np.isinf(cost):
            prev_cost = cost
            stepsize *= 1.1
            if stepsize > 1.0:
                stepsize = 1.0
            # update with new gradients
            for ts in gradients_with_momentum:
                gradients_with_momentum[ts] = gradients[ts] + momentum * gradients_with_momentum[ts]
            old_pts = {ts: m.pts.copy() for ts, m in meshes.iteritems()}
            for ts in meshes:
                meshes[ts].pts -= stepsize * gradients_with_momentum[ts]
            # if iter % 500 == 0:
            #     print("{} Good step: cost {}  stepsize {}".format(iter, cost, stepsize))
        else:  # we took a bad step: undo it, scale down stepsize, and start over
            for ts in meshes:
                meshes[ts].pts = old_pts[ts]
            stepsize *= 0.5
            gradients_with_momentum = {ts: 0 for ts in meshes}
            # if iter % 500 == 0:
            #     print("{} Bad step: stepsize {}".format(iter, stepsize))
        if iter % 100 == 0:
            print("iter {}: C: {}, MO: {}, S: {}".format(iter, cost, mean_offsets(meshes, links, sorted_slices[-1], plot=False), stepsize))

        for ts in meshes:
            assert not np.any(~ np.isfinite(meshes[ts].pts))
    print("last MO: {}\n".format(mean_offsets(meshes, links, sorted_slices[-1], plot=True)))

    # Prepare per-layer output
    out_positions = {}

    for tsfile, layerid in tsfile_to_layerid.iteritems():
        if layerid in present_slices:
            meshidx = mesh_pt_offsets[layerid]
            out_positions[tsfile] = [(mesh.pts / mesh.layer_scale).tolist(),
                                     (all_mesh_pts[meshidx, :] / mesh.layer_scale).tolist()]
            if "DUMP_LOCATIONS" in os.environ:
                cPickle.dump([tsfile, out_positions[tsfile]], open("newpos{}.pickle".format(meshidx), "w"))
        else:
            out_positions[tsfile] = [(mesh.pts / mesh.layer_scale).tolist(),
                                     (mesh.pts / mesh.layer_scale).tolist()]

    return out_positions


if __name__ == '__main__':
    meshes = {}

    # extract meshes
    for match_file in sys.argv[1:-1]:
        data = json.load(open(match_file))
        if not data["tilespec1"] in meshes:
            meshes[data["tilespec1"]] = Mesh(data["mesh"])

    links = {}
    # extract links
    for match_file in sys.argv[1:-1]:
        print match_file
        data = json.load(open(match_file))
        ts1 = data["tilespec1"]
        ts2 = data["tilespec2"]
        pts1 = np.array([p["point1"] for p in data["pointmatches"]])
        pts2 = np.array([p["point2"] for p in data["pointmatches"]])
        if len(pts1) > 0:
            links[ts1, ts2] = (meshes[ts1].query_barycentrics(pts1),
                               meshes[ts2].query_barycentrics(pts2))
        if not data["tilespec1"] in meshes:
            meshes[data["tilespec1"]] = Mesh(data["mesh"])

    new_positions = optimize_meshes(meshes, links)

    out_file = sys.argv[-1]
    json.dump(new_positions, open(out_file, "w"), indent=1)
