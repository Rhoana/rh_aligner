import sys
import json
import os
import os.path
import numpy as np
import cPickle
from scipy.spatial import Delaunay

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
        bcoords = np.c_[b, 1 - b.sum(axis=1)]
        return self.triangulation.simplices[simplex_indices].astype(np.uint32), bcoords

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


def mean_offset(all_mesh_pts,
                all_pairs,
                src_indices,
                pt_indices,
                pt_weights,
                pair_offsets,
                lo, hi):
    means = []
    all_offsets = np.zeros((hi - lo, hi - lo))
    for idx, (id1, id2) in enumerate(all_pairs):
        if id1 >= hi or id2 >= hi:
            continue
        if id1 < lo and id2 < lo:
            continue
        offset = pair_offsets[idx]
        next_offset = pair_offsets[idx + 1]
        src_pts = all_mesh_pts[id1, src_indices[offset:next_offset]]  # Nx2
        dest_pts = all_mesh_pts[id2, pt_indices[offset:next_offset]]  # Nx3x2
        dest_weights = pt_weights[offset:next_offset]  # Nx3
        dest_pts = np.einsum('ij,ijk->ik', dest_weights, dest_pts)
        lens = np.sqrt(((src_pts - dest_pts) ** 2).sum(axis=1))
        all_offsets[id1 - lo, id2 - lo] = np.median(lens)
        if abs(int(id1) - int(id2)) == 1:
            means.append(np.median(lens))
    return np.mean(means)


def optimize_meshes(meshes, links, conf_dict={}):
    # set default values
    cross_slice_weight = conf_dict.get("cross_slice_weight", 1.0)
    cross_slice_winsor = conf_dict.get("cross_slice_winsor", 1000)
    intra_slice_weight = conf_dict.get("intra_slice_weight", 1.0)
    intra_slice_winsor = conf_dict.get("intra_slice_winsor", 200)
    intra_slice_weight = 3.0

    # block_size = conf_dict.get("block_size", 35)
    # block_step = conf_dict.get("block_step", 25)
    rigid_iterations = conf_dict.get("rigid_iterations", 50)
    # min_iterations = conf_dict.get("min_iterations", 200)
    max_iterations = conf_dict.get("max_iterations", 5000)
    # mean_offset_threshold = conf_dict.get("mean_offset_threshold", 5)
    # num_threads = conf_dict.get("optimization_threads", 8)

    stepsize = 0.0001
    momentum = 0.5
    prev_cost = np.inf
    gradients_with_momentum = {ts: 0.0 for ts in meshes}
    old_pts = None

    # Build internal structural mesh
    # (edge_indices, edge_lengths, face_indices, face_areas)
    structural_meshes = {ts: mesh.internal_structural_mesh() for ts, mesh in meshes.iteritems()}

    # pre-optimization: rigid alignment of slices
    sorted_slices = sorted(meshes.keys())
    for active_ts in sorted_slices[1:]:
        for iter in range(rigid_iterations):
            cost = 0.0

            active_gradient = np.zeros_like(meshes[active_ts].pts)
            ignore_gradient = np.empty_like(active_gradient)

            for (ts1, ts2), ((idx1, w1), (idx2, w2)) in links.iteritems():
                if active_ts in (ts1, ts2) and (ts1 <= active_ts) and (ts2 <= active_ts):
                    if ts1 != active_ts:
                        ts1, ts2 = ts2, ts1
                        idx1, idx2 = idx2, idx1
                        w1, w2 = w2, w1
                    cost += mesh_derivs.external_grad(meshes[ts1].pts, meshes[ts2].pts,
                                                      active_gradient, ignore_gradient,
                                                      idx1, w1,
                                                      idx2, w2,
                                                      cross_slice_weight, cross_slice_winsor)

            g = linearize_grad(meshes[ts1].pts, active_gradient)
            mesh[ts1].pts -= 0.1 * g
            print iter, cost, active_ts

    for iter in range(max_iterations):
        cost = 0.0

        gradients = {ts: np.zeros_like(mesh.pts) for ts, mesh in meshes.iteritems()}

        for ts in meshes:
            cost += mesh_derivs.internal_grad(meshes[ts].pts, gradients[ts],
                                              *((structural_meshes[ts]) +
                                                (intra_slice_weight, intra_slice_winsor)))
        for (ts1, ts2), ((idx1, w1), (idx2, w2)) in links.iteritems():
            cost += mesh_derivs.external_grad(meshes[ts1].pts, meshes[ts2].pts,
                                              gradients[ts1], gradients[ts2],
                                              idx1, w1,
                                              idx2, w2,
                                              cross_slice_weight, cross_slice_winsor)

        if cost < prev_cost and not np.isinf(cost):
            stepsize *= 1.1
            if stepsize > 1.0:
                stepsize = 1.0
            # update with new gradients
            for ts in gradients_with_momentum:
                gradients_with_momentum[ts] = gradients[ts] + momentum * gradients_with_momentum[ts]
            old_pts = {ts: m.pts.copy() for ts, m in meshes.iteritems()}
            for ts in meshes:
                meshes[ts].pts -= stepsize * gradients_with_momentum[ts]
            print("Good step: cost {}  stepsize {}".format(cost, stepsize))
        else:  # we took a bad step: undo it, scale down stepsize, and start over
            for ts in meshes:
                meshes[ts].pts = old_pts[ts]
            stepsize *= 0.5
            gradients_with_momentum = {ts: 0 for ts in meshes}
            prev_cost = np.inf
            print("Bad step: stepsize {}".format(stepsize))

        for ts in meshes:
            assert not np.any(~ np.isfinite(meshes[ts].pts))

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
    for match_file in sys.argv[1:]:
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
