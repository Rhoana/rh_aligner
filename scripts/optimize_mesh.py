import sys
import json
import glob
import os.path
import time
import numpy as np
from numpy.linalg import lstsq
import cPickle as pickle
from collections import defaultdict
import cPickle
from scipy.spatial import Delaunay
#from progressbar import ProgressBar, ETA, Bar, Counter

import pyximport
pyximport.install()
import mesh_derivs

FLOAT_TYPE = np.float64


sys.setrecursionlimit(10000)  # for grad

class MeshParser(object):
    def __init__(self, mesh_file, multiplier=100):
        # load the mesh
        self.mesh = json.load(open(mesh_file))
        self.pts = np.array([(p["x"], p["y"]) for p in self.mesh["points"]], dtype=FLOAT_TYPE)
        self.rowcols = np.array([(p["row"], p["col"]) for p in self.mesh["points"]])
        self.layer_scale = float(self.mesh["layerScale"])

        print("# points in base mesh {}".format(self.pts.shape[0]))

        self.multiplier = multiplier
        # seed rowcolidx to match mesh_pts array
        self._rowcolidx = {}
        for idx, pt in enumerate(self.pts):
            self._rowcolidx[int(pt[0] * multiplier), int(pt[1] * multiplier)] = idx

        # for neighbor searching and internal mesh
        self.triangulation = Delaunay(self.pts)

    def rowcolidx(self, xy):
        return self._rowcolidx[int(xy[0] * self.multiplier), int(xy[1] * self.multiplier)]

    def query_internal(self):
        simplices = self.triangulation.simplices
        # find unique edges
        edge_indices = np.vstack((simplices[:, :2],
                                  simplices[:, 1:],
                                  simplices[:, [0, 2]]))
        edge_indices = np.vstack({tuple(sorted(tuple(row))) for row in edge_indices})
        # mesh.pts[edge_indices, :].shape =(#edges, #pts-per-edge, #values-per-pt)
        edge_lengths = np.sqrt((np.diff(self.pts[edge_indices], axis=1) ** 2).sum(axis=2)).ravel()
        triangles_as_pts = self.pts[simplices]
        triangle_areas = 0.5 * np.cross(triangles_as_pts[:, 2, :] - triangles_as_pts[:, 0, :],
                                        triangles_as_pts[:, 1, :] - triangles_as_pts[:, 0, :])
        return edge_indices, edge_lengths, simplices, triangle_areas

    def query_cross_barycentrics(self, points):
        """Returns the mesh indices that surround a point, and the barycentric weights of those points"""
        simplex_indices = self.triangulation.find_simplex(points)
        assert not np.any(simplex_indices == -1)
        # http://codereview.stackexchange.com/questions/41024/faster-computation-of-barycentric-coordinates-for-many-points
        X = self.triangulation.transform[simplex_indices, :2]
        Y = points - self.triangulation.transform[simplex_indices, 2]
        b = np.einsum('ijk,ik->ij', X, Y)
        bcoords = np.c_[b, 1 - b.sum(axis=1)]
        return simplex_indices, bcoords

def load_matches(matches_files, mesh, url_to_layerid):
    for midx, mf in enumerate((matches_files)):
        for m in json.load(open(mf)):
            if not m['shouldConnect']:
                continue
            # parse matches file, and get p1's mesh x and y points
            orig_p1s = np.array([(pair["p1"]["l"][0], pair["p1"]["l"][1]) for pair in m["correspondencePointPairs"]], dtype=FLOAT_TYPE)
            p1_rc_indices = [mesh.rowcolidx(p1) for p1 in orig_p1s]
            p2_locs = np.array([pair["p2"]["l"] for pair in m["correspondencePointPairs"]])
            p2_locs[p2_locs < 0] = 0.01
            surround_simplices, surround_weights = mesh.query_cross_barycentrics(p2_locs)
            yield (url_to_layerid[m["url1"]], url_to_layerid[m["url2"]],
                   np.array(p1_rc_indices, dtype=np.uint32),
                   np.array(surround_simplices, dtype=np.uint32),
                   surround_weights)

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

def mean_offset(all_pairs, all_mesh_pts, bary_indices, bary_weights, lo, hi):
    num_pts = all_mesh_pts.shape[1]
    means = []
    for id1, id2, baryoff1, baryoff2 in all_pairs:
        if id1 >= hi or id2 >= hi:
            continue
        if id1 < lo and id2 < lo:
            continue
        if abs(int(id1) - int(id2)) == 1:
            mesh1 = all_mesh_pts[id1, ...]
            mesh2 = all_mesh_pts[id2, ...]
            bindices = bary_indices[baryoff1:(baryoff1 + num_pts), ...]
            mesh_1_matches = mesh2[bindices, ...]
            mesh_1_matches *= bary_weights[baryoff1:(baryoff1 + num_pts), ..., np.newaxis]
            delta = (mesh1 - mesh_1_matches.sum(axis=1)) ** 2
            means.append(np.sqrt(delta.sum(axis=1)[bindices[:, 0] != -1]).mean())

            bindices = bary_indices[baryoff2:(baryoff2 + num_pts), ...]
            mesh_2_matches = mesh1[bindices, ...]
            mesh_2_matches *= bary_weights[baryoff2:(baryoff2 + num_pts), ..., np.newaxis]
            delta = (mesh2 - mesh_2_matches.sum(axis=1)) ** 2
            means.append(np.median(np.sqrt(delta.sum(axis=1)[bindices[:, 0] != -1])))
    return np.mean(means)


def optimize_meshes(mesh_file, matches_files, url_to_layerid, conf_dict={}):
    # set default values
    cross_slice_weight = conf_dict.get("cross_slice_weight", 1.0)
    cross_slice_winsor = conf_dict.get("cross_slice_winsor", 1000)
    intra_slice_weight = conf_dict.get("intra_slice_weight", 1.0 / 6)
    intra_slice_winsor = conf_dict.get("intra_slice_winsor", 200)
    intra_slice_weight = 1.0 / 6

    block_size = conf_dict.get("block_size", 35)
    block_step = conf_dict.get("block_step", 25)
    rigid_iterations = conf_dict.get("rigid_iterations", 50)
    min_iterations = conf_dict.get("min_iterations", 200)
    max_iterations = conf_dict.get("max_iterations", 2000)
    num_threads = conf_dict.get("optimization_threads", 4)

    # Load the mesh
    mesh = MeshParser(mesh_file)
    num_pts = mesh.pts.shape[0]

    # Build internal structural mesh
    edge_indices, edge_lengths, face_indices, face_areas = mesh.query_internal()

    # Adjust winsor values according to layer scale
    cross_slice_winsor = cross_slice_winsor * mesh.layer_scale
    intra_slice_winsor = intra_slice_winsor * mesh.layer_scale

    # load the slice-to-slice matches
    cross_links = sorted(load_matches(matches_files, mesh, url_to_layerid))

    # find all the slices represented
    present_slices = sorted(list(set(k[0] for k in cross_links) | set(k[1] for k in cross_links)))
    num_meshes = len(present_slices)

    # build mesh array for all meshes
    all_mesh_pts = np.concatenate([mesh.pts[np.newaxis, ...]] * len(present_slices), axis=0)
    mesh_pt_offsets = dict(zip(present_slices, np.arange(len(present_slices))))

    # concatenate the simplicial indices and weights for cross slice matches
    src_indices = np.concatenate([indices for (_, _, indices, _, _) in cross_links])
    tri_indices, tri_weights = zip(*[(sidx, sweight) for (_, _, _, sidx, sweight) in cross_links])
    tri_offsets = np.cumsum([0] + [si.shape[0] for si in tri_indices]).astype(np.uint32)
    tri_indices = np.concatenate(tri_indices)
    print tri_indices.shape
    tri_weights = np.vstack(tri_weights)

    # build the list of all pairs with their offsets
    all_pairs = np.vstack([(mesh_pt_offsets[id1], mesh_pt_offsets[id2])
                           for id1, id2, _, _, _ in cross_links]).astype(np.uint32)
    between_mesh_weights = np.array([cross_slice_weight / float(abs(int(id1) - int(id2)))
                                     for id1, id2 in all_pairs],
                                    dtype=FLOAT_TYPE)


    oldtick = time.time()

    #pbar = ProgressBar(widgets=[Bar(), ETA()])

    for block_lo in (range(0, max(1, num_meshes - block_size + 1), block_step)):
        print
        block_hi = min(block_lo + block_size, num_meshes)

        gradient = np.empty_like(all_mesh_pts[block_lo:block_hi, ...])
        cost = mesh_derivs.all_derivs(all_mesh_pts,
                                      gradient,
                                      mesh.triangulation.simplices.astype(np.uint32),
                                      all_pairs,
                                      between_mesh_weights,
                                      src_indices,
                                      tri_indices,
                                      tri_weights,
                                      tri_offsets,
                                      edge_indices.astype(np.uint32),
                                      edge_lengths,
                                      intra_slice_weight,
                                      cross_slice_winsor,
                                      intra_slice_winsor,
                                      block_lo, block_hi,
                                      num_threads)

        m_o = mean_offset(all_pairs, all_mesh_pts, bary_indices, bary_weights, block_lo, block_hi)
        print "BEFORE", "C", cost, "MO", m_o

        # first, do some rigid alignment, slice at a time
        for single_slice_idx in range(block_lo, block_hi):
            gradient = np.empty_like(all_mesh_pts[single_slice_idx:(single_slice_idx + 1), ...])
            step_size = 0.1
            prev_cost = np.inf
            for iter in range(rigid_iterations):
                cost = mesh_derivs.all_derivs(all_mesh_pts,
                                              gradient,
                                              neighbor_indices,
                                              dists,
                                              bary_indices,
                                              bary_weights,
                                              between_mesh_weights,
                                              intra_slice_weight,
                                              cross_slice_winsor,
                                              intra_slice_winsor,
                                              all_pairs,
                                              single_slice_idx, single_slice_idx + 1,
                                              1)
                if cost < prev_cost:
                    lin_grad = linearize_grad(all_mesh_pts[single_slice_idx, ...],
                                              gradient[0, ...])
                    all_mesh_pts[single_slice_idx, ...] -= step_size * lin_grad
                    step_size = min(1.0, step_size * 1.1)
                else:
                    all_mesh_pts[single_slice_idx, ...] += step_size * lin_grad
                    step_size = 0.5 * step_size
                prev_cost = cost

        gradient_with_momentum = 0
        stepsize = 0.1
        prev_cost = np.inf
        gradient = np.empty_like(all_mesh_pts[block_lo:block_hi, ...])

        for iter in range(max_iterations):
            cost = mesh_derivs.all_derivs(all_mesh_pts,
                                          gradient,
                                          neighbor_indices,
                                          dists,
                                          bary_indices,
                                          bary_weights,
                                          between_mesh_weights,
                                          intra_slice_weight,
                                          cross_slice_winsor,
                                          intra_slice_winsor,
                                          all_pairs,
                                          block_lo, block_hi,
                                          4)

            if iter % 100 == 0:
                m_o = mean_offset(all_pairs, all_mesh_pts, bary_indices, bary_weights, block_lo, block_hi)
                print iter, "SL:", block_lo, block_hi, num_meshes, "COST:", cost, "MO:", m_o, "SZ:", stepsize, "T:", time.time() - oldtick
                oldtick = time.time()
                if (iter >= min_iterations) and ((m_o < .75) or (stepsize < 1e-20)):
                    break

                distortions = []
                for single_slice_idx in range(block_lo, block_hi):
                    dmax = []
                    for whichn, ni in enumerate(neighbor_indices.T):
                        deltas = np.sqrt(((all_mesh_pts[single_slice_idx, ...] - all_mesh_pts[single_slice_idx, ni, ...]) ** 2).sum(axis=1))
                        deltas = abs(deltas - dists[:, whichn])
                        dmax.append(int(deltas.max()))
                    distortions.append(max(dmax))
                print "DIS", distortions
                worst = np.argmax(distortions)

            # relaxation of the mesh
            # initially, mesh is held rigid (all points transform together).
            # mesh is allowed to deform as iterations progress.
            relaxation_end = int(min_iterations)
            if iter < relaxation_end:
                # for each mesh, compute a linear fit to the gradient
                for meshidx in range(block_lo, block_hi):
                    gidx = meshidx - block_lo
                    linearized = linearize_grad(all_mesh_pts[meshidx, ...], gradient[gidx, ...])
                    gradient[gidx, ...] = blend(linearized, gradient[gidx, ...], iter / float(relaxation_end))

            # step size adjustment
            if cost <= prev_cost:
                stepsize *= 1.1
                if stepsize > 1.0:
                    stepsize = 1.0
                # update with new gradients
                gradient_with_momentum = (gradient + 0.5 * gradient_with_momentum)
                all_mesh_pts[block_lo:block_hi, ...] -= stepsize * gradient_with_momentum
                prev_cost = cost
            else:  # we took a bad step: undo it, scale down stepsize, and start over
                all_mesh_pts[block_lo:block_hi, ...] += stepsize * gradient_with_momentum
                stepsize *= 0.5
                gradient_with_momentum = 0.0
                prev_cost = np.inf

    # Prepare per-layer output
    out_positions = {}

    for url, layerid in url_to_layerid.iteritems():
        if layerid in present_slices:
            meshidx = mesh_pt_offsets[layerid]
            out_positions[url] = [(mesh.pts / mesh.layer_scale).tolist(),
                                  (all_mesh_pts[meshidx, :] / mesh.layer_scale).tolist()]
            cPickle.dump([url, out_positions[url]], open("newpos{}.pickle".format(meshidx), "w"))
        else:
            out_positions[url] = [(mesh.pts / mesh.layer_scale).tolist(),
                                  (mesh.pts / mesh.layer_scale).tolist()]

    return out_positions


if __name__ == '__main__':
    mesh_file = sys.argv[1]
    matches_files = glob.glob(os.path.join(sys.argv[2], '*W02_sec0[012]*W02_sec0[012]*.json'))
    print("Found {} match files".format(len(matches_files)))
    url_to_layerid = None
    new_positions = optimize_meshes(mesh_file, matches_files, url_to_layerid)

    out_file = sys.argv[3]
    json.dump(new_positions, open(out_file, "w"), indent=1)
