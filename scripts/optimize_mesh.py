import sys
import json
import glob
import os
import os.path
import time
import numpy as np
import cPickle
from progressbar import ProgressBar, ETA, Bar, Counter
import h5py
from scipy.spatial import Delaunay

import pyximport
pyximport.install()
import mesh_derivs

FLOAT_TYPE = np.float64


sys.setrecursionlimit(10000)  # for grad

class MeshParser(object):
    def __init__(self, mesh_file, multiplier=10):
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
        p = points.copy()
        p[p < 0] = 0.01
        simplex_indices = self.triangulation.find_simplex(p)
        assert not np.any(simplex_indices == -1)
        
        # http://codereview.stackexchange.com/questions/41024/faster-computation-of-barycentric-coordinates-for-many-points
        X = self.triangulation.transform[simplex_indices, :2]
        Y = points - self.triangulation.transform[simplex_indices, 2]
        b = np.einsum('ijk,ik->ij', X, Y)
        bcoords = np.c_[b, 1 - b.sum(axis=1)]
        return self.triangulation.simplices[simplex_indices], bcoords

def load_matches_hdf5(matches_files, mesh, tsfile_to_layerid):
    pbar = ProgressBar(widgets=['Loading matches: ', Counter(), ' / ', str(len(matches_files)), " ", Bar(), ETA()])

    for midx, mf in enumerate(pbar(matches_files)):
        with h5py.File(mf, 'r') as m:
            for idx in range(2):
                # if not m['shouldConnect']:
                #    continue
                if not "matches_{}_p1".format(idx) in m.keys():
                    continue

                # parse matches file, and get p1's mesh x and y points
                orig_p1s = m["matches_{}_p1".format(idx)][...]

                # Instead of shouldConnect
                if len(orig_p1s) < 3:
                    continue

                p1_rc_indices = [mesh.rowcolidx(p1) for p1 in orig_p1s]

                p2_locs = m["matches_{}_p2".format(idx)][...]
                surround_points, surround_weights = mesh.query_cross_barycentrics(p2_locs)

                layer1 = tsfile_to_layerid[os.path.basename(str(m["matches{}_url1".format(idx)][...]))]
                layer2 = tsfile_to_layerid[os.path.basename(str(m["matches{}_url2".format(idx)][...]))]

                yield (layer1, layer2,
                       np.array(p1_rc_indices, dtype=np.uint32),
                       np.array(surround_points, dtype=np.uint32),
                       surround_weights)

def load_matches(matches_files, mesh, tsfile_to_layerid):
    pbar = ProgressBar(widgets=['Loading matches: ', Counter(), ' / ', str(len(matches_files)), " ", Bar(), ETA()])

    for midx, mf in enumerate(pbar(matches_files)):
        for m in json.load(open(mf)):
            if not m['shouldConnect']:
                continue
            # parse matches file, and get p1's mesh x and y points
            orig_p1s = np.array([(pair["p1"]["l"][0], pair["p1"]["l"][1]) for pair in m["correspondencePointPairs"]], dtype=FLOAT_TYPE)
            p1_rc_indices = [mesh.rowcolidx(p1) for p1 in orig_p1s]
            p2_locs = np.array([pair["p2"]["l"] for pair in m["correspondencePointPairs"]])
            surround_points, surround_weights = mesh.query_cross_barycentrics(p2_locs)
            yield (tsfile_to_layerid[os.path.basename(m["url1"])],
                   tsfile_to_layerid[os.path.basename(m["url2"])],
                   np.array(p1_rc_indices, dtype=np.uint32),
                   np.array(surround_points, dtype=np.uint32),
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


def mean_offset(all_mesh_pts,
                all_pairs,
                src_indices,
                pt_indices,
                pt_weights,
                pair_offsets,
                lo, hi):
    num_pts = all_mesh_pts.shape[1]
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
    #for r in all_offsets:
    #    print " ".join("{0:.2f}".format(v) if v > 0.01 else "    " for v in r)
    return np.mean(means)


def optimize_meshes(mesh_file, matches_files, tsfile_to_layerid, conf_dict={}):
    # set default values
    cross_slice_weight = conf_dict.get("cross_slice_weight", 1.0)
    cross_slice_winsor = conf_dict.get("cross_slice_winsor", 1000)
    intra_slice_weight = conf_dict.get("intra_slice_weight", 1.0)
    intra_slice_winsor = conf_dict.get("intra_slice_winsor", 200)
    print "OLD WEIGHT", intra_slice_weight
    intra_slice_weight = 3.0
    

    block_size = conf_dict.get("block_size", 35)
    block_step = conf_dict.get("block_step", 25)
    rigid_iterations = conf_dict.get("rigid_iterations", 50)
    min_iterations = conf_dict.get("min_iterations", 200)
    max_iterations = conf_dict.get("max_iterations", 500)
    mean_offset_threshold = conf_dict.get("mean_offset_threshold", 5)
    num_threads = conf_dict.get("optimization_threads", 8)
    

    # Load the mesh
    mesh = MeshParser(mesh_file)
    num_pts = mesh.pts.shape[0]

    # Build internal structural mesh
    edge_indices, edge_lengths, face_indices, face_areas = mesh.query_internal()

    # Adjust winsor values according to layer scale
    cross_slice_winsor = cross_slice_winsor * mesh.layer_scale
    intra_slice_winsor = intra_slice_winsor * mesh.layer_scale

    # load the slice-to-slice matches
    if 'hdf5' in matches_files[0]:
        cross_links = sorted(load_matches_hdf5(matches_files, mesh, tsfile_to_layerid))
    else:
        cross_links = sorted(load_matches(matches_files, mesh, tsfile_to_layerid))

    # find all the slices represented
    present_slices = sorted(list(set(k[0] for k in cross_links) | set(k[1] for k in cross_links)))
    num_meshes = len(present_slices)

    # build mesh array for all meshes
    all_mesh_pts = np.concatenate([mesh.pts[np.newaxis, ...]] * len(present_slices), axis=0)
    mesh_pt_offsets = dict(zip(present_slices, np.arange(len(present_slices))))

    # concatenate the simplicial indices and weights for cross slice matches
    src_indices, dest_indices, dest_weights = zip(*[(sidx, pidx, pweight) for (_, _, sidx, pidx, pweight) in cross_links])
    match_offsets = np.cumsum([0] + [pi.shape[0] for pi in src_indices]).astype(np.uint32)
    src_indices = np.concatenate(src_indices).astype(np.uint32)
    dest_indices = np.vstack(dest_indices)
    dest_weights = np.vstack(dest_weights)

    # build the list of all pairs with their offsets
    all_pairs = np.vstack([(mesh_pt_offsets[id1], mesh_pt_offsets[id2])
                           for id1, id2, _, _, _ in cross_links]).astype(np.uint32)
    between_mesh_weights = np.array([cross_slice_weight / float(abs(int(id1) - int(id2)))
                                     for id1, id2 in all_pairs],
                                    dtype=FLOAT_TYPE)

    # pbar = ProgressBar(widgets=[Bar(), ETA()])

    for block_lo in (range(0, max(1, num_meshes - block_size + 1), block_step)):
        print
        block_hi = min(block_lo + block_size, num_meshes)

        gradient = np.empty_like(all_mesh_pts[block_lo:block_hi, ...])
        cost = mesh_derivs.all_derivs(all_mesh_pts,
                                      gradient,
                                      all_pairs,
                                      between_mesh_weights,
                                      src_indices,
                                      dest_indices,
                                      dest_weights,
                                      match_offsets,
                                      edge_indices.astype(np.uint32),
                                      edge_lengths,
                                      face_indices.astype(np.uint32),
                                      face_areas,
                                      intra_slice_weight,
                                      cross_slice_winsor,
                                      intra_slice_winsor,
                                      block_lo, block_hi,
                                      num_threads)

        m_o = mean_offset(all_mesh_pts,
                          all_pairs,
                          src_indices,
                          dest_indices,
                          dest_weights,
                          match_offsets,
                          block_lo, block_hi)
        print "BEFORE", "C", cost, "MO", m_o
        oldtick = time.time()

        # first, do some rigid alignment, slice at a time
        for single_slice_idx in range(block_lo, block_hi):
            gradient = np.empty_like(all_mesh_pts[single_slice_idx:(single_slice_idx + 1), ...])
            step_size = 0.1
            prev_cost = np.inf
            for iter in range(rigid_iterations):
                cost = mesh_derivs.all_derivs(all_mesh_pts,
                                              gradient,
                                              all_pairs,
                                              between_mesh_weights,
                                              src_indices,
                                              dest_indices,
                                              dest_weights,
                                              match_offsets,
                                              edge_indices.astype(np.uint32),
                                              edge_lengths,
                                              face_indices.astype(np.uint32),
                                              face_areas,
                                              intra_slice_weight,
                                              cross_slice_winsor,
                                              intra_slice_winsor,
                                              single_slice_idx, single_slice_idx + 1,
                                              num_threads)
                if cost < prev_cost:
                    lin_grad = linearize_grad(all_mesh_pts[single_slice_idx, ...],
                                              gradient[0, ...])
                    all_mesh_pts[single_slice_idx, ...] -= step_size * lin_grad
                    step_size = min(1.0, step_size * 1.1)
                else:
                    all_mesh_pts[single_slice_idx, ...] += step_size * lin_grad
                    step_size = 0.5 * step_size
                prev_cost = cost

        cost = mesh_derivs.all_derivs(all_mesh_pts,
                                      gradient,
                                      all_pairs,
                                      between_mesh_weights,
                                      src_indices,
                                      dest_indices,
                                      dest_weights,
                                      match_offsets,
                                      edge_indices.astype(np.uint32),
                                      edge_lengths,
                                      face_indices.astype(np.uint32),
                                      face_areas,
                                      intra_slice_weight,
                                      cross_slice_winsor,
                                      intra_slice_winsor,
                                      block_lo, block_hi,
                                      num_threads)

        m_o = mean_offset(all_mesh_pts,
                          all_pairs,
                          src_indices,
                          dest_indices,
                          dest_weights,
                          match_offsets,
                          block_lo, block_hi)
        print "AFTER", "C", cost, "MO", m_o, "T:", time.time() - oldtick
        oldtick = time.time()

        gradient_with_momentum = 0
        stepsize = 0.1
        prev_cost = np.inf
        gradient = np.empty_like(all_mesh_pts[block_lo:block_hi, ...])
        old_pts = all_mesh_pts[block_lo:block_hi, ...].copy()

        for iter in range(max_iterations):
            cost = mesh_derivs.all_derivs(all_mesh_pts,
                                          gradient,
                                          all_pairs,
                                          between_mesh_weights,
                                          src_indices,
                                          dest_indices,
                                          dest_weights,
                                          match_offsets,
                                          edge_indices.astype(np.uint32),
                                          edge_lengths,
                                          face_indices.astype(np.uint32),
                                          face_areas,
                                          intra_slice_weight,
                                          cross_slice_winsor,
                                          intra_slice_winsor,
                                          block_lo, block_hi,
                                          num_threads)

            if iter > 0 and iter % 100 == 0:
                m_o = mean_offset(all_mesh_pts,
                                  all_pairs,
                                  src_indices,
                                  dest_indices,
                                  dest_weights,
                                  match_offsets,
                                  block_lo, block_hi)

                print iter, "SL:", block_lo, block_hi, num_meshes, "COST:", cost, "MO:", m_o, "SZ:", stepsize, "T:", time.time() - oldtick
                oldtick = time.time()
                # TODO: parameter for m_o cutoff
                if (cost < prev_cost) and \
                   (iter >= min_iterations) and \
                   ((m_o < mean_offset_threshold * mesh.layer_scale) or (stepsize < 1e-20)):
                    break

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
            if cost < prev_cost and not np.isinf(cost):
                stepsize *= 1.1
                if stepsize > 1.0:
                    stepsize = 1.0
                # update with new gradients
                gradient_with_momentum = (gradient + 0.5 * gradient_with_momentum)
                old_pts = all_mesh_pts[block_lo:block_hi, ...].copy()
                all_mesh_pts[block_lo:block_hi, ...] -= stepsize * gradient_with_momentum
                prev_cost = cost
            else:  # we took a bad step: undo it, scale down stepsize, and start over
                all_mesh_pts[block_lo:block_hi, ...] = old_pts[...]
                stepsize *= 0.5
                gradient_with_momentum = 0.0
                prev_cost = np.inf
            assert not np.any(~ np.isfinite(all_mesh_pts))

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
    mesh_file = sys.argv[1]
    matches_files = glob.glob(os.path.join(sys.argv[2], '*W02_sec0[012]*W02_sec0[012]*.json'))
    print("Found {} match files".format(len(matches_files)))
    tsfile_to_layerid = None
    new_positions = optimize_meshes(mesh_file, matches_files, tsfile_to_layerid)

    out_file = sys.argv[3]
    json.dump(new_positions, open(out_file, "w"), indent=1)
