import sys
import json
import glob
import os.path
import time
import numpy as np
from numpy.linalg import lstsq
import cPickle as pickle
from scipy.spatial import cKDTree as KDTree  # for searching surrounding points
from collections import defaultdict
from progressbar import ProgressBar, ETA, Bar, Counter
import threading
import Queue

import pyximport
pyximport.install()
import mesh_derivs


sys.setrecursionlimit(10000)  # for grad

class MeshParser(object):
    def __init__(self, mesh_file, multiplier=1000):
        # load the mesh
        self.mesh = json.load(open(mesh_file))
        self.pts = np.array([(p["x"], p["y"]) for p in self.mesh["points"]], dtype=np.float32)
        self.rowcols = np.array([(p["row"], p["col"]) for p in self.mesh["points"]])
        self.layer_scale = float(self.mesh["layerScale"])

        print("# points in base mesh {}".format(self.pts.shape[0]))

        self.multiplier = multiplier
        # seed rowcolidx to match mesh_pts array
        self._rowcolidx = {}
        for idx, pt in enumerate(self.pts):
            self._rowcolidx[int(pt[0] * multiplier), int(pt[1] * multiplier)] = idx

        # Build the KDTree for neighbor searching
        self.kdt = KDTree(self.pts, leafsize=3)

        mesh_p_x = self.pts[:, 0]
        mesh_p_y = self.pts[:, 1]
        self.long_row_min_x = min(mesh_p_x)
        self.long_row_min_y = min(mesh_p_y)
        self.long_row_max_x = max(mesh_p_x)
        self.long_row_max_y = max(mesh_p_y)
        print("Mesh long-row boundary values: min_x: {}, miny: {}, max_x: {}, max_y: {}".format(
            self.long_row_min_x, self.long_row_min_y, self.long_row_max_x, self.long_row_max_y))
        self.short_row_min_x = min(mesh_p_x[mesh_p_x > self.long_row_min_x])
        self.short_row_min_y = min(mesh_p_y[mesh_p_y > self.long_row_min_y])
        self.short_row_max_x = max(mesh_p_x[mesh_p_x < self.long_row_max_x])
        self.short_row_max_y = max(mesh_p_y[mesh_p_y < self.long_row_max_y])
        print("Mesh short-row boundary values: min_x: {}, miny: {}, max_x: {}, max_y: {}".format(
            self.short_row_min_x, self.short_row_min_y, self.short_row_max_x, self.short_row_max_y))

    def rowcolidx(self, xy):
        return self._rowcolidx[int(xy[0] * self.multiplier), int(xy[1] * self.multiplier)]

    def query_internal(self, points, k):
        """Returns the k nearest neighbors, while taking into account the mesh formation.
        If a point is on the boudaries of the mesh, only the relevant neighbors will be returned,
        and all others wil have distance -1, and location [-1, -1]"""
        dists, surround_indices = self.kdt.query(points, k)

        # Find out if the point is on the "boundary"
        for i, p in enumerate(points):
            # on the left side of the mesh
            if p[0] < self.short_row_min_x:
                # remove any surrounding point that is to the right of short_row_min_x
                point_surround_indices = surround_indices[i].astype(np.int32)
                tris_x = self.pts[point_surround_indices, 0]
                dists[i][tris_x > self.short_row_min_x] = 0.0
                surround_indices[i][tris_x > self.short_row_min_x] = surround_indices[i][0]
            # on the left side of the mesh
            if p[0] > self.short_row_max_x:
                # remove any surrounding point that is to the left of short_row_max_x
                point_surround_indices = surround_indices[i].astype(np.int32)
                tris_x = self.pts[point_surround_indices, 0]
                dists[i][tris_x < self.short_row_max_x] = 0.0
                surround_indices[i][tris_x < self.short_row_max_x] = surround_indices[i][0]
            # on the upper side of the mesh
            if p[1] < self.short_row_min_y:
                # remove any surrounding point that is above short_row_min_y
                point_surround_indices = surround_indices[i].astype(np.int32)
                tris_y = self.pts[point_surround_indices, 1]
                dists[i][tris_y > self.short_row_min_y] = 0.0
                surround_indices[i][tris_y > self.short_row_min_y] = surround_indices[i][0]
            # on the lower side of the mesh
            if p[1] > self.short_row_max_y:
                # remove any surrounding point that is under short_row_max_y
                point_surround_indices = surround_indices[i].astype(np.int32)
                tris_y = self.pts[point_surround_indices, 1]
                dists[i][tris_y < self.short_row_max_y] = 0.0
                surround_indices[i][tris_y < self.short_row_max_y] = surround_indices[i][0]

        return dists, surround_indices

    def query_cross(self, points, k):
        """Returns the k nearest neighbors"""
        dists, surround_indices = self.kdt.query(points, k)

        return dists, surround_indices


# Original huber function (Huber's M-estimator from http://www.statisticalconsultants.co.nz/blog/m-estimators.html)
def huber(target, output, delta):
    ''' from http://breze.readthedocs.org/en/latest/specify.html '''
    d = target - output
    a = .5 * d ** 2
    b = delta * (abs(d) - delta / 2.)
    l = np.switch(abs(d) <= delta, a, b)
    return l.sum()


# Based on Tukey's bisquare M-estimator function (Tukey's bisquare M-estimator from http://www.statisticalconsultants.co.nz/blog/m-estimators.html)
def bisquare(target, output, c):
    z_i = target - output
    a = (c ** 6 - (c ** 2 - z_i ** 2) ** 3) / 6.
    b = (c ** 6) / 6.
    l = np.switch(abs(z_i) <= c, a, b)
    return l.sum()

def link_cost(lengths, weight, winsor, rest_length):
    '''cost for edges, using a winsorized loss function.

    Springs are quadratic within a window of +/- winsor of their rest lenght,
    and linear beyond that.
    '''

    return weight * huber(lengths, rest_length, winsor)
    # return weight * bisquare(lengths, rest_length, winsor)

def regularized_lengths(vec):
    return np.sqrt(np.sum(np.sqr(vec), axis=1) + 0.0001)

def barycentric(pt, verts_x, verts_y):
    '''computes the barycentric weights to reconstruct an array of points in an
    array of triangles.

    '''
    x, y = pt.T
    x_1, x_2, x_3 = verts_x.T
    y_1, y_2, y_3 = verts_y.T

    # from wikipedia
    den = ((y_2 - y_3) * (x_1 - x_3) + (x_3 - x_2) * (y_1 - y_3))
    l1 = ((y_2 - y_3) * (x - x_3) + (x_3 - x_2) * (y - y_3)) / den
    l2 = ((y_3 - y_1) * (x - x_3) + (x_1 - x_3) * (y - y_3)) / den
    l3 = 1 - l1 - l2
    return l1.reshape((-1, 1)).astype(np.float32), l2.reshape((-1, 1)).astype(np.float32), l3.reshape((-1, 1)).astype(np.float32)

def load_matches(matches_files, mesh):
    pbar = ProgressBar(widgets=['Loading matches: ', Counter(), ' / ', str(len(matches_files)), " ", Bar(), ETA()])

    for midx, mf in enumerate(pbar(sorted(matches_files))):
        for m in json.load(open(mf)):
            if not m['shouldConnect']:
                continue

            # TODO - if one of the layers should be skipped, we don't need to add its meshes

            # parse matches file, and get p1's mesh x and y points
            orig_p1s = np.array([(pair["p1"]["l"][0], pair["p1"]["l"][1]) for pair in m["correspondencePointPairs"]], dtype=np.float32)

            p1_rc_indices = [mesh.rowcolidx(p1) for p1 in orig_p1s]

            p2_locs = np.array([pair["p2"]["l"] for pair in m["correspondencePointPairs"]])
            dists, surround_indices = mesh.query_cross(p2_locs, 3)
            surround_indices = surround_indices.astype(np.uint32)
            tris_x = mesh.pts[surround_indices, 0]
            tris_y = mesh.pts[surround_indices, 1]
            w1, w2, w3 = barycentric(p2_locs, tris_x, tris_y)
            m["url1"] = m["url1"].replace("/n/regal/pfister_lab/adisuis/Alyssa_P3_W02_to_W08", "/data/Adi/mesh_optimization/data")
            m["url2"] = m["url2"].replace("/n/regal/pfister_lab/adisuis/Alyssa_P3_W02_to_W08", "/data/Adi/mesh_optimization/data")

            reorder = np.argsort(p1_rc_indices)
            p1_rc_indices = np.array(p1_rc_indices).astype(np.uint32)[reorder]
            surround_indices = surround_indices[reorder, :]
            weights = np.hstack((w1, w2, w3)).astype(np.float32)[reorder, :]
            assert p1_rc_indices.shape[0] == weights.shape[0]
            yield m["url1"], m["url2"], p1_rc_indices, surround_indices, weights

def linearize_grad(positions, gradients):
    '''perform a least-squares fit, then return the values from that fit'''
    newpos = np.hstack((positions, np.ones((positions.shape[0], 1))))
    fit, residuals, rank, s = lstsq(newpos, gradients)
    return np.dot(newpos, fit).astype(np.float32)

def blend(a, b, t):
    '''at t=0, return a, at t=1, return b'''
    return a + (b - a) * t

def compute_derivs_worker(queue, mesh_locks, costs,
                          per_tile_mesh, new_grads,
                          neighbor_indices, dists,
                          source_indices, surround_indices, bary_weights,
                          intra_slice_weight, intra_slice_winsor,
                          separations, cross_slice_winsor):
    '''worker thread for computing derivatives'''
    while True:
        url1, url2 = queue.get()
        # try to acquire the locks (nonblocking)
        locked_1 = True
        locked_2 = True
        # locked_1 = mesh_locks[url1].acquire(True)
        # locked_2 = mesh_locks[url2].acquire(False)
        if locked_1 and locked_2:
            # we have the locks, do some work
            if url1 == url2:
                # compute within-mesh cost and derivs
                cost = 0
                for idx in range(1, 7):
                    m = per_tile_mesh[url1]
                    tc = mesh_derivs.internal_mesh_derivs(m,
                                                          new_grads[url1],
                                                          neighbor_indices[:, idx],
                                                          dists[:, idx],
                                                          np.float64(intra_slice_weight),
                                                          np.float64(intra_slice_winsor))
                    cost += tc
                costs[url1, url2] = cost
            else:
                # compute between-mesh cost and derivs
                separation = separations[url1, url2]
                m1 = per_tile_mesh[url1]
                m2 = per_tile_mesh[url2]
                cost = mesh_derivs.crosslink_mesh_derivs(m1, m2,
                                                         new_grads[url1],
                                                         new_grads[url2],
                                                         source_indices[url1, url2],
                                                         surround_indices[url1, url2],
                                                         bary_weights[url1, url2],
                                                         1.0 / separation,
                                                         np.float32(cross_slice_winsor))
                costs[url1, url2] = cost
        else:  # we didn't get the locks. :sad_face:
            # put the work back into the queue (note that this creates a separate instance of this task)
            queue.put((url1, url2))

        # mark this task as done
        queue.task_done()
        # release locks
        if locked_1:
            pass # mesh_locks[url1].release()
        if locked_2:
            pass # mesh_locks[url2].release()




def optimize_meshes(mesh_file, matches_files, url_to_layerid, conf_dict={}):
    # set default values
    cross_slice_weight = conf_dict.get("cross_slice_weight", 1.0)
    cross_slice_winsor = conf_dict.get("cross_slice_winsor", 1000)
    intra_slice_weight = conf_dict.get("intra_slice_weight", 1.0 / 6)
    intra_slice_winsor = conf_dict.get("intra_slice_winsor", 200)
    max_iterations = conf_dict.get("max_iterations", 200)
    max_iterations = 200

    # Load the mesh
    mesh = MeshParser(mesh_file)

    # Adjust winsor values according to layer scale
    cross_slice_winsor = cross_slice_winsor * mesh.layer_scale
    intra_slice_winsor = intra_slice_winsor * mesh.layer_scale

    # Create the per-tile meshes
    per_tile_mesh = defaultdict(lambda: mesh.pts.copy())

    cross_links = list(load_matches(matches_files, mesh))

    # Build structural mesh
    dists, neighbor_indices = mesh.query_internal(mesh.pts, 7)
    neighbor_indices = neighbor_indices.astype(np.uint32)
    dists = dists.astype(np.float32)

    def check_nan(g):
        assert not np.any(np.isnan(g))

    cost = 0.0

    prev_cost = np.inf
    stepsize = 0.1
    grads = defaultdict(lambda: 0.0)

    # compute layer separations taking into account missing slices
    all_cross_link_pairs = set((v[0], v[1]) for v in cross_links)

    # make sure every slice that is in url_to_layerid has matches with all of
    # its neighbors that are also present
    present_slices = sorted(list(set(url_to_layerid[v[0]] for v in all_cross_link_pairs) | set(url_to_layerid[v[1]] for v in all_cross_link_pairs)))

    separations = {}
    for url1, url2 in all_cross_link_pairs:
        lo, hi = sorted([url_to_layerid[url1], url_to_layerid[url2]])
        separations[url1, url2] = present_slices[present_slices.index(lo):].index(hi)

    class MonitorValues(object):
        '''add cost, etc. to progress bar'''
        def mean_cross_dist(self, save_all):
            badcount = 0
            dists = []
            match_pts = {}
            for url1, url2, p1_rc, surround, bary_weights in cross_links:
                separation = separations[url1, url2]
                pts1 = per_tile_mesh[url1][p1_rc]
                mesh2 = per_tile_mesh[url2]
                pts2 = (bary_weights[:, 0].reshape((-1, 1)) * mesh2.take(surround[:, 0], axis=0) +
                        bary_weights[:, 1].reshape((-1, 1)) * mesh2.take(surround[:, 1], axis=0) +
                        bary_weights[:, 2].reshape((-1, 1)) * mesh2.take(surround[:, 2], axis=0))
                sep = np.sqrt(((pts1 - pts2) ** 2).sum(axis=1))
                badcount += sum(sep > 100)
                if separation == 1:
                    dists.append(sep.mean())
                if separation == 1 or save_all:
                    match_pts[url_to_layerid[url1], url_to_layerid[url2]] = [pts1, pts2]
            return np.mean(dists), match_pts

        def update(self, pbar):
            l = 0
            if pbar.currval > 0:
                l, match_pts = self.mean_cross_dist(pbar.currval % 50 == 0)
                pickle.dump(match_pts, open("match_pts_{}.pickle".format(pbar.currval), "wb"))
            return 'Err: {:.2f}  step: {:.2f}  |g|_1: {:.2f}  Len: {:.2f}'.format(cost, stepsize, sum(np.sum(abs(g)) for g in grads.values()), l)

    # This will get emptied after each loop through all the meshes
    new_grads = defaultdict(lambda: np.zeros_like(mesh.pts))

    # locks for each mesh
    mesh_locks = defaultdict(lambda: threading.RLock())

    # costs for each iteration
    costs = {}

    # cross-mesh links
    source_indices = {}
    surround_indices = {}
    bary_weights = {}
    for url1, url2, m1_indices, m2_surround_indices, b_weights in cross_links:
        source_indices[url1, url2] = m1_indices
        surround_indices[url1, url2] = m2_surround_indices
        bary_weights[url1, url2] = b_weights
        assert m1_indices.shape[0] == b_weights.shape[0]

    # the work queue
    queue = Queue.Queue()

    # start threads
    num_threads = 5
    for i in range(num_threads):
        t = threading.Thread(target=compute_derivs_worker, args=(queue, mesh_locks, costs,
                                                                 per_tile_mesh, new_grads,
                                                                 neighbor_indices, dists,
                                                                 source_indices, surround_indices, bary_weights,
                                                                 intra_slice_weight, intra_slice_winsor,
                                                                 separations, cross_slice_winsor))
        t.daemon = True
        t.start()

    pbar = ProgressBar(widgets=['Iter ', Counter(), '/{0} '.format(max_iterations), MonitorValues(), Bar(), ETA()])
    for iter in pbar(range(max_iterations)):
        print("")  # keep progress lines from overwriting

        new_grads.clear()
        costs.clear()

        # send work to the workers
        for url1, url2, _, _, _ in cross_links:
            queue.put((url1, url2))

        start = time.time()
        for url in per_tile_mesh.keys():
            queue.put((url, url))

        # wait for all derivatives to be computed
        queue.join()
        cost = sum(costs.values())
        tottime = time.time() - start
        print tottime, cost

        # relaxation of the mesh
        relaxation_end = int(max_iterations * 0.75)
        if iter < relaxation_end:
            for url in new_grads.keys():
                linearized = linearize_grad(per_tile_mesh[url], new_grads[url])
                new_grads[url] = blend(linearized, new_grads[url], iter / float(relaxation_end))

        # step size adjustment
        if cost <= prev_cost:
            stepsize *= 1.1
            if stepsize > 1.0:
                stepsize = 1.0
            # update with new gradients
            for url in grads.keys():
                grads[url] = new_grads[url] + 0.75 * grads[url]  # momentum of 0.5

            # step to next evaluation point
            for url in per_tile_mesh.keys():
                per_tile_mesh[url] = (per_tile_mesh[url] - stepsize * grads[url])

            prev_cost = cost

        else:  # we took a bad step: undo it, scale down stepsize, and start over
            for url in per_tile_mesh.keys():
                per_tile_mesh[url] = (per_tile_mesh[url] + stepsize * grads[url])

            stepsize *= 0.5

            for url in grads.keys():
                grads[url] *= 0.0  # clear momentum when we take too large a step

            prev_cost = np.inf

    # Prepare per-layer output
    out_positions = {}

    for url in per_tile_mesh.keys():
        out_positions[url] = [[(pt[0] / mesh.layer_scale, pt[1] / mesh.layer_scale) for pt in mesh.pts],
                              [(pt[0] / mesh.layer_scale, pt[1] / mesh.layer_scale) for pt in per_tile_mesh[url]]]

    return out_positions


if __name__ == '__main__':
    mesh_file = sys.argv[1]
    matches_files = glob.glob(os.path.join(sys.argv[2], '*W02_sec0[012]*W02_sec0[012]*.json'))
    print("Found {} match files".format(len(matches_files)))
    url_to_layerid = None
    new_positions = optimize_meshes(mesh_file, matches_files, url_to_layerid)

    out_file = sys.argv[3]
    json.dump(new_positions, open(out_file, "w"), indent=1)
