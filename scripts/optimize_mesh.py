import sys
import json
import glob
import os.path
import theano
import theano.tensor as T
import numpy as np
from scipy.spatial import cKDTree as KDTree  # for searching surrounding points
from collections import defaultdict
from progressbar import ProgressBar, ETA, Bar, Counter

sys.setrecursionlimit(10000)  # for grad

class MeshParser(object):
    def __init__(self, mesh_file, multiplier=1000):
        # load the mesh
        self.mesh = json.load(open(mesh_file))
        self.pts = np.array([(p["x"], p["y"]) for p in self.mesh["points"]], dtype=np.float32)
        self.rowcols = np.array([(p["row"], p["col"]) for p in self.mesh["points"]])
        self.layer_scale = float(self.mesh["layerScale"])

        print "# points in base mesh", self.pts.shape[0]

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
        print "Mesh long-row boundary values: min_x: {}, miny: {}, max_x: {}, max_y: {}".format(
            self.long_row_min_x, self.long_row_min_y, self.long_row_max_x, self.long_row_max_y)
        self.short_row_min_x = min(mesh_p_x[mesh_p_x > self.long_row_min_x])
        self.short_row_min_y = min(mesh_p_y[mesh_p_y > self.long_row_min_y])
        self.short_row_max_x = max(mesh_p_x[mesh_p_x < self.long_row_max_x])
        self.short_row_max_y = max(mesh_p_y[mesh_p_y < self.long_row_max_y])
        print "Mesh short-row boundary values: min_x: {}, miny: {}, max_x: {}, max_y: {}".format(
            self.short_row_min_x, self.short_row_min_y, self.short_row_max_x, self.short_row_max_y)

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
    l = T.switch(abs(d) <= delta, a, b)
    return l.sum()


# Based on Tukey's bisquare M-estimator function (Tukey's bisquare M-estimator from http://www.statisticalconsultants.co.nz/blog/m-estimators.html)
def bisquare(target, output, c):
    z_i = target - output
    a = (c ** 6 - (c ** 2 - z_i ** 2) ** 3) / 6.
    b = (c ** 6) / 6.
    l = T.switch(abs(z_i) <= c, a, b)
    return l.sum()

def link_cost(lengths, weight, winsor, rest_length):
    '''cost for edges, using a winsorized loss function.

    Springs are quadratic within a window of +/- winsor of their rest lenght,
    and linear beyond that.
    '''

    return weight * huber(lengths, rest_length, winsor)
    # return weight * bisquare(lengths, rest_length, winsor)

def regularized_lengths(vec):
    return T.sqrt(T.sum(T.sqr(vec) + 0.01, axis=1))

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

def make_cross_gradfun(cross_slice_weight, cross_slice_winsor):
    # set up cost function for cross-section springs
    mesh1 = T.matrix('mesh1')
    mesh2 = T.matrix('mesh2')
    idx1 = T.ivector('idx1')
    idx2_1 = T.ivector('idx2_1')
    idx2_2 = T.ivector('idx2_2')
    idx2_3 = T.ivector('idx2_3')
    w1 = T.col('w1')
    w2 = T.col('w2')
    w3 = T.col('w3')
    separation = T.scalar('separation')

    p1_locs = mesh1.take(idx1, axis=0)
    p2_locs = (w1 * mesh2.take(idx2_1, axis=0) +
               w2 * mesh2.take(idx2_2, axis=0) +
               w3 * mesh2.take(idx2_3, axis=0))
    lengths = regularized_lengths(p1_locs - p2_locs)
    this_cost = link_cost(lengths, cross_slice_weight / separation, cross_slice_winsor, 0)
    return theano.function([mesh1, mesh2,
                            idx1,
                            idx2_1, idx2_2, idx2_3,
                            w1, w2, w3,
                            separation],
                           [this_cost,
                            theano.Out(T.grad(this_cost, mesh1), borrow=True),
                            theano.Out(T.grad(this_cost, mesh2), borrow=True)],
                           on_unused_input='warn')

def make_internal_gradfun(intra_slice_weight, intra_slice_winsor):
    # set up cost function for in-sectino springs
    mesh = T.matrix('mesh')
    neighbor_idx = T.ivector()
    rest_lengths = T.vector()
    lengths = regularized_lengths(mesh - mesh.take(neighbor_idx, axis=0))
    this_cost = link_cost(lengths, intra_slice_weight, intra_slice_winsor, rest_lengths)
    return theano.function([mesh, neighbor_idx, rest_lengths],
                           [this_cost,
                            theano.Out(T.grad(this_cost, mesh), borrow=True)])


def load_matches(matches_files, mesh):
    pbar = ProgressBar(widgets=['Loading matches: ', Counter(), ' / ', str(len(matches_files)), " ", Bar(), ETA()])

    for midx, mf in enumerate(pbar(matches_files)):
        for m in json.load(open(mf)):
            if not m['shouldConnect']:
                continue

            # TODO - if one of the layers should be skipped, we don't need to add its meshes

            # parse matches file, and get p1's mesh x and y points
            orig_p1s = np.array([(pair["p1"]["l"][0], pair["p1"]["l"][1]) for pair in m["correspondencePointPairs"]], dtype=np.float32)

            p1_rc_indices = [mesh.rowcolidx(p1) for p1 in orig_p1s]

            p2_locs = np.array([pair["p2"]["l"] for pair in m["correspondencePointPairs"]])
            dists, surround_indices = mesh.query_cross(p2_locs, 3)
            surround_indices = surround_indices.astype(np.int32)
            tris_x = mesh.pts[surround_indices, 0]
            tris_y = mesh.pts[surround_indices, 1]
            w1, w2, w3 = barycentric(p2_locs, tris_x, tris_y)
            yield m["url1"], m["url2"], p1_rc_indices, surround_indices, w1, w2, w3


def optimize_meshes(mesh_file, matches_files, url_to_layerid, conf_dict={}):
    # set default values
    cross_slice_weight = conf_dict.get("cross_slice_weight", 1.0)
    cross_slice_winsor = conf_dict.get("cross_slice_winsor", 1000)
    intra_slice_weight = conf_dict.get("intra_slice_weight", 1.0 / 6)
    intra_slice_winsor = conf_dict.get("intra_slice_winsor", 200)
    max_iterations = conf_dict.get("max_iterations", 5000)

    # Load the mesh
    mesh = MeshParser(mesh_file)

    # Adjust winsor values according to layer scale
    cross_slice_winsor = cross_slice_winsor * mesh.layer_scale
    intra_slice_winsor = intra_slice_winsor * mesh.layer_scale

    # Create the per-tile meshes
    per_tile_mesh = defaultdict(lambda: theano.shared(mesh.pts))  # not borrowed

    cross_links = list(load_matches(matches_files, mesh))

    # create url_to_layerid if it wasn't passed in
    if url_to_layerid is None:
        url_to_layerid = {}
        for cl in cross_links:
            for url in cl[:2]:
                if url not in url_to_layerid:
                    url_to_layerid[url] = int(url[:-5][-3:])

    Fcross = make_cross_gradfun(cross_slice_weight, cross_slice_winsor)
    Finternal = make_internal_gradfun(intra_slice_weight, intra_slice_winsor)

    # Build structural mesh
    dists, neighbor_indices = mesh.query_internal(mesh.pts, 7)
    neighbor_indices = neighbor_indices.astype(np.int32)
    dists = dists.astype(np.float32)

    def check_nan(g):
        assert not np.any(np.isnan(g))

    cost = 0.0

    prev_cost = np.inf
    stepsize = 0.1
    grads = defaultdict(lambda: 0.0)

    class MonitorValues(object):
        '''add cost, etc. to progress bar'''
        def mean_cross_dist(self):
            badcount = 0
            mesh_values = {url: per_tile_mesh[url].get_value() for url in per_tile_mesh.keys()}
            dists = []
            for url1, url2, p1_rc, surround, w1, w2, w3 in cross_links:
                separation = abs(url_to_layerid[url1] - url_to_layerid[url2])
                pts1 = mesh_values[url1][p1_rc]
                mesh2 = mesh_values[url2]
                pts2 = (w1 * mesh2.take(surround[:, 0], axis=0) +
                        w2 * mesh2.take(surround[:, 1], axis=0) +
                        w3 * mesh2.take(surround[:, 2], axis=0))
                sep = np.sqrt(((pts1 - pts2) ** 2).sum(axis=1))
                badcount += sum(sep > 100)
                if separation == 1:
                    dists.append(sep.mean())
            return np.mean(dists)

        def update(self, count):
            l = 0
            if count.currval > 0:
                l = self.mean_cross_dist()
            return 'Err: {:.2f}  step: {:.2f}  |g|_1: {:.2f}  Len: {:.2f}'.format(cost, stepsize, sum(np.sum(abs(g)) for g in grads.values()), l)


    pbar = ProgressBar(widgets=['Iter ', Counter(), '/{0} '.format(max_iterations), MonitorValues(), Bar(), ETA()])
    for iter in pbar(range(max_iterations)):
        cost = 0.0

        new_grads = defaultdict(lambda: 0.0)

        for url1, url2, m1_indices, m2_surround_indices, w1, w2, w3 in cross_links:
            # TODO - the separation value needs to be set according to the diff in layer id (not the wafer/section number)
            # separation = abs(int(url1.split('.')[-2][-3:]) - int(url2.split('.')[-2][-3:]))
            separation = abs(url_to_layerid[url1] - url_to_layerid[url2])
            c, g1, g2 = Fcross(per_tile_mesh[url1].get_value(borrow=True),
                               per_tile_mesh[url2].get_value(borrow=True),
                               m1_indices,
                               m2_surround_indices[:, 0], m2_surround_indices[:, 1], m2_surround_indices[:, 2],
                               w1, w2, w3,
                               separation)
            cost += c
            new_grads[url1] += g1
            new_grads[url2] += g2
            check_nan(g1)
            check_nan(g2)

        for url in per_tile_mesh.keys():
            for idx in range(1, 7):
                c, g = Finternal(per_tile_mesh[url].get_value(borrow=True),
                                 neighbor_indices[:, idx],
                                 dists[:, idx])
                cost += c
                new_grads[url] += g
                check_nan(g)

        # step size adjustment
        if cost <= prev_cost:
            stepsize *= 1.05
            if stepsize > 1.0:
                stepsize = 1.0
            # update with new gradients
            for url in grads.keys():
                grads[url] = new_grads[url] + 0.5 * grads[url]  # momentum of 0.5

            # step to next evaluation point
            for url in per_tile_mesh.keys():
                per_tile_mesh[url].set_value(per_tile_mesh[url].eval() - stepsize * grads[url])

            prev_cost = cost


        else:  # we took a bad step: undo it, scale down stepsize, and start over
            for url in per_tile_mesh.keys():
                per_tile_mesh[url].set_value(per_tile_mesh[url].eval() + stepsize * grads[url])

            stepsize *= 0.5

            for url in grads.keys():
                grads[url] *= 0.0  # clear momentum when we take too large a step

            prev_cost = np.inf


    # Prepare per-layer output
    out_positions = {}

    for url in per_tile_mesh.keys():
        out_positions[url] = [[(pt[0] / mesh.layer_scale, pt[1] / mesh.layer_scale) for pt in mesh.pts],
                              [(pt[0] / mesh.layer_scale, pt[1] / mesh.layer_scale) for pt in per_tile_mesh[url].eval()]]

    return out_positions


if __name__ == '__main__':
    mesh_file = sys.argv[1]
    matches_files = glob.glob(os.path.join(sys.argv[2], '*W02_sec0[012]*W02_sec0[012]*.json'))
    print "Found {} match files".format(len(matches_files))
    url_to_layerid = None
    new_positions = optimize_meshes(mesh_file, matches_files, url_to_layerid)

    out_file = sys.argv[3]
    json.dump(new_positions, open(out_file, "w"), indent=1)
