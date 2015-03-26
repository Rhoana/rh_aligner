import sys
import json
import glob
import os.path
import theano
import theano.tensor as T
import numpy as np
from scipy.spatial import cKDTree as KDTree  # for searching surrounding points
from collections import defaultdict
from progressbar import ProgressBar, ETA, Bar, Counter, FormatLabel

sys.setrecursionlimit(10000)  # for grad

cross_slice_weight = 1.0
cross_slice_winsor = 30

intra_slice_weight = 1.0 / 6
intra_slice_winsor = 30

stepsize = 0.1

class MeshParser(object):
    mesh = None
    kdt = None  # for the Neighbors
    pts = None
    rowcols = None
    layer_scale = None

    def __init__(self, mesh_file):
        # load the mesh
        self.mesh = json.load(open(mesh_file))
        self.pts = np.array([(p["x"], p["y"]) for p in self.mesh["points"]], dtype=np.float32)
        self.rowcols = np.array([(p["row"], p["col"]) for p in self.mesh["points"]])
        self.layer_scale = float(self.mesh["layerScale"])

        print "# points in base mesh", self.pts.shape[0]

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

        # Build the KDTree for neighbor searching
        self.kdt = KDTree(self.pts, leafsize=3)

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
        """Returns the k nearest neighbors, while taking into account the mesh formation.
        If a point is on the boudaries of the mesh, only the relevant neighbors will be returned,
        and all others wil have distance -1, and location [-1, -1]"""
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

def make_cross_gradfun():
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
                            w1, w2, w3, separation],
                           [this_cost, T.grad(this_cost, mesh1), T.grad(this_cost, mesh2)])

def make_internal_gradfun():
    # set up cost function for in-sectino springs
    mesh = T.matrix('mesh')
    neighbor_idx = T.ivector()
    rest_lengths = T.vector()
    lengths = regularized_lengths(mesh - mesh.take(neighbor_idx, axis=0))
    this_cost = link_cost(lengths, intra_slice_weight, intra_slice_winsor, rest_lengths)
    return theano.function([mesh, neighbor_idx, rest_lengths],
                           [this_cost, T.grad(this_cost, mesh)])


def parse_mesh_file(mesh_file):
    mesh = json.load(open(mesh_file))
    pts = np.array([(p["x"], p["y"]) for p in mesh["points"]], dtype=np.float32)
    rowcols = np.array([(p["row"], p["col"]) for p in mesh["points"]])

    num_base = pts.shape[0]
    print "# points in base mesh", num_base
    return pts, rowcols

def optimize_meshes(mesh_file, matches_files, url_to_layerid, conf_dict):
    # set default values
    max_iterations = 50

    # read parameters from the configuration dictionary
    if "cross_slice_weight" in conf_dict.keys():
        cross_slice_weight = conf_dict["cross_slice_weight"]
    if "cross_slice_winsor" in conf_dict.keys():
        cross_slice_winsor = conf_dict["cross_slice_winsor"]
    if "intra_slice_weight" in conf_dict.keys():
        intra_slice_weight = conf_dict["intra_slice_weight"]
    if "intra_slice_winsor" in conf_dict.keys():
        intra_slice_winsor = conf_dict["intra_slice_winsor"]
    if "max_iterations" in conf_dict.keys():
        max_iterations = conf_dict["max_iterations"]

    mesh = MeshParser(mesh_file)

    # Adjust winsor values according to layer scale
    cross_slice_winsor = int(round(cross_slice_winsor * mesh.layer_scale))
    intra_slice_winsor = int(round(intra_slice_winsor * mesh.layer_scale))

    print "cross_slice_winsor: {}, intra_slice_winsor: {}".format(cross_slice_winsor, intra_slice_winsor)
    print "cross_slice_weight: {}, intra_slice_weight: {}".format(cross_slice_weight, intra_slice_weight)

    # seed rowcolidx to match mesh_pts array
    rowcolidx = {}
    for idx, rc in enumerate(mesh.rowcols):
        rowcolidx[tuple(rc)] = idx

    # create a dictionary from the mesh p1 points x,y to a unique row,col (we compare shift 3 decimal points, and compare integers instead of floats)
    points_multiplier = 10 ** 3
    mesh_y_to_row_dict = dict([(int(p[1] * points_multiplier), rowcol[0]) for (p,rowcol) in zip(mesh.pts, mesh.rowcols)])
    mesh_x_to_col_dict = dict([(int(p[0] * points_multiplier), rowcol[1]) for (p,rowcol) in zip(mesh.pts, mesh.rowcols)])

#    print "mesh_x_to_col_dict:", sorted(mesh_x_to_col_dict.keys())
#    print "mesh_y_to_row_dict:", sorted(mesh_y_to_row_dict.keys())

    per_tile_mesh_pts = {}

    def mesh_for_tile(tile_name):
        if tile_name not in per_tile_mesh_pts:
            per_tile_mesh_pts[tile_name] = theano.shared(mesh.pts, name=tile_name)  # not borrowed
        return per_tile_mesh_pts[tile_name]

    cross_links = []

    pbar = ProgressBar(widgets=['Loading matches: ', Counter(), ' / ', str(len(matches_files)), " ", Bar(), ETA()])

    for midx, mf in enumerate(pbar(matches_files)):
        for m in json.load(open(mf)):
            if not m['shouldConnect']:
                continue

            # TODO - if one of the layers should be skipped, we don't need to add its meshes

            # parse matches file, and get p1's mesh x and y points
            orig_p1s = np.array([(pair["p1"]["l"][0], pair["p1"]["l"][1]) for pair in m["correspondencePointPairs"]], dtype=np.float32)

            p1_rc_indices = [rowcolidx[
                mesh_y_to_row_dict[int(p1[1] * points_multiplier)],
                mesh_x_to_col_dict[int(p1[0] * points_multiplier)]] for p1 in orig_p1s]

            p2_locs = np.array([pair["p2"]["l"] for pair in m["correspondencePointPairs"]])
            dists, surround_indices = mesh.query_cross(p2_locs, 3)
            surround_indices = surround_indices.astype(np.int32)
            tris_x = mesh.pts[surround_indices, 0]
            tris_y = mesh.pts[surround_indices, 1]
            w1, w2, w3 = barycentric(p2_locs, tris_x, tris_y)

            cross_links.append((m["url1"], m["url2"], p1_rc_indices, surround_indices, w1, w2, w3))

    Fcross = make_cross_gradfun()
    Finternal = make_internal_gradfun()

    # Build structural mesh
    dists, neighbor_indices = mesh.query_internal(mesh.pts, 7)
    neighbor_indices = neighbor_indices.astype(np.int32)
    dists = dists.astype(np.float32)

    def check_nan(g):
        assert not np.any(np.isnan(g))

    cost = 0.0

    class Foo(object):
        def update(self, ignore):
            return '{:.2f}'.format(cost)

    pbar = ProgressBar(widgets=['Iter ', Counter(), '/{0} '.format(max_iterations), Foo(), Bar(), ETA()])
    for iter in pbar(range(max_iterations)):
        cost = 0.0
        grads = defaultdict(int)

        for url1, url2, m1_indices, m2_surround_indices, w1, w2, w3 in cross_links:
            # TODO - the separation value needs to be set according to the diff in layer id (not the wafer/section number)
            # separation = abs(int(url1.split('.')[-2][-3:]) - int(url2.split('.')[-2][-3:]))
            separation = abs(url_to_layerid[url1] - url_to_layerid[url2])
            c, g1, g2 = Fcross(mesh_for_tile(url1).eval(),
                               mesh_for_tile(url2).eval(),
                               m1_indices,
                               m2_surround_indices[:, 0], m2_surround_indices[:, 1], m2_surround_indices[:, 2],
                               w1, w2, w3,
                               separation)
            cost += c
            grads[url1] += g1
            grads[url2] += g2
            check_nan(g1)
            check_nan(g2)

        for url in per_tile_mesh_pts.keys():
            for idx in range(1, 7):
                c, g = Finternal(mesh_for_tile(url).eval(),
                                 neighbor_indices[:, idx],
                                 dists[:, idx])
                cost += c
                grads[url] += g
                check_nan(g)

        for url in per_tile_mesh_pts.keys():
            mesh_for_tile(url).set_value((mesh_for_tile(url) - stepsize * grads[url]).eval())

    # Prepare per-layer output
    out_positions = {}

    # create a dictionary from the mesh p1 row,col to the corresponding x,y points
    mesh_row_to_y_dict = dict([(rowcol[0], p[1]) for (p, rowcol) in zip(mesh.pts, mesh.rowcols)])
    mesh_col_to_x_dict = dict([(rowcol[1], p[0]) for (p, rowcol) in zip(mesh.pts, mesh.rowcols)])

    for url in per_tile_mesh_pts.keys():
        out_positions[url] = [[(mesh_col_to_x_dict[c] / mesh.layer_scale, mesh_row_to_y_dict[r] / mesh.layer_scale) for r, c in mesh.rowcols],
                              [(float(new_x) / mesh.layer_scale, float(new_y) / mesh.layer_scale) for new_x, new_y in mesh_for_tile(url).eval()]]

    return out_positions


if __name__ == '__main__':
    mesh_file = sys.argv[1]
    matches_files = glob.glob(os.path.join(sys.argv[2], '*W02_sec0[012]*W02_sec0[012]*.json'))
    print "Found {} match files".format(len(matches_files))

    pts, rowcols = parse_mesh_file(mesh_file)

    # Build the KDTree for neighbor searching
    kdt = KDTree(pts, leafsize=3)

    # seed rowcolidx to match pts array
    rowcolidx = {}
    for idx, rc in enumerate(rowcols):
        rowcolidx[tuple(rc)] = idx

    mesh_pts = {}

    def mesh_for_tile(tile_name):
        if tile_name not in mesh_pts:
            mesh_pts[tile_name] = theano.shared(pts, name=tile_name)  # not borrowed
        return mesh_pts[tile_name]

    cross_links = []
    costs_by_mesh = defaultdict(list)

    pbar = ProgressBar(widgets=['Loading matches: ', Counter(), ' / ', str(len(matches_files)), " ", Bar(), ETA()])

    for midx, mf in enumerate(pbar(matches_files)):
        for m in json.load(open(mf)):
            if not m['shouldConnect']:
                continue

            # find barycentric weights
            p1s = (pair["p1"] for pair in m["correspondencePointPairs"])
            p1_rc_indices = [rowcolidx[p1["row"], p1["col"]] for p1 in p1s]

            p2_locs = np.array([pair["p2"]["l"] for pair in m["correspondencePointPairs"]])
            dists, surround_indices = kdt.query(p2_locs, 3)
            surround_indices = surround_indices.astype(np.int32)
            tris_x = pts[surround_indices, 0]
            tris_y = pts[surround_indices, 1]
            w1, w2, w3 = barycentric(p2_locs, tris_x, tris_y)

            cross_links.append((m["url1"], m["url2"], p1_rc_indices, surround_indices, w1, w2, w3))

    Fcross = make_cross_gradfun()
    Finternal = make_internal_gradfun()

    # Build structural mesh
    dists, neighbor_indices = kdt.query(pts, 7)
    neighbor_indices = neighbor_indices.astype(np.int32)
    dists = dists.astype(np.float32)

    def check_nan(g):
        assert not np.any(np.isnan(g))

    cost = 0.0

    class Foo(object):
        '''class to get the cost into the progress bar'''
        def update(self, ignore):
            return '{:.2f}'.format(cost)

    max_iterations = 1000

    pbar = ProgressBar(widgets=['Iter ', Counter(), '/{0} '.format(max_iterations), Foo(), Bar(), ETA()])
    for iter in pbar(range(max_iterations)):
        cost = 0.0
        grads = defaultdict(int)

        for url1, url2, m1_indices, m2_surround_indices, w1, w2, w3 in cross_links:
            separation = abs(int(url1.split('.')[-2][-3:]) - int(url2.split('.')[-2][-3:]))
            c, g1, g2 = Fcross(mesh_for_tile(url1).eval(),
                               mesh_for_tile(url2).eval(),
                               m1_indices,
                               m2_surround_indices[:, 0], m2_surround_indices[:, 1], m2_surround_indices[:, 2],
                               w1, w2, w3,
                               separation)
            cost += c
            grads[url1] += g1
            grads[url2] += g2
            check_nan(g1)
            check_nan(g2)

        for url in mesh_pts.keys():
            for idx in range(1, 7):
                c, g = Finternal(mesh_for_tile(url).eval(),
                                 neighbor_indices[:, idx],
                                 dists[:, idx])
                cost += c
                grads[url] += g
                check_nan(g)

        for url in mesh_pts.keys():
            mesh_for_tile(url).set_value(mesh_for_tile(url).eval() - stepsize * grads[url])

    new_positions = []
    for url in mesh_pts.keys():
        out_dict = {}
        out_dict["url"] = url
        out_dict["new_positions"] = [{"row": r,
                                      "col": c,
                                      "new_x": float(new_x),
                                      "new_y": float(new_y)}
                                     for ((r, c), (new_x, new_y)) in
                                     zip(rowcols, mesh_for_tile(url).get_value())]
        new_positions.append(out_dict)

    out_file = sys.argv[3]
    json.dump(new_positions, open(out_file, "w"), indent=1)
