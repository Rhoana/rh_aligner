import numpy as np
import sys
import copy
from scipy.spatial import Delaunay
import scipy.interpolate
import cv2



class AbstractModel(object):
    def __init__(self):
        pass

    def score(self, X, y, epsilon=100.0, min_inlier_ratio=0.01, min_num_inlier=7):
        """
        Computes how good is the transformation.
        This is done by applying the transformation to the collection of points in X,
        and then computing the corresponding distance to the matched point in y.
        If the distance is less than epsilon, the match is considered good.
        """
        X2 = self.apply_special(X)
        # dists_sqr = np.sum((y - X2) ** 2, axis=1)
        dists = np.sqrt(np.sum((y - X2) ** 2, axis=1))
        # print "dists", dists
        good_dists_mask = dists < epsilon
        good_dists_num = np.sum(good_dists_mask)
        # good_dists = dists[dists < epsilon]
        # accepted_ratio = float(good_dists.shape[0]) / X2.shape[0]
        accepted_ratio = float(good_dists_num) / X2.shape[0]

        # The transformation does not adhere to the wanted values, give it a very low score
        if good_dists_num < min_num_inlier or accepted_ratio < min_inlier_ratio:
            return -1, None, -1

        return accepted_ratio, good_dists_mask, 0

    def apply(self, p):
        raise RuntimeError, "Not implemented, but probably should be"
 
    def apply_special(self, p):
        raise RuntimeError, "Not implemented, but probably should be"


    def fit(self, X, y):
        raise RuntimeError, "Not implemented, but probably should be"

    def set_from_modelspec(self, s):
        raise RuntimeError, "Not implemented, but probably should be"

    def is_affine(self):
        return False


class AbstractAffineModel(AbstractModel):
    def __init__(self):
        pass

    def get_matrix(self):
        raise RuntimeError, "Not implemented, but probably should be"


    def apply(self, p):
        """
        Returns a new 2D point(s) after applying the transformation on the given point(s) p
        """
        # Check if p is a single 2D point or a list/array of 2D points
        m = self.get_matrix()
        if len(p.shape) == 1: # a single 2D point
            return np.dot(m, np.append(p, [1]))[:2]
        elif len(p.shape) == 2: # A list of 2D points
            return np.vstack([np.dot(m, np.append(p_i, [1]))[:2] for p_i in p])
        raise RuntimeError, "Invalid points input"

    def is_affine(self):
        return True


class TranslationModel(AbstractAffineModel):
    MIN_MATCHES_NUM = 2
    class_name = "mpicbg.trakem2.transform.TranslationModel2D"

    def __init__(self, delta=np.array([0, 0])):
        self.delta = delta

    def set(self, delta):
        self.delta = np.array(delta)

    def apply(self, p):
        # Check if p is a single 2D point or a list/array of 2D points
        if len(p.shape) == 1: # a single 2D point
            return p + self.delta
        elif len(p.shape) == 2: # A list of 2D points
            return np.vstack([p_i + self.delta for p_i in p])
        raise RuntimeError, "Invalid points input"

    def apply_special(self, p):
        return np.atleast_2d(p) + np.asarray(self.delta).reshape((-1, 2))

    def to_str(self):
        return "T={}".format(self.delta)

    def to_modelspec(self):
        return {
                "className" : self.class_name,
                "dataString" : "{}".format(' '.join([str(float(x)) for x in self.delta]))
            }

    def set_from_modelspec(self, s):
        self.delta = np.array([float(d) for d in s.split()])

    def get_matrix(self):
        return np.array([
                            [1.0, 0.0, self.delta[0]],
                            [0.0, 1.0, self.delta[1]],
                            [0.0, 0.0, 1.0]
                        ])

    def fit(self, X, y):
        """
        A non-weighted fitting of a collection of 2D points in X to a collection of 2D points in y.
        X and y are assumed to be arrays of 2D points of the same shape.
        """
        assert(X.shape[0] >= 2) # the minimal number of of matches for a 2d rigid transformation

        pc = np.mean(X, axis=0)
        qc = np.mean(y, axis=0)

        self.delta = qc - pc
        return True




class RigidModel(AbstractAffineModel):
    MIN_MATCHES_NUM = 2
    class_name = "mpicbg.trakem2.transform.RigidModel2D"

    def __init__(self, r=0.0, delta=np.array([0, 0])):
        self.set(r, delta)

    def set(self, r, delta):
        self.cos_val = np.cos(r)
        self.sin_val = np.sin(r)
        self.delta = np.array(delta)

    def apply(self, p):
        """
        Returns a new 2D point(s) after applying the transformation on the given point(s) p
        """
        if len(p.shape) == 1: # a single 2D point
            return np.array([
                self.cos_val * p[0] - self.sin_val * p[1],
                self.sin_val * p[0] + self.cos_val * p[1]]) + self.delta
        elif len(p.shape) == 2: # A list of 2D points
            return np.vstack([
                    np.array([self.cos_val * p_i[0] - self.sin_val * p_i[1],
                              self.sin_val * p_i[0] + self.cos_val * p_i[1]]) + self.delta
                for p_i in p])
        raise RuntimeError, "Invalid points input"

    def apply_special(self, p):
        pts = np.atleast_2d(p)
        return np.dot([[self.cos_val, -self.sin_val],
                       [self.sin_val, self.cos_val]],
                       pts.T).T + np.asarray(self.delta).reshape((1, 2))

    def to_str(self):
        return "R={}, T={}".format(np.arccos(self.cos_val), self.delta)

    def to_modelspec(self):
        return {
                "className" : self.class_name,
                "dataString" : "{} {}".format(np.arccos(self.cos_val), ' '.join([str(float(x)) for x in self.delta]))
            }

    def set_from_modelspec(self, s):
        splitted = s.split()
        r = float(splitted[0])
        self.cos_val = np.cos(r)
        self.sin_val = np.sin(r)
        self.delta = np.array([float(d) for d in splitted[1:]])

    def get_matrix(self):
        return np.vstack([
            [self.cos_val, -self.sin_val, self.delta[0]],
            [self.sin_val, self.cos_val, self.delta[1]],
            [0, 0, 1]
            ])

    def fit(self, X, y):
        """
        A non-weighted fitting of a collection of 2D points in X to a collection of 2D points in y.
        X and y are assumed to be arrays of 2D points of the same shape.
        """
        assert(X.shape[0] >= 2) # the minimal number of of matches for a 2d rigid transformation

        pc = np.mean(X, axis=0)
        qc = np.mean(y, axis=0)

        delta_c = pc - qc
        # dx = pc[0] - qc[0]
        # dy = pc[1] - qc[1]

        cosd = 0.0
        sind = 0.0
        delta1 = X - pc
        # delta2 = y - qc + np.array([dx, dy])
        delta2 = y - qc + delta_c

        # for xy1, xy2 in zip(delta1, delta2):
        #     sind += xy1[0] * xy2[1] - xy1[1] * xy2[0]
        #     cosd += xy1[0] * xy2[0] + xy1[1] * xy2[1]
        sind = np.sum(delta1[:,0] * delta2[:,1] - delta1[:,1] * delta2[:,0])
        cosd = np.sum(delta1[:,0] * delta2[:,0] + delta1[:,1] * delta2[:,1])
        norm = np.sqrt(cosd * cosd + sind * sind)
        if norm < 0.0001:
            # print "normalization may be invalid, skipping fitting"
            return False
        cosd /= norm
        sind /= norm

        self.cos_val = cosd
        self.sin_val = sind
        self.delta[0] = qc[0] - cosd * pc[0] + sind * pc[1]
        self.delta[1] = qc[1] - sind * pc[0] - cosd * pc[1]
        return True



class SimilarityModel(AbstractAffineModel):
    MIN_MATCHES_NUM = 2
    class_name = "mpicbg.trakem2.transform.SimilarityModel2D"

    def __init__(self, s=0.0, delta=np.array([0, 0])):
        self.set(s, delta)

    def set(self, s, delta):
        self.scos_val = np.cos(s)
        self.ssin_val = np.sin(s)
        self.delta = np.array(delta)

    def apply(self, p):
        """
        Returns a new 2D point(s) after applying the transformation on the given point(s) p
        """
        if len(p.shape) == 1: # a single 2D point
            return np.array([
                self.scos_val * p[0] - self.ssin_val * p[1],
                self.ssin_val * p[0] + self.scos_val * p[1]]) + self.delta
        elif len(p.shape) == 2: # A list of 2D points
            return np.vstack([
                    np.array([self.scos_val * p_i[0] - self.ssin_val * p_i[1],
                              self.ssin_val * p_i[0] + self.scos_val * p_i[1]]) + self.delta
                for p_i in p])
        raise RuntimeError, "Invalid points input"

    def to_str(self):
        return "S={}, T={}".format(np.arccos(self.scos_val), self.delta)

    def to_modelspec(self):
        return {
                "className" : self.class_name,
                "dataString" : "{} {} {}".format(self.scos_val, self.ssin_val, ' '.join([str(float(x)) for x in self.delta]))
            }

    def set_from_modelspec(self, s):
        splitted = s.split()
        r = float(splitted[0])
        self.scos_val = np.cos(r)
        self.ssin_val = np.sin(r)
        self.delta = np.array([float(d) for d in splitted[1:]])

    def get_matrix(self):
        return np.vstack(
            np.array([self.scos_val, -self.ssin_val, self.delta[0]]),
            np.array([self.ssin_val, self.scos_val, self.delta[1]]),
            np.array([0, 0, 1])
            )

    def fit(self, X, y):
        """
        A non-weighted fitting of a collection of 2D points in X to a collection of 2D points in y.
        X and y are assumed to be arrays of 2D points of the same shape.
        """
        assert(X.shape[0] >= 2) # the minimal number of of matches for a 2d rigid transformation

        pc = np.mean(X, axis=0)
        qc = np.mean(y, axis=0)

        delta_c = pc - qc
        # dx = pc[0] - qc[0]
        # dy = pc[1] - qc[1]

        scosd = 0.0
        ssind = 0.0
        delta1 = X - pc
        # delta2 = y - qc + np.array([dx, dy])
        delta2 = y - qc + delta_c

        norm = 0.0
        for xy1, xy2 in zip(delta1, delta2):
            ssind += xy1[0] * xy2[1] - xy1[1] * xy2[0]
            scosd += xy1[0] * xy2[0] + xy1[1] * xy2[1]
            norm += xy1[0] ** 2 + xy1[1] ** 2
        if norm < 0.0001:
            # print "normalization may be invalid, skipping fitting"
            return False
        scosd /= norm
        ssind /= norm

        self.scos_val = scosd
        self.ssin_val = ssind
        self.delta[0] = qc[0] - scosd * pc[0] + ssind * pc[1]
        self.delta[1] = qc[1] - ssind * pc[0] - scosd * pc[1]
        return True




class AffineModel(AbstractAffineModel):
    MIN_MATCHES_NUM = 3
    class_name = "mpicbg.trakem2.transform.AffineModel2D"

    def __init__(self, m=np.eye(3)):
        """m is a 3x3 matrix"""
        self.set(m)

    def set(self, m):
        """m is a 3x3 matrix"""
        # make sure that this a 3x3 matrix
        m = np.array(m)
        if m.shape != (3, 3):
            raise RuntimeError, "Error when parsing the given affine matrix, should be of size 3x3"
        self.m = m

    def apply(self, p):
        """
        Returns a new 2D point(s) after applying the transformation on the given point(s) p
        """
        if len(p.shape) == 1: # a single 2D point
            return np.dot(self.m, np.append(p, [1]))[:2]
        elif len(p.shape) == 2: # A list of 2D points
            return np.vstack([
                    np.dot(self.m, np.append(p_i, [1]))[:2]
                for p_i in p])
        raise RuntimeError, "Invalid points input"

    # def apply(self, p):
    #     # Check if p is a single 2D point or a list/array of 2D points
    #     if len(p.shape) == 1: # a single 2D point
    #         return np.dot(self.m, p)
    #     elif len(p.shape) == 2: # A list of 2D points
    #         return np.vstack([np.dot(self.m, p_i) for p_i in p])
    #     raise RuntimeError, "Invalid points input"

    def apply_special(self, p):
        pts = np.atleast_2d(p)
        return np.dot(self.m[:2,:2],
                       pts.T).T + np.asarray(self.m.T[2][:2]).reshape((1, 2))

    def to_str(self):
        return "M={}".format(self.m)

    def to_modelspec(self):
        return {
                "className" : self.class_name,
                # keeping it in the Fiji model format
                "dataString" : "{}".format(' '.join([str(float(x)) for x in self.m[:2].T.flatten()]))
            }

    def set_from_modelspec(self, s):
        splitted = s.split()
        # The input is 6 numbers that correspond to m00 m10 m01 m11 m02 m12
        self.m = np.vstack(
            np.array([float(d) for d in splitted[0::2]]),
            np.array([float(d) for d in splitted[1::2]]),
            np.array([0.0, 0.0, 1.0])
            )

    def get_matrix(self):
        return self.m

    def fit(self, X, y):
        """
        A non-weighted fitting of a collection of 2D points in X to a collection of 2D points in y.
        X and y are assumed to be arrays of 2D points of the same shape.
        """
        assert(X.shape[0] >= 2) # the minimal number of of matches for a 2d rigid transformation

        pc = np.mean(X, axis=0)
        qc = np.mean(y, axis=0)


        delta1 = X - pc
        delta2 = y - qc

        a00 = np.sum(delta1[:,0] * delta1[:,0])
        a01 = np.sum(delta1[:,0] * delta1[:,1])
        a11 = np.sum(delta1[:,1] * delta1[:,1])
        b00 = np.sum(delta1[:,0] * delta2[:,0])
        b01 = np.sum(delta1[:,0] * delta2[:,1])
        b10 = np.sum(delta1[:,1] * delta2[:,0])
        b11 = np.sum(delta1[:,1] * delta2[:,1])

        det = a00 * a11 - a01 * a01

        if det == 0:
            # print "determinant is 0, skipping fitting"
            return False

        m00 = (a11 * b00 - a01 * b10) / det
        m01 = (a00 * b10 - a01 * b00) / det
        m10 = (a11 * b01 - a01 * b11) / det
        m11 = (a00 * b11 - a01 * b01) / det

        self.m = np.array([
                [m00, m01, qc[0] - m00 * pc[0] - m01 * pc[1]],
                [m10, m11, qc[1] - m10 * pc[0] - m11 * pc[1]],
                [0.0, 0.0, 1.0]
            ])
        return True


    # def fit(self, X, y):
    #     """
    #     A non-weighted fitting of a collection of 2D points in X to a collection of 2D points in y.
    #     X and y are assumed to be arrays of 2D points of the same shape.
    #     """
    #     fp = X
    #     tp = y
    #     assert(fp.shape[0] >= 3)

    #     """ taken from: http://www.janeriksolem.net/2009/06/affine-transformations-and-warping.html
    #     find H, affine transformation, such that 
    #     tp is affine transf of fp"""

    #     if fp.shape != tp.shape:
    #         raise RuntimeError, 'number of points do not match'

    #     #condition points
    #     #-from points-
    #     m = np.mean(fp[:2], axis=1)
    #     maxstd = max(np.std(fp[:2], axis=1))
    #     C1 = np.diag([1 / maxstd, 1 / maxstd, 1]) 
    #     C1[0][2] = -m[0] / maxstd
    #     C1[1][2] = -m[1] / maxstd
    #     fp_cond = np.dot(C1, fp)

    #     #-to points-
    #     m = np.mean(tp[:2], axis=1)
    #     C2 = C1.copy() #must use same scaling for both point sets
    #     C2[0][2] = -m[0] / maxstd
    #     C2[1][2] = -m[1] / maxstd
    #     tp_cond = np.dot(C2, tp)

    #     #conditioned points have mean zero, so translation is zero
    #     A = np.concatenate((fp_cond[:2], tp_cond[:2]), axis=0)
    #     U,S,V = np.linalg.svd(A.T)

    #     #create B and C matrices as Hartley-Zisserman (2:nd ed) p 130.
    #     tmp = V[:2].T
    #     B = tmp[:2]
    #     C = tmp[2:4]

    #     tmp2 = np.concatenate((np.dot(C, np.linalg.pinv(B)), np.zeros((2, 1))), axis=1) 
    #     H = np.vstack((tmp2, [0, 0, 1]))

    #     #decondition
    #     H = np.dot(np.linalg.inv(C2), np.dot(H, C1))

    #     self.m = H / H[2][2]
    #     return True



class RestrictedMovingLeastSquaresTransform2(AbstractModel):
    class_name = "mpicbg.trakem2.transform.RestrictedMovingLeastSquaresTransform2"

    def __init__(self, radius=None, point_map=None):
        assert((radius is None and point_map is None) or (radius is not None and point_map is not None))
        self.radius = radius
        self.point_map = point_map
        self.interpolator = None

    def apply(self, p):
        return None
#        self.compute_interpolator()
#        
#        self.compute_affine_transforms()
#
#        # Find an index of a simplex that contains p
#        simplex_index = self.triang.find_simplex(p)[0]
#        assert(simplex_index != -1)
#
#        # compute the barycentric weights for point p in the simplex
#        b = self.triang.transform[simplex_index, :2].dot(p - self.triang.transform[simplex_index, 2])
#        bary = np.c_[b, 1 - b.sum(axis=1)]
#
#        # Compute the weighted average of the affine transformations of the simplex vertices
#        simplex = self.triang.simplices[simplex_index]
#        final_affine = np.average(self.point_avg_affine[simplex], axis=0, weights=bary)
#
#        return np.dot(final_affine[:2,:2], p.T).T + np.asarray(final_affine.T[2][:2]).reshape((1, 2))
 
    def apply_special(self, pts):
        return None
#        # Computes the affine transformation for many points
#        self.compute_affine_transforms()
#
#        # Find the indices of all simplices, for each of the points in pts
#        simplex_indices = self.triang.find_simplex(p)
#        assert not np.any(simplex_indices == -1)
#
#        # http://codereview.stackexchange.com/questions/41024/faster-computation-of-barycentric-coordinates-for-many-points
#        X = self.triang.transform[simplex_indices, :2]
#        Y = pts - self.triang.transform[simplex_indices, 2]
#        b = np.einsum('ijk,ik->ij', X, Y)
#        barys = np.c_[b, 1 - b.sum(axis=1)] # shape: (pts#, 3) --> for each point its 3 barycentric values
#
#        # apply for each point in pts the 3 affine transformations around it
#        pt_indices = self.triangulation.simplices[simplex_indices].astype(np.uint32)
#        all_affine_transfroms = self.point_avg_affine[pt_indices] # shape: (pts#, 3, 2, 3) --> three 2x3 affine transform matrices for each point in pts
#        
#        # TODO - need to improve speed
#        res = np.array((pts.shape[0], 3, 2), dtype=np.float) # will store for each point the three coordinates after affine transform
#        for p_i, p in enumerate(pts):
#            pt_affine_transforms = all_affine_transforms[p_i] # three affine transfrom matrices (2x3)
#            for a_i, pt_affine_transform in enumerate(pt_affine_transforms):
#                res[p_i][a_i] = np.dot(pt_affine_transform[:2,:2], p.T).T + np.asarray(pt_affine_transform.T[2][:2]).reshape((1, 2))
#        
#        global_res = np.array((pts.shape[0], 2), dtype=np.float)
#        for res_i, pt_affines in enumerate(res):
#            global_res[res_i] = np.average(res_i, axis=0, weights=barys[res_i])
#
#        return global_res

    def set_from_modelspec(self, s):
        data = s.split()
        assert(data[0] == 'affine')
        assert(data[1] == '2')
        assert(data[2] == '2.0')
        self.radius = float(data[3])
        points_data = data[4:] # format is: p1_src_x p1_src_y p1_dest_x p1_dest_y 1.0 ... (1.0 is the weight of the match)
        src = np.array(
                        [np.array(points_data[0::5], dtype=np.float32),
                         np.array(points_data[1::5], dtype=np.float32)]
                      ).T
        dest = np.array(
                        [np.array(points_data[2::5], dtype=np.float32),
                         np.array(points_data[3::5], dtype=np.float32)]
                      ).T
        self.point_map = (src, dest)
        self.interpolator = None

    def get_point_map(self):
        return self.point_map

#    def compute_interpolator(self):
#        """Uses griddata to interpolate all the pixels in the output window"""
#        if self.interpolator is not None:
#            return
#
#        # Compute the interpolator using scipy.interpolate.LinearNDInterpolator
#        self.interpolator = LinearNDInterpolator(#TODO)

#    def compute_affine_transforms(self):
#        """Computes an average affine transformation for each point from the source points,
#           using the affine trasnformations of all neighboring triangles"""
#        if self.point_avg_affine is not None:
#            return
#
#        # Create the triangulation of the source points
#        src = self.point_map[0]
#        dest = self.point_map[1]
#        self.triang = Delaunay(src)
#        # Compute a per simplex affine transformation
#        simplex_transforms = np.array((len(self.triang.simplices), 2, 3), dtype=np.float)
#        # Also, set a per-point list of all simplices around it
#        point_neighboring_simplices = [] * src.shape[0]
#        for simplex_i, simplex in enumerate(self.triang.simplices):
#            simplex_src_points = src[simplex]
#            simplex_dest_points = dest[simplex]
#            affine_transform = cv2.getAffineTransform(simplex_src_points, simplex_dest_points)
#            simplex_transforms[simplex_i] = affine_transform
#            # add the simplex as a neighbor to all of its points
#            point_neighboring_simplices.append(simplex_i)
#            
#        # Compute a per-point average affine transform
#        self.point_avg_affine = np.array((src.shape[0], 2, 3), dtype=np.float)
#        for p_i, _ in enumerate(src):
#            self.point_avg_affine[p_i] = np.mean(simplex_transforms[np.array(point_neighboring_simplices[p_i])], axis=0)


class Transforms(object):
    transformations = [ TranslationModel(), RigidModel(), SimilarityModel(), AffineModel() ]
    transforms_classnames = {
        TranslationModel.class_name : TranslationModel(),
        RigidModel.class_name : RigidModel(),
        SimilarityModel.class_name : SimilarityModel(),
        AffineModel.class_name : AffineModel(),
        RestrictedMovingLeastSquaresTransform2.class_name : RestrictedMovingLeastSquaresTransform2(),
        }

    @classmethod
    def create(cls, model_type_idx):
        return copy.deepcopy(cls.transformations[model_type_idx])

    @classmethod
    def from_tilespec(cls, ts_transform):
        transform = copy.deepcopy(cls.transforms_classnames[ts_transform["className"]])
        transform.set_from_modelspec(ts_transform["dataString"])
        return transform


