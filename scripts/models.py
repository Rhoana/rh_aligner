import numpy as np
import copy



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
        return np.dot(np.eye(3), np.append(self.delta, [1]))

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
        return m

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




class Transforms(object):
    transformations = [ TranslationModel(), RigidModel(), SimilarityModel(), AffineModel() ]
    transforms_classnames = {
        TranslationModel.class_name : TranslationModel(),
        RigidModel.class_name : RigidModel(),
        SimilarityModel.class_name : SimilarityModel(),
        AffineModel.class_name : AffineModel(),
        }

    @classmethod
    def create(cls, model_type_idx):
        return copy.deepcopy(cls.transformations[model_type_idx])

    @classmethod
    def from_tilespec(cls, ts_transform):
        transform = copy.deepcopy(cls.transforms_classnames[ts_transform["className"]])
        transform.set_from_modelspec(ts_transform["dataString"])
        return transform


