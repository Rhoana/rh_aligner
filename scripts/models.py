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
        X2 = copy.deepcopy(X)
        for i, p in enumerate(X2):
            X2[i] = self.apply(p)
        dists = np.sqrt(np.sum((y - X2) ** 2, axis=1))
        #print "dists", dists
        good_dists = dists[dists < epsilon]
        accepted_ratio = float(good_dists.shape[0]) / X2.shape[0]

        # The transformation does not adhere to the wanted values, give it a very low score
        if good_dists.shape[0] < min_num_inlier or accepted_ratio < min_inlier_ratio:
            return -1, None, -1

        return accepted_ratio, dists < epsilon, np.mean(good_dists)

    def apply(self, p):
        raise RuntimeError, "Not implemented, but probably should be"


    def fit(self, X, y):
        raise RuntimeError, "Not implemented, but probably should be"

    def set_from_str(self, s):
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
    MIN_MATCHES_NUM = 1

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

    def to_str(self):
        return "T={}".format(self.delta)

    def set_from_str(self, s):
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

    def __init__(self, r=0.0, delta=np.array([0, 0])):
        self.set(r, delta)

    def set(self, r, delta):
        self.cos_val = np.cos(r)
        self.sin_val = np.sin(r)
        self.delta = np.array(delta)

    # def apply(self, p):
    #     """
    #     Returns a new 2D point(s) after applying the transformation on the given point(s) p
    #     """
    #     # TODO - Check if p is a single 2D point or a list/array of 2D points
    #     return np.array([
    #         self.cos_val * p[0] - self.sin_val * p[1],
    #         self.sin_val * p[0] + self.cos_val * p[1]]) + self.delta

    def to_str(self):
        return "R={}, T={}".format(np.arccos(self.cos_val), self.delta)

    def set_from_str(self, s):
        splitted = s.split()
        r = float(splitted[0])
        self.cos_val = np.cos(r)
        self.sin_val = np.sin(r)
        self.delta = np.array([float(d) for d in splitted[1:]])

    def get_matrix(self):
        return np.vstack([
            np.array([self.cos_val, -self.sin_val, self.delta[0]]),
            np.array([self.sin_val, self.cos_val, self.delta[1]]),
            np.array([0, 0, 1])
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

        for xy1, xy2 in zip(delta1, delta2):
            sind += xy1[0] * xy2[1] - xy1[1] * xy2[0]
            cosd += xy1[0] * xy2[0] + xy1[1] * xy2[1]
        norm = np.sqrt(cosd * cosd + sind * sind)
        if norm < 0.0001:
            print "normalization may be invalid, skipping fitting"
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

    def __init__(self, s=0.0, delta=np.array([0, 0])):
        self.set(s, delta)

    def set(self, s, delta):
        self.scos_val = np.cos(s)
        self.ssin_val = np.sin(s)
        self.delta = np.array(delta)


    def to_str(self):
        return "S={}, T={}".format(np.arccos(self.scos_val), self.delta)

    def set_from_str(self, s):
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
            print "normalization may be invalid, skipping fitting"
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

    def __init__(self, m=np.eye(3)):
        """m is a 3x3 matrix"""
        self.set(m)

    def set(self, m):
        """m is a 3x3 matrix"""
        self.m = m

    # def apply(self, p):
    #     # Check if p is a single 2D point or a list/array of 2D points
    #     if len(p.shape) == 1: # a single 2D point
    #         return np.dot(self.m, p)
    #     elif len(p.shape) == 2: # A list of 2D points
    #         return np.vstack([np.dot(self.m, p_i) for p_i in p])
    #     raise RuntimeError, "Invalid points input"

    def to_str(self):
        return "M={}".format(self.m)

    def set_from_str(self, s):
        splitted = s.split()
        self.m = np.vstack(
            np.array([float(d) for d in splitted[:3]]),
            np.array([float(d) for d in splitted[3:]])
            )

    def get_matrix(self):
        return m

    def fit(self, X, y):
        """
        A non-weighted fitting of a collection of 2D points in X to a collection of 2D points in y.
        X and y are assumed to be arrays of 2D points of the same shape.
        """
        fp = X
        tp = y
        assert(fp.shape[0] >= 3)

        """ taken from: http://www.janeriksolem.net/2009/06/affine-transformations-and-warping.html
        find H, affine transformation, such that 
        tp is affine transf of fp"""

        if fp.shape != tp.shape:
            raise RuntimeError, 'number of points do not match'

        #condition points
        #-from points-
        m = mean(fp[:2], axis=1)
        maxstd = max(std(fp[:2], axis=1))
        C1 = diag([1 / maxstd, 1 / maxstd, 1]) 
        C1[0][2] = -m[0] / maxstd
        C1[1][2] = -m[1] / maxstd
        fp_cond = dot(C1, fp)

        #-to points-
        m = mean(tp[:2], axis=1)
        C2 = C1.copy() #must use same scaling for both point sets
        C2[0][2] = -m[0] / maxstd
        C2[1][2] = -m[1] / maxstd
        tp_cond = dot(C2, tp)

        #conditioned points have mean zero, so translation is zero
        A = concatenate((fp_cond[:2], tp_cond[:2]), axis=0)
        U,S,V = linalg.svd(A.T)

        #create B and C matrices as Hartley-Zisserman (2:nd ed) p 130.
        tmp = V[:2].T
        B = tmp[:2]
        C = tmp[2:4]

        tmp2 = concatenate((dot(C, linalg.pinv(B)), zeros((2, 1))), axis=1) 
        H = vstack((tmp2, [0, 0, 1]))

        #decondition
        H = dot(linalg.inv(C2), dot(H, C1))

        self.m = H / H[2][2]
        return True




class Transforms(object):
    transformations = [ TranslationModel(), RigidModel(), SimilarityModel(), AffineModel() ]
    transforms_classnames = {
        "mpicbg.trakem2.transform.TranslationModel2D": TranslationModel(),
        "mpicbg.trakem2.transform.RigidModel2D": RigidModel(),
        "mpicbg.trakem2.transform.SimilarityModel2D": SimilarityModel(),
        "mpicbg.trakem2.transform.AffineModel2D": AffineModel(),
        }

    @classmethod
    def create(cls, model_type_idx):
        return copy.deepcopy(cls.transformations[model_type_idx])

    @classmethod
    def from_tilespec(cls, ts_transform):
        transform = copy.deepcopy(cls.transforms_classnames[ts_transform["className"]])
        transform.set_from_str(ts_transform["dataString"])
        return transform


