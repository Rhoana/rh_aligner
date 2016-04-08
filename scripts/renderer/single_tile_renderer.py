# A class that takes a single image, applies transformations (both affine and non-affine), and renders it
# (and possibly a pixel-mask to tell which pixels are coming from the image).
# Assumption: there is only one non-affine transformation. TODO - get rid of this assumption
# The class will only load the image when the render function is called (lazy evaluation).
# Consecutive affine transformations will be condensed into a single transformation
import cv2
import numpy as np
import math
import scipy.interpolate as spint
import scipy.spatial.qhull as qhull
from scipy.spatial import ConvexHull

class SingleTileRenderer:

    def __init__(self, img_path, width, height, compute_mask=False, compute_distances=True):
        self.img_path = img_path
        self.width = width
        self.height = height
        # Starting with a single identity affine transformation
        self.pre_non_affine_transform = np.eye(3)[:2]
        self.non_affine_transform = None
        self.post_non_affine_transform = np.eye(3)[:2]
        self.compute_mask = compute_mask
        self.mask = None
        self.compute_distances = compute_distances
        self.weights = None
        self.bbox = [0, width - 1, 0, height - 1]
        self.shape = (width, height)
        # Store the pixel locations (x,y) of the surrounding polygon of the image
        self.surrounding_polygon = np.array([[0., 0.], [width - 1., 0.], [width - 1., height - 1.], [0., height - 1.]])
        self.start_point = (0, 0) # If only affine is used then this is always (bbox[0], bbox[2]), with affine it can be different

        # Save for caching
        self.already_rendered = False

    def add_transformation(self, model):
        if model.is_affine():
            new_model_matrix = model.get_matrix()
            # Need to add the transformation either to the pre_non_affine or the post_non_affine
            if self.non_affine_transform is None:
                cur_transformation = self.pre_non_affine_transform
            else:
                cur_transformation = self.post_non_affine_transform

            # Compute the new transformation (multiply from the left)
            new_transformation = np.dot(new_model_matrix, np.vstack((cur_transformation, [0., 0., 1.])))[:2]

            if self.non_affine_transform is None:
                self.pre_non_affine_transform = new_transformation
            else:
                self.post_non_affine_transform = new_transformation

            # Apply the model to each of the surrounding polygon locations
            self.surrounding_polygon = model.apply_special(self.surrounding_polygon)
        else:
            # Non-affine transformation
            self.non_affine_transform = model

            # TODO - need to see if this returns a sufficient bounding box for the reverse transformation
            # Find the new surrounding polygon locations
            # using a forward transformation from the boundaries of the source image to the destination
            boundary1 = np.array([[float(p), 0.] for p in np.arange(self.width)])
            boundary2 = np.array([[float(p), float(self.height - 1)] for p in np.arange(self.width)])
            boundary3 = np.array([[0., float(p)] for p in np.arange(self.height)])
            boundary4 = np.array([[float(self.width - 1), float(p)] for p in np.arange(self.height)])
            boundaries = np.concatenate((boundary1, boundary2, boundary3, boundary4))
            boundaries = np.dot(self.pre_non_affine_transform[:2, :2], boundaries.T).T + self.pre_non_affine_transform[:, 2].reshape((1, 2))

            src_points, dest_points = model.get_point_map()
            cubic_interpolator = spint.CloughTocher2DInterpolator(src_points, dest_points)
            self.surrounding_polygon = cubic_interpolator(boundaries)

            # Find the new surrounding polygon locations (using the destination points of the non affine transformation)
            #dest_points = model.get_point_map()[1]
            #hull = ConvexHull(dest_points)
            #self.surrounding_polygon = dest_points[hull.vertices]

        # Update bbox and shape according to the new borders
        self.bbox, self.shape = compute_bbox_and_shape(self.surrounding_polygon)

        # Remove any rendering
        self.already_rendered = False
        self.img = None

    def get_bbox(self):
        return self.bbox

#    def contains_point(self, p):
#        """Return True if the point is inside the image boundaries (bounding polygon)."""

#    def get_min_distance(self, points):
#        """Returns a list of minimal distances between each of the given points and any of the image boundaries (bounding polygon).
#           Assumes that the given points are inside the bounding polygon."""
#        #assert(p.shape == (2,))
#        # Get the normals of each line, and compute the distance between the point and the normal
#        # Based on method 2 (but for 2D) from: http://www.qc.edu.hk/math/Advanced%20Level/Point_to_line.htm
#        denominators = [np.linalg.norm(self.corners[i] - self.corners[(i + 1) % len(self.corners)]) for i in range(len(self.corners))]
#        self_normals = get_normals(self.corners)
#        if points.shape == (2,): # A single point
#            dist = np.min([np.linalg.norm(np.dot(n, points - c)) / denom
#                        for c, n, denom in zip(self.corners, self_normals, denominators)])
#            return dist
#        else: # multiple points
#            dists = [np.min([np.linalg.norm(np.dot(n, p - c)) / denom
#                        for c, n, denom in zip(self.corners, self_normals, denominators)])
#                            for p in points]
#            return dists

#    def is_overlapping(self, other_tile):
#        """Uses Separating Axes Theorem (http://www.dyn4j.org/2010/01/sat/) in order to decide
#           whether the the current transformed tile and the other transformed tile are overlapping"""
#        # Fetch the normals of each tile
#        self_normals = get_normals(self.corners)
#        other_normals = get_normals(other_tile.corners)
#        # Check all edges of self against the normals of the other tile
#        if not check_normals_side(self.corners, self_normals, other_tile.corners):
#            return True
#        # Check all edges of the other tile against the normals of self
#        if not check_normals_side(other_tile.corners, other_normals, self.corners):
#            return True
#        return False

    def render(self):
        """Returns the rendered image (after transformation), and the start point of the image in global coordinates"""
        if self.already_rendered:
            return self.img, self.start_point

        img = cv2.imread(self.img_path, 0)
        self.start_point = np.array([self.bbox[0], self.bbox[2]]) # may be different for non-affine result

        if self.non_affine_transform is None:
            # If there wasn't a non-affine transformation, we only need to apply an affine transformation
            adjusted_transform = self.pre_non_affine_transform[:2].copy()
            adjusted_transform[0][2] -= self.bbox[0]
            adjusted_transform[1][2] -= self.bbox[2]

            self.img = cv2.warpAffine(img, adjusted_transform, self.shape, flags=cv2.INTER_AREA)
            if self.compute_mask:
                mask_img = np.ones(img.shape)
                self.mask = cv2.warpAffine(mask_img, adjusted_transform, self.shape, flags=cv2.INTER_AREA)
                self.mask[self.mask > 0] = 1
            if self.compute_distances:
                # The initial weights for each pixel is the minimum from the image boundary
                grid = np.mgrid[0:self.height, 0:self.width]
                weights_img = np.minimum(
                                    np.minimum(grid[0], self.height - 1 - grid[0]),
                                    np.minimum(grid[1], self.width - 1 - grid[1])
                                ).astype(np.float32)
                self.weights = cv2.warpAffine(weights_img, adjusted_transform, self.shape, flags=cv2.INTER_AREA)

        else:
            # Apply a reverse pre affine transformation on the source points of the non-affine transformation,
            # and a post affine transformation on the destination points
            src_points, dest_points = self.non_affine_transform.get_point_map()
            inverted_pre = np.linalg.inv(np.vstack([self.pre_non_affine_transform, [0., 0., 1.]]))[:2]
            src_points = np.dot(inverted_pre[:2, :2], src_points.T).T + inverted_pre[:, 2].reshape((1, 2))
            dest_points = np.dot(self.post_non_affine_transform[:2, :2], dest_points.T).T + self.post_non_affine_transform[:, 2].reshape((1, 2))

            # Move the destination points to start at (0, 0) --> less rendering
            dest_points = dest_points - np.array([self.bbox[0], self.bbox[2]])

            # Set the target grid using the shape
            out_grid_x, out_grid_y = np.mgrid[0:self.shape[0], 0:self.shape[1]]

            # TODO - is there a way to further restrict the target grid size, and speed up the interpolation?
            # Use griddata to interpolate all the destination points
            #out_grid_z = spint.griddata(dest_points, src_points, (out_grid_x, out_grid_y), method='linear', fill_value=-1.)
            out_grid_z = spint.griddata(dest_points, src_points, (out_grid_x, out_grid_y), method='cubic', fill_value=-1.)

            map_x = np.append([], [ar[:,0] for ar in out_grid_z]).reshape(self.shape[0], self.shape[1]).astype('float32')
            map_y = np.append([], [ar[:,1] for ar in out_grid_z]).reshape(self.shape[0], self.shape[1]).astype('float32')
            # find all rows and columns that are mapped before or after the boundaries of the source image, and remove them
            map_valid_cells = np.where((map_x >= 0.) & (map_x < float(self.width)) & (map_y >= 0.) & (map_y < float(self.height)))
            min_col_row = np.min(map_valid_cells, axis=1)
            max_col_row = np.max(map_valid_cells, axis=1)
            map_x = map_x[min_col_row[0]:max_col_row[0], min_col_row[1]:max_col_row[1]]
            map_y = map_y[min_col_row[0]:max_col_row[0], min_col_row[1]:max_col_row[1]]

            # remap the source points to the destination points
            self.img = cv2.remap(img, map_x, map_y, cv2.INTER_CUBIC).T
            self.start_point = self.start_point + min_col_row

            # Add mask and weights computation
            if self.compute_mask:
                mask_img = np.ones(img.shape)
                self.mask = cv2.remap(mask_img, map_x, map_y, cv2.INTER_CUBIC).T
                self.mask[self.mask > 0] = 1
            if self.compute_distances:
                # The initial weights for each pixel is the minimum from the image boundary
                grid = np.mgrid[0:self.height, 0:self.width]
                weights_img = np.minimum(
                                    np.minimum(grid[0], self.height - 1 - grid[0]),
                                    np.minimum(grid[1], self.width - 1 - grid[1])
                                ).astype(np.float32)
                self.weights = cv2.remap(weights_img, map_x, map_y, cv2.INTER_CUBIC).T
                self.weights[self.weights < 0] = 0



        self.already_rendered = True
        return self.img, self.start_point

    def fetch_mask(self):
        assert(self.compute_mask)

        if not self.already_rendered:
            self.render()

        return self.mask, (self.bbox[0], self.bbox[2])

    def crop(self, from_x, from_y, to_x, to_y):
        """Returns the cropped image, its starting point, and the cropped mask (if the mask was computed).
           The given coordinates are specified using world coordinates."""
        # find the overlapping area of the given coordinates and the transformed tile
        overlapping_area = [max(from_x, self.bbox[0]), min(to_x, self.bbox[1]), max(from_y, self.bbox[2]), min(to_y, self.bbox[3])]
        overlapping_width = overlapping_area[1] - overlapping_area[0] + 1
        overlapping_height = overlapping_area[3] - overlapping_area[2] + 1
        if overlapping_width <= 0 or overlapping_height <= 0:
            # No overlap between the area and the tile
            return None, None, None

        cropped_mask = None
        # Make sure the image was rendered
        self.render()
        # Check with the actual image bounding box (may be different because of the non-affine transformation)
        actual_bbox = [self.start_point[0], self.start_point[0] + self.img.shape[1], self.start_point[1], self.start_point[1] + self.img.shape[0]]
        overlapping_area = [max(from_x, actual_bbox[0]), min(to_x, actual_bbox[1]), max(from_y, actual_bbox[2]), min(to_y, actual_bbox[3])]
        overlapping_width = overlapping_area[1] - overlapping_area[0] + 1
        overlapping_height = overlapping_area[3] - overlapping_area[2] + 1
        if overlapping_width <= 0 or overlapping_height <= 0:
            # No overlap between the area and the tile
            return None, None, None


        cropped_img = self.img[overlapping_area[2] - actual_bbox[2]:overlapping_area[3] - actual_bbox[2] + 1,
                               overlapping_area[0] - actual_bbox[0]:overlapping_area[1] - actual_bbox[0] + 1]
        if self.compute_mask:
            cropped_mask = self.mask[overlapping_area[2] - actual_bbox[2]:overlapping_area[3] - actual_bbox[2] + 1,
                                     overlapping_area[0] - actual_bbox[0]:overlapping_area[1] - actual_bbox[0] + 1]
        # Take only the parts that are overlapping
        return cropped_img, (overlapping_area[0], overlapping_area[2]), cropped_mask

    def crop_with_distances(self, from_x, from_y, to_x, to_y):
        """Returns the cropped image, its starting point, and the cropped image L1 distances of each pixel inside the image from the edge
           of the rendered image (if the mask was computed).
           The given coordinates are specified using world coordinates."""
        # find the overlapping area of the given coordinates and the transformed tile
        overlapping_area = [max(from_x, self.bbox[0]), min(to_x, self.bbox[1]), max(from_y, self.bbox[2]), min(to_y, self.bbox[3])]
        overlapping_width = overlapping_area[1] - overlapping_area[0] + 1
        overlapping_height = overlapping_area[3] - overlapping_area[2] + 1
        if overlapping_width <= 0 or overlapping_height <= 0:
            # No overlap between the area and the tile
            return None, None, None

        cropped_distances = None
        # Make sure the image was rendered
        self.render()
        # Check with the actual image bounding box (may be different because of the non-affine transformation)
        actual_bbox = [self.start_point[0], self.start_point[0] + self.img.shape[1], self.start_point[1], self.start_point[1] + self.img.shape[0]]
        overlapping_area = [max(from_x, actual_bbox[0]), min(to_x, actual_bbox[1]), max(from_y, actual_bbox[2]), min(to_y, actual_bbox[3])]
        overlapping_width = overlapping_area[1] - overlapping_area[0] + 1
        overlapping_height = overlapping_area[3] - overlapping_area[2] + 1
        if overlapping_width <= 0 or overlapping_height <= 0:
            # No overlap between the area and the tile
            return None, None, None

        cropped_img = self.img[overlapping_area[2] - actual_bbox[2]:overlapping_area[3] - actual_bbox[2] + 1,
                               overlapping_area[0] - actual_bbox[0]:overlapping_area[1] - actual_bbox[0] + 1]
        if self.compute_distances:
            cropped_distances = self.weights[overlapping_area[2] - actual_bbox[2]:overlapping_area[3] - actual_bbox[2] + 1,
                                             overlapping_area[0] - actual_bbox[0]:overlapping_area[1] - actual_bbox[0] + 1]
           
        # Take only the parts that are overlapping
        return cropped_img, (overlapping_area[0], overlapping_area[2]), cropped_distances



    # Helper methods (shouldn't be used from the outside)


def compute_bbox_and_shape(polygon):
    # find the new bounding box
    min_XY = np.min(polygon, axis=0)
    max_XY = np.max(polygon, axis=0)
    # Rounding to avoid float precision errors due to representation
    new_bbox = [int(math.floor(round(min_XY[0], 5))), int(math.ceil(round(max_XY[0], 5))), int(math.floor(round(min_XY[1], 5))), int(math.ceil(round(max_XY[1], 5)))]
    #new_bbox = [int(min_XY[0] + math.copysign(0.5, min_XY[0])), int(max_XY[0] + math.copysign(0.5, max_XY[1])), int(min_XY[1] + math.copysign(0.5, min_XY[1])), int(max_XY[1] + math.copysign(0.5, max_XY[1]))]
    new_shape = (new_bbox[1] - new_bbox[0] + 1, new_bbox[3] - new_bbox[2] + 1)
    return new_bbox, new_shape



#def get_normals(corners):
#    """Given a polygon corners list, returns a list of non-normalized normals for each edge"""
#    edges = [(corners[i] - corners[(i + 1) % len(corners)]) for i in range(len(corners))]
#    normals = [(-e[1], e[0]) for e in edges]
#    return normals

#def check_normals_side(corners1, normals1, corners2):
#    """Checks if all corners2 appear on one side of polygon1"""
#    assert(len(corners1) == len(normals1))
#    for c, n in zip(corners1, normals1):
#        signs2 = [np.sign(np.dot(n, p - c)) for p in corners2]
#        signs2 = [s for s in signs2 if abs(s - 0.) > 0.0001] # remove all +-0.
#        if np.any(signs2 != signs2[0]):
#            return False
#    return True

#def find_per_row_first_last_one(arr):
#    """Given a 2D array (of only 1's in a quadrangle shape, and 0's), for each row find the first and the last occurrance of 1.
#       Returns a 2D array with the same number of rows as arr, and 2 columns with the column-indices
#       of the first and last one. If a row has only 0's, -1 will be returned on both indices"""
#    res = np.full((arr.shape[0], 2), -1, dtype=np.int16)
#    # take the first and last column of arr, and find all 1's
#    arr_T = arr.T
#    first_col_non_zero = np.nonzero(arr_T[0])
#    last_col_non_zero = np.nonzero(arr_T[-1])
#    for r in first_col_non_zero[0]:
#        res[r, 0] = 0
#    for r in last_col_non_zero[0]:
#        res[r, 1] = arr.shape[1] - 1
#    # Now find the positions where the value changes in the middle of the matrix using np.diff
#    nonzero = np.nonzero(np.diff(arr))
#    # nonzero contents for each row, r:
#    #   if nonzero doesn't have a row with r, the row has the same value (either 0 or 1)
#    #   if nonzero has a single row with r, the row changes the value once (either from 0 to 1 or from 1 to 0)
#    #   if nonzero has row r twice, the row changes both from 0 to 1 and then from 1 to 0
#    for r, c in zip(*nonzero):
#        if res[r, 0] > -1:
#            # already updated the left value, or there is a single change from 1 to 0
#            res[r, 1] = c
#        else:
#            res[r, 0] = c + 1
#    return res

# An implementation of scipy's interpolate.griddata broken to 2 parts
# (taken from: http://stackoverflow.com/questions/20915502/speedup-scipy-griddata-for-multiple-interpolations-between-two-irregular-grids)

def interp_weights(xyz, uvw):
    """The initial part of griddata (using linear interpolation) -
       Creates a Delaunay mesh triangulation of the source points,
       each point in the mesh is transformed to the new mesh
       using an interpolation of each point inside a triangle is done using barycentric coordinates"""
    tri = qhull.Delaunay(xyz)
    simplex = tri.find_simplex(uvw)
    vertices = np.take(tri.simplices, simplex, axis=0)
    temp = np.take(tri.transform, simplex, axis=0)
    delta = uvw - temp[:, d]
    bary = np.einsum('njk,nk->nj', temp[:, :d, :], delta)
    return vertices, np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))

def interpolate(values, vtx, wts, fill_value=np.nan):
    """Executes the interpolation step of griddata"""
    ret = np.einsum('nj,nj->n', np.take(values, vtx), wts)
    ret[np.any(wts < 0, axis=1)] = fill_value
    return ret

