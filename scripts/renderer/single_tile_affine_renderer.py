# A class that takes a single image, applies affine transformations, and renders it
# (and possibly a pixel-mask to tell which pixels are coming from the image)
# The class will only load the image when the render function is called (lazy evaluation)
import cv2
import numpy as np
import math

class SingleTileAffineRenderer:

    def __init__(self, img_path, width, height, compute_mask=False, compute_distances=True):
        self.img_path = img_path
        self.width = width
        self.height = height
        self.transform_matrix = np.eye(3)[:2]
        self.compute_mask = compute_mask
        self.mask = None
        self.compute_distances = compute_distances
        self.weights = None
        self.update_img_transformed_corners_and_bbox()

        # Save for caching
        self.already_rendered = False

    def add_transformation(self, transform_matrix):
        assert(transform_matrix.shape == (2, 3))
        self.transform_matrix = np.dot(np.vstack((transform_matrix, [0., 0., 1.])), np.vstack((self.transform_matrix, [0., 0., 1.])))[:2]
        self.update_img_transformed_corners_and_bbox()

        # Remove any rendering
        self.already_rendered = False
        self.img = None

    def get_start_point(self):
        return (self.bbox[0], self.bbox[2])

    def get_bbox(self):
        return self.bbox

#    def contains_point(self, p):
#        """Return True if the point is inside the image boundaries (bounding polygon)."""

    def get_min_distance(self, points):
        """Returns a list of minimal distances between each of the given points and any of the image boundaries (bounding polygon).
           Assumes that the given points are inside the bounding polygon."""
        #assert(p.shape == (2,))
        # Get the normals of each line, and compute the distance between the point and the normal
        # Based on method 2 (but for 2D) from: http://www.qc.edu.hk/math/Advanced%20Level/Point_to_line.htm
        denominators = [np.linalg.norm(self.corners[i] - self.corners[(i + 1) % len(self.corners)]) for i in range(len(self.corners))]
        self_normals = get_normals(self.corners)
        if points.shape == (2,): # A single point
            dist = np.min([np.linalg.norm(np.dot(n, points - c)) / denom
                        for c, n, denom in zip(self.corners, self_normals, denominators)])
            return dist
        else: # multiple points
            dists = [np.min([np.linalg.norm(np.dot(n, p - c)) / denom
                        for c, n, denom in zip(self.corners, self_normals, denominators)])
                            for p in points]
            return dists

    def is_overlapping(self, other_tile):
        """Uses Separating Axes Theorem (http://www.dyn4j.org/2010/01/sat/) in order to decide
           whether the the current transformed tile and the other transformed tile are overlapping"""
        # Fetch the normals of each tile
        self_normals = get_normals(self.corners)
        other_normals = get_normals(other_tile.corners)
        # Check all edges of self against the normals of the other tile
        if not check_normals_side(self.corners, self_normals, other_tile.corners):
            return True
        # Check all edges of the other tile against the normals of self
        if not check_normals_side(other_tile.corners, other_normals, self.corners):
            return True
        return False

    def render(self):
        """Returns the rendered image (after transformation), and the start point of the image in global coordinates"""
        if self.already_rendered:
            return self.img, np.array([self.bbox[0], self.bbox[1]])

        img = cv2.imread(self.img_path, 0)
        adjusted_transform = self.transform_matrix[:2].copy()
        adjusted_transform[0][2] -= self.bbox[0]
        adjusted_transform[1][2] -= self.bbox[2]
        
        self.img = cv2.warpAffine(img, adjusted_transform, self.shape)
        self.already_rendered = True
        if self.compute_mask:
            mask_img = np.ones(img.shape)
            self.mask = cv2.warpAffine(mask_img, adjusted_transform, self.shape)
            self.mask[self.mask > 0] = 1
        if self.compute_distances:
            # The initial weights for each pixel is the minimum from the image boundary
            grid = np.mgrid[0:self.height, 0:self.width]
            weights_img = np.minimum(
                                np.minimum(grid[0], self.height - 1 - grid[0]),
                                np.minimum(grid[1], self.width - 1 - grid[1])
                            ).astype(np.float32)
            self.weights = cv2.warpAffine(weights_img, adjusted_transform, self.shape)
        # Returns the transformed image and the start point
        return self.img, (self.bbox[0], self.bbox[2])

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
        cropped_img = self.img[overlapping_area[2] - self.bbox[2]:overlapping_area[3] - self.bbox[2] + 1,
                               overlapping_area[0] - self.bbox[0]:overlapping_area[1] - self.bbox[0] + 1]
        if self.compute_mask:
            cropped_mask = self.mask[overlapping_area[2] - self.bbox[2]:overlapping_area[3] - self.bbox[2] + 1,
                                     overlapping_area[0] - self.bbox[0]:overlapping_area[1] - self.bbox[0] + 1]
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
        cropped_img = self.img[overlapping_area[2] - self.bbox[2]:overlapping_area[3] - self.bbox[2] + 1,
                               overlapping_area[0] - self.bbox[0]:overlapping_area[1] - self.bbox[0] + 1]
#        if self.compute_mask:
#            cropped_mask = self.mask[overlapping_area[2] - self.bbox[2]:overlapping_area[3] - self.bbox[2] + 1,
#                                     overlapping_area[0] - self.bbox[0]:overlapping_area[1] - self.bbox[0] + 1]
#            # find for each row where is the first and last one
#            rows_ones = find_per_row_first_last_one(cropped_mask)
#            cols_ones = find_per_row_first_last_one(cropped_mask.T)
#            # fill the weights matrix according to the distance of each pixel from it's row and col values
#            grid = np.mgrid[0:cropped_mask.shape[0], 0:cropped_mask.shape[1]]
#            cropped_distances = np.minimum(
#                                    np.minimum(np.abs(grid[0] - cols_ones[:, 0]), np.abs(grid[0] - cols_ones[:, 1])),
#                                    np.minimum(np.abs(grid[1].T - rows_ones[:, 0].T).T, np.abs(grid[1].T - rows_ones[:, 1].T).T)
#                                )
#            # Filter only the weights that are on the mask (using elementwise multiplication)
#            cropped_distances = cropped_distances * cropped_mask

        if self.compute_distances:
            cropped_distances = self.weights[overlapping_area[2] - self.bbox[2]:overlapping_area[3] - self.bbox[2] + 1,
                                             overlapping_area[0] - self.bbox[0]:overlapping_area[1] - self.bbox[0] + 1]
           
        # Take only the parts that are overlapping
        return cropped_img, (overlapping_area[0], overlapping_area[2]), cropped_distances



    # Helper methods (shouldn't be used from the outside)
    def update_img_transformed_corners_and_bbox(self):
        pts = np.array([[0., 0.], [self.width - 1, 0.], [self.width - 1, self.height - 1], [0., self.height - 1]])
        self.corners = np.dot(self.transform_matrix[:2,:2], pts.T).T + np.asarray(self.transform_matrix.T[2][:2]).reshape((1, 2))
        min_XY = np.min(self.corners, axis=0)
        max_XY = np.max(self.corners, axis=0)
        # Rounding to avoid float precision errors due to representation
        self.bbox = [int(math.floor(round(min_XY[0], 5))), int(math.ceil(round(max_XY[0], 5))), int(math.floor(round(min_XY[1], 5))), int(math.ceil(round(max_XY[1], 5)))]
        #self.bbox = [int(min_XY[0] + math.copysign(0.5, min_XY[0])), int(max_XY[0] + math.copysign(0.5, max_XY[1])), int(min_XY[1] + math.copysign(0.5, min_XY[1])), int(max_XY[1] + math.copysign(0.5, max_XY[1]))]
        self.shape = (self.bbox[1] - self.bbox[0] + 1, self.bbox[3] - self.bbox[2] + 1)


def get_normals(corners):
    """Given a polygon corners list, returns a list of non-normalized normals for each edge"""
    edges = [(corners[i] - corners[(i + 1) % len(corners)]) for i in range(len(corners))]
    normals = [(-e[1], e[0]) for e in edges]
    return normals

def check_normals_side(corners1, normals1, corners2):
    """Checks if all corners2 appear on one side of polygon1"""
    assert(len(corners1) == len(normals1))
    for c, n in zip(corners1, normals1):
        signs2 = [np.sign(np.dot(n, p - c)) for p in corners2]
        signs2 = [s for s in signs2 if abs(s - 0.) > 0.0001] # remove all +-0.
        if np.any(signs2 != signs2[0]):
            return False
    return True

def find_per_row_first_last_one(arr):
    """Given a 2D array (of only 1's in a quadrangle shape, and 0's), for each row find the first and the last occurrance of 1.
       Returns a 2D array with the same number of rows as arr, and 2 columns with the column-indices
       of the first and last one. If a row has only 0's, -1 will be returned on both indices"""
    res = np.full((arr.shape[0], 2), -1, dtype=np.int16)
    # take the first and last column of arr, and find all 1's
    arr_T = arr.T
    first_col_non_zero = np.nonzero(arr_T[0])
    last_col_non_zero = np.nonzero(arr_T[-1])
    for r in first_col_non_zero[0]:
        res[r, 0] = 0
    for r in last_col_non_zero[0]:
        res[r, 1] = arr.shape[1] - 1
    # Now find the positions where the value changes in the middle of the matrix using np.diff
    nonzero = np.nonzero(np.diff(arr))
    # nonzero contents for each row, r:
    #   if nonzero doesn't have a row with r, the row has the same value (either 0 or 1)
    #   if nonzero has a single row with r, the row changes the value once (either from 0 to 1 or from 1 to 0)
    #   if nonzero has row r twice, the row changes both from 0 to 1 and then from 1 to 0
    for r, c in zip(*nonzero):
        if res[r, 0] > -1:
            # already updated the left value, or there is a single change from 1 to 0
            res[r, 1] = c
        else:
            res[r, 0] = c + 1
    return res

