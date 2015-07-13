import numpy as np
import copy
from models import Transforms

def ransac(matches, target_model_type, iterations, epsilon, min_inlier_ratio, min_num_inlier):
    # model = Model.create_model(target_model_type)
    assert(len(matches[0]) == len(matches[1]))

    best_model = None
    best_model_score = 0 # The higher the better
    best_inlier_mask = None
    best_model_mean_dists = 0
    proposed_model = Transforms.create(target_model_type)
    for i in xrange(iterations):
        # choose a minimal number of matches randomly
        min_matches_idxs = np.random.choice(xrange(len(matches[0])), size=proposed_model.MIN_MATCHES_NUM, replace=False)
        # Try to fit them to the model
        proposed_model.fit(matches[0][min_matches_idxs], matches[1][min_matches_idxs])
        # print "proposed_model", proposed_model.to_str()
        # Verify the new model 
        proposed_model_score, inlier_mask, proposed_model_mean = proposed_model.score(matches[0], matches[1], epsilon, min_inlier_ratio, min_num_inlier)
        # print "proposed_model_score", proposed_model_score
        if proposed_model_score > best_model_score:
            best_model = copy.deepcopy(proposed_model)
            best_model_score = proposed_model_score
            best_inlier_mask = inlier_mask
            best_model_mean_dists = proposed_model_mean
    '''
    if best_model is None:
        print "Cannot find a good model during ransac. best_model_score {}".format(best_model_score)
    else:
        print "RANSAC result: best_model_score", best_model_score, "best_model:", best_model.to_str(), "best_model_mean_dists:", best_model_mean_dists
    '''
    return best_inlier_mask, best_model, best_model_mean_dists


def filter_after_ransac(candidates, model, max_trust, min_num_inliers):
    """
    Estimate the AbstractModel and filter potential outliers by robust iterative regression.
    This method performs well on data sets with low amount of outliers (or after RANSAC).
    """
    # copy the model
    new_model = copy.deepcopy(model)

    # iteratively find a new model, by fitting the candidates, and removing those that are far than max_trust*median-distance
    # until the set of remaining candidates does not change its size

    # for the initial iteration, we set a value that is higher the given candidates size
    prev_iteration_num_inliers = candidates.shape[1] + 1

    # keep a copy of the candidates that will be changed due to fitting and error 
    inliers = copy.copy(candidates[0])

    # keep track of the candidates using a mask
    candidates_mask = np.ones((candidates.shape[1]), dtype=np.bool)

    while prev_iteration_num_inliers > np.sum(candidates_mask):
        prev_iteration_num_inliers = np.sum(candidates_mask)
        # Get the inliers and their corresponding matches
        inliers = candidates[0][candidates_mask]
        to_image_candidates = candidates[1][candidates_mask]

        # try to fit the model
        if new_model.fit(inliers, to_image_candidates) == False:
            break

        # get the meidan error (after transforming the points)
        pts_after_transform = new_model.apply_special(inliers)
        dists = np.sqrt(np.sum((pts_after_transform - to_image_candidates) ** 2, axis=1))
        median = np.median(dists)
        # print "dists mean", np.mean(dists)
        # print "median", median
        # print dists <= (median * max_trust)
        inliers_mask = dists <= (median * max_trust)
        candidates_mask[candidates_mask == True] = inliers_mask


    if np.sum(candidates_mask) < min_num_inliers:
        return None, None, -1

    return new_model, candidates_mask, np.mean(dists)


def filter_matches(matches, target_model_type, iterations, epsilon, min_inlier_ratio, min_num_inlier, max_trust):
    """Perform a RANSAC filtering given all the matches"""
    new_model = None
    filtered_matches = None
    meandists = -1

    # Apply RANSAC
    # print "Filtering {} matches".format(matches.shape[1])
    inliers_mask, model, _ = ransac(matches, target_model_type, iterations, epsilon, min_inlier_ratio, min_num_inlier)

    # Apply further filtering
    if inliers_mask is not None:
        inliers = np.array([matches[0][inliers_mask], matches[1][inliers_mask]])
        # print "Found {} good matches out of {} matches after RANSAC".format(inliers.shape[1], matches.shape[1])
        new_model, filtered_inliers_mask, meandists = filter_after_ransac(inliers, model, max_trust, min_num_inlier)
        filtered_matches = np.array([inliers[0][filtered_inliers_mask], inliers[1][filtered_inliers_mask]])
    '''
    if new_model is None:
        print "No model found after RANSAC"
    else:
        # _, filtered_matches_mask, mean_val = new_model.score(matches[0], matches[1], epsilon, min_inlier_ratio, min_num_inlier)
        # filtered_matches = np.array([matches[0][filtered_matches], matches[1][filtered_matches]])
        print "Model found after robust regression: {}, applies to {} out of {} matches.".format(new_model.to_str(), filtered_matches.shape[1], matches.shape[1])
    '''
    return new_model, filtered_matches

