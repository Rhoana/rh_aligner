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
        if (i + 1) % 100 == 0:
            print "starting RANSAC iteration {}".format(i + 1)
        # choose a minimal number of matches randomly
        min_matches_idxs = np.random.choice(xrange(len(matches[0])), size=proposed_model.MIN_MATCHES_NUM, replace=False)
        # Try to fit them to the model
        proposed_model.fit(matches[0][min_matches_idxs], matches[1][min_matches_idxs])
        # Verify the new model 
        proposed_model_score, inlier_mask, proposed_model_mean = proposed_model.score(matches[0], matches[1], epsilon, min_inlier_ratio, min_num_inlier)
        # print "proposed_model_score", proposed_model_score
        if proposed_model_score > best_model_score:
            best_model = copy.deepcopy(proposed_model)
            best_model_score = proposed_model_score
            best_inlier_mask = inlier_mask
            best_model_mean_dists = proposed_model_mean

    print "best_model_score", best_model_score, "best_model:", best_model.to_str(), "best_model_mean_dists:", best_model_mean_dists
    return best_inlier_mask, best_model, best_model_mean_dists


def filter_ransac(candidates, model, max_trust, min_num_inliers):
    """
    Estimate the AbstractModel and filter potential outliers by robust iterative regression.
    This method performs well on data sets with low amount of outliers (or after RANSAC).
    """
    # copy the model
    new_model = copy.deepcopy(model)

    num_inliers = candidates.shape[1] + 1 # for the initial while iteration, this should be increased by 1
    to_image_inliers = copy.copy(candidates[1])
    inliers = copy.copy(candidates[0])
    # print "to_image_inliers", to_image_inliers
    # print "from_image_inliers", inliers
    while num_inliers > inliers.shape[0]:
        temp = copy.copy(inliers)
        # fit the model
        if new_model.fit(temp, to_image_inliers) == False:
            break

        # get the median error
        dists = np.zeros((temp.shape[0]), dtype=np.float64)
        for i, match in enumerate(zip(temp, to_image_inliers)):
            new_point = new_model.apply(match[0])
            # add the l2 distance
            dists[i] = np.sqrt(np.sum((new_point - match[1]) ** 2))
        median = np.median(dists)
        # print "dists mean", np.mean(dists)
        # print "median", median
        # print dists <= (median * max_trust)
        inliers_mask = dists <= (median * max_trust)
        inliers = temp[inliers_mask]
        to_image_inliers = to_image_inliers[inliers_mask]
        num_inliers = inliers.shape[0]

    if num_inliers < min_num_inliers:
        return None, None

    return new_model, inliers_mask


def filter_matches(matches, target_model_type, iterations, epsilon, min_inlier_ratio, min_num_inlier, max_trust):
    """Perform a RANSAC filtering given all the matches"""
    new_model = None
    filtered_matches = None

    # Apply RANSAC
    print "Filtering {} matches".format(matches.shape[1])
    inliers_mask, model, _ = ransac(matches, target_model_type, iterations, epsilon, min_inlier_ratio, min_num_inlier)
    inliers = np.array([matches[0][inliers_mask], matches[1][inliers_mask]])

    # Apply further filtering
    if inliers is not None:
        print "Found {} good matches out of {} matches after RANSAC".format(inliers.shape[1], matches.shape[1])
        new_model, filtered_inliers_mask = filter_ransac(inliers, model, max_trust, min_num_inlier)
        filtered_matches = np.array([inliers[0][filtered_inliers_mask], inliers[1][filtered_inliers_mask]])

    if new_model is None:
        print "No model found after RANSAC"
    else:
        # _, filtered_matches_mask, mean_val = new_model.score(matches[0], matches[1], epsilon, min_inlier_ratio, min_num_inlier)
        # filtered_matches = np.array([matches[0][filtered_matches], matches[1][filtered_matches]])
        print "Model found: {}, applies to {} out of {} matches.".format(new_model.to_str(), filtered_matches.shape[1], matches.shape[1])

    return new_model, filtered_matches

