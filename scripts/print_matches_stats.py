# Prints some stats between each two tiles given a correspondence file (that has matched sifts or matches after block matching)

import sys
import argparse
import json
import math

def distance(loc1, loc2):
    dx = loc1[0] - loc2[0]
    dy = loc1[1] - loc2[1]
    return math.sqrt(dx * dx + dy * dy)

def min_mean_max(matches):
    distances = []

    for match in matches:
        loc1 = match["p1"]["w"]
        loc2 = match["p2"]["w"]
        distances.append(distance(loc1, loc2))

    if len(distances) > 0:
        min_d = min(distances)
        mean_d = sum(distances) / len(distances)
        max_d = max(distances)
    else:
        min_d = 0
        mean_d = 0
        max_d = 0

    return [min_d, mean_d, max_d]

def print_matches_stats(correspondence_file):

    with open(correspondence_file) as data_file:
        all_corrs = json.load(data_file)

    for corr_pair in all_corrs:
        print "{} matches found between {} and {}".format(len(corr_pair["correspondencePointPairs"]), corr_pair["url1"], corr_pair["url2"])

        min_d, mean_d, max_d = min_mean_max(corr_pair["correspondencePointPairs"])
        print "Point-Match Distances: min {}, mean {}, max {}".format(min_d, mean_d, max_d)


def main():
    # Command line parser
    parser = argparse.ArgumentParser(description='Prints some stats between each two tiles given' \
                        ' a correspondence file (that has matched sifts or matches after block matching)')
    parser.add_argument('correspondence_file', metavar='correspondence_file', type=str, 
                        help='a correspondence_spec file')


    args = parser.parse_args()

    #print args

    print_matches_stats(args.correspondence_file)

if __name__ == '__main__':
    main()

