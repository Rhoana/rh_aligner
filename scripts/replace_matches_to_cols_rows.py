import sys
import os
import argparse
import json

multipliers = {}

def round_float_to_int(num, decimalPlace = 3):
    if not decimalPlace in multipliers.keys():
        mul = 1
        for i in range(decimalPlace):
            mul *= 10
        multipliers[decimalPlace] = mul
    mul = multipliers[decimalPlace]
    return int(num * mul)

def read_mesh(mesh_json_fname):
    xy_to_colrow = {}
    with open(mesh_json_fname, 'r') as mesh_file:
        mesh = json.load(mesh_file)
    for point in mesh['points']:
        p_x = round_float_to_int(point['x'])
        p_y = round_float_to_int(point['y'])
        p_row = int(point['row'])
        p_col = int(point['col'])

        if not p_x in xy_to_colrow.keys():
            xy_to_colrow[p_x] = {}
        if not p_y in xy_to_colrow[p_x].keys():
            xy_to_colrow[p_x][p_y] = []
        xy_to_colrow[p_x][p_y] = [p_col, p_row]
    return xy_to_colrow

def read_matches(matches_json_fname):
    matches = None
    with open(matches_json_fname, 'r') as matches_file:
        matches = json.load(matches_file)
    return matches



def replace_matches_to_cols_rows(matches_json_fname, mesh_json_fname, output_fname):

    # read mesh and matches
    xy_to_colrow = read_mesh(mesh_json_fname)
    matches = read_matches(matches_json_fname)

    # replace the matches data
    replaced_matches = []
    for match in matches:
        replaced_match = {}
        replaced_match['mipmapLevel'] = match['mipmapLevel']
        replaced_match['url1'] = match['url1']
        replaced_match['url2'] = match['url2']
        replaced_match['shouldConnect'] = match['shouldConnect']

        if 'correspondencePointPairs' in match.keys():
            replaced_point_pairs = []
            for point_pair in match['correspondencePointPairs']:
                point_pair_x = round_float_to_int(point_pair['p1']['l'][0])
                point_pair_y = round_float_to_int(point_pair['p1']['l'][1])
                if (not point_pair_x in xy_to_colrow.keys()) or (not point_pair_y in xy_to_colrow[point_pair_x].keys()):
                    print "Error: could not find point {},{} in the mesh file".format(point_pair['p1']['l'][0], point_pair['p1']['l'][1])
                else:
                    replaced_point_pair = {}
                    p1 = {}
                    p1['col'] = xy_to_colrow[point_pair_x][point_pair_y][0]
                    p1['row'] = xy_to_colrow[point_pair_x][point_pair_y][1]
                    replaced_point_pair['p1'] = p1
                    replaced_point_pair['p2'] = point_pair['p2']
                    replaced_point_pairs.append(replaced_point_pair)
            replaced_match['correspondencePointPairs'] = replaced_point_pairs
        replaced_matches.append(replaced_match)

    # output the new json file
    with open(output_fname, 'w') as out_file:
        json.dump(replaced_matches, out_file, sort_keys=True, indent=4)



def main():
    # Command line parser
    parser = argparse.ArgumentParser(description='Given a matches json file and a mesh json file, creates a json file with row,col entries (instead of x,y float locations).')
    parser.add_argument('matches_json_fname', metavar='matches_json_fname', type=str, 
                        help='a json file that contains the matches between two tiles/sections')
    parser.add_argument('mesh_json_fname', metavar='mesh_json_fname', type=str, 
                        help='a json file that contains the mesh')
    parser.add_argument('-o', '--output_fname', type=str, 
                        help='an output filename (default: ./new_matches.json)',
                        default='./new_matches.json')

    args = parser.parse_args()


    replace_matches_to_cols_rows(args.matches_json_fname, args.mesh_json_fname, args.output_fname)

if __name__ == '__main__':
    main()

