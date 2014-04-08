# Iterates over a directory that contains Tile-Spec json files (one for each tile), and a directory of sift features (for each tile)
# and matches the features of every two tiles that overlap.
# The output is either in the same directory or in a different, user-provided, directory
# (in either case, we use a different file name)
#
# requires:
# - java (executed from the command line)
# - 

import sys
import os
import glob
import argparse
from subprocess import call
from bounding_box import BoundingBox
import json
import itertools

# common functions



def match_two_tiles_sift_features(tile1, tile2, jar, working_dir):
	fname1, ext1 = os.path.splitext(tile1['imageUrl'].split(os.path.sep)[-1])
	fname2, ext2 = os.path.splitext(tile2['imageUrl'].split(os.path.sep)[-1])
	match_out_file = '{0}_{1}_matchFeatures.json'.format(fname1, fname2)
	match_out_file = os.path.join(working_dir, match_out_file)
	java_cmd = 'java -cp "{0}" org.janelia.alignment.MatchSiftFeatures --featurefile1 {1} --featurefile2 {2} --targetPath {3}'.format(\
		jar, tile1['featuresSpec'], tile2['featuresSpec'], match_out_file)
	print "Executing: {0}".format(java_cmd)
	call(java_cmd, shell=True) # w/o shell=True it seems that the env-vars are not set


def load_entire_data(tile_files, features_files):
	# Loads the entire collection of tile and features files, and returns
	# a mapping of a tile->[imageUrl, bounding_box, sift_features]
	# This is has a large memory overhead, but is needed because we
	# do not have the data in a database or some global dictionary that is indexed
	tiles = {}
	for tile_file in tile_files:
		tile = {}

		# load tile_file (json)
		with open(tile_file, 'r') as data_file:
			data = json.load(data_file)

		tile['tileSpec'] = tile_file
		tile['imageUrl'] = data[0]['imageUrl']
		tile['boundingBox'] = data[0]['boundingBox']
		tiles[tile['imageUrl']] = tile

	# load and add for each tile the corresponding features file
	for features_file in features_files:
		tile = None

		# load tile_file (json)
		with open(features_file, 'r') as data_file:
			data = json.load(data_file)

		if data[0]['imageUrl'] in tiles.keys():
			tile = tiles[data[0]['imageUrl']]
			tile['featuresSpec'] = features_file
			tile['featureList'] = data[0]['featureList']
		else:
			print "Warning: found a sift features file ({0}) without a tile-spec file.".format(features_file)

	return tiles



def match_sift_features(tiles_dir, features_dir, working_dir, jar_file):
	# create a workspace directory if not found
	if not os.path.exists(working_dir):
		os.makedirs(working_dir)


	tile_files = glob.glob(os.path.join(tiles_dir, '*'))
	features_files = glob.glob(os.path.join(features_dir, '*'))

	tiles = load_entire_data(tile_files, features_files)

	# TODO: add all tiles to a kd-tree so it will be faster to find overlap between tiles

	# iterate over the tiles, and for each tile, find intersecting tiles that overlap,
	# and match their features
	# Nested loop:
	#	for each tile_i in range[0..N):
	#		for each tile_j in range[tile_i..N)]
	for pair in itertools.combinations(tiles, 2):
		# if the two tiles intersect, match them
		bbox1 = BoundingBox(tiles[pair[0]]['boundingBox'])
		bbox2 = BoundingBox(tiles[pair[1]]['boundingBox'])
		if bbox1.overlap(bbox2):
			print "Matching sift of tiles: {0} and {1}".format(pair[0], pair[1])
			match_two_tiles_sift_features(tiles[pair[0]], tiles[pair[1]], jar_file, working_dir)
		#else:
		#	print "Tiles: {0} and {1} do not overlap, so no matching is done".format(pair[0], pair[1])




def main():
	# Command line parser
	parser = argparse.ArgumentParser(description='Iterates over a directory that contains Tile-Spec json files (one for each tile), \
		and a directory of sift features (for each tile) and matches the features of every two tiles that overlap.')
	parser.add_argument('tiles_dir', metavar='tiles_dir', type=str, 
	                	help='a directory that contains tile_spec files')
	parser.add_argument('features_dir', metavar='features_dir', type=str, 
	                	help='a directory that contains sift_features files')
	parser.add_argument('-w', '--workspace_dir', type=str, 
	                	help='a directory where the output files will be kept (default: ./temp)',
	                	default='./temp')
	parser.add_argument('-j', '--jar_file', type=str, 
	                	help='the jar file that includes the render (default: ../target/render-0.0.1-SNAPSHOT.jar)',
	                	default='../target/render-0.0.1-SNAPSHOT.jar')

	args = parser.parse_args()

	#print args

	match_sift_features(args.tiles_dir, args,features_dir, args.workspace_dir, args.jar_file)

if __name__ == '__main__':
	main()

