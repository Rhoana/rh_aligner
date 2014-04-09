# Iterates over a directory that contains json files, and creates the sift features of each file.
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
import urllib
import urlparse


# common functions

def path2url(path):
	return urlparse.urljoin(
		'file:', urllib.pathname2url(os.path.abspath(path)))



def compute_all_tiles_sift_features(tile_files, jar, working_dir):
	sift_files = []
	for tile_file in tile_files:
		tile_url = path2url(tile_file)
		fname, ext = os.path.splitext(tile_file.split(os.path.sep)[-1])
		sift_out_file = '{0}_siftFeatures.json'.format(fname)
		sift_out_file = os.path.join(working_dir, sift_out_file)
		java_cmd = 'java -cp "{0}" org.janelia.alignment.ComputeSiftFeatures --url {1} --targetPath {2}'.format(jar, tile_url, sift_out_file)
		#print "Executing: {0}".format(java_cmd)
		call(java_cmd, shell=True) # w/o shell=True it seems that the env-vars are not set


def create_sift_features(tiles_dir, working_dir, jar_file):
	# create a workspace directory if not found
	if not os.path.exists(working_dir):
		os.makedirs(working_dir)


	tile_files = glob.glob(os.path.join(tiles_dir, '*'))

	# Compute the Sift features for each tile
	compute_all_tiles_sift_features(tile_files, jar_file, working_dir)




def main():
	# Command line parser
	parser = argparse.ArgumentParser(description='Iterates over a directory that contains json files, \
		and creates the sift features of each file. \
		The output is either in the same directory or in a different, user-provided, directory \
		(in either case, we use a different file name).')
	parser.add_argument('tiles_dir', metavar='tiles_dir', type=str, 
	                	help='a directory that contains tile_spec files. Sift features will be extracted from each tile')
	parser.add_argument('-w', '--workspace_dir', type=str, 
	                	help='a directory where the output files will be kept (default: ./temp)',
	                	default='./temp')
	parser.add_argument('-j', '--jar_file', type=str, 
	                	help='the jar file that includes the render (default: ../target/render-0.0.1-SNAPSHOT.jar)',
	                	default='../target/render-0.0.1-SNAPSHOT.jar')

	args = parser.parse_args()

	#print args

	create_sift_features(args.tiles_dir, args.workspace_dir, args.jar_file)

if __name__ == '__main__':
	main()

