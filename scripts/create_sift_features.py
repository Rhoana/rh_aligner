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



def compute_sift_features(tile_files, jar, working_dir):
	sift_files = []
	for tile_file in tile_files:
		tile_url = path2url(tile_file)
		fname, ext = os.path.splitext(tile_file.split(os.path.sep)[-1])
		sift_out_file = '{0}_siftFeatures.json'.format(fname)
		sift_out_file = os.path.join(working_dir, sift_out_file)
		java_cmd = 'java -cp "{0}" org.janelia.alignment.ComputeSiftFeatures --url {1} --targetPath {2}'.format(jar, tile_url, sift_out_file)
		#print "Executing: {0}".format(java_cmd)
		call(java_cmd, shell=True) # w/o shell=True it seems that the env-vars are not set



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
                	help='the jar file that includes the render (default: ./render-0.0.1-SNAPSHOT.jar)',
                	default='./render-0.0.1-SNAPSHOT.jar')

args = parser.parse_args()

#print args

# create a workspace directory if not found
if not os.path.exists(args.workspace_dir):
	os.makedirs(args.workspace_dir)


tile_files = glob.glob(os.path.join(args.tiles_dir, '*'))

# Compute the Sift features for each of the relevant tiles (between pairs of overlapping tiles)
sift_files = compute_sift_features(tile_files, args.jar_file, args.workspace_dir)

