# Iterates over a directory that contains correspondence list json files, and optimizes the montage by perfroming the transform on each file.
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




def optimize_montage_transform(correspondence_file, tilespec_file, output_file, jar_file):

	corr_url = path2url(correspondence_file)
	tiles_url = path2url(tilespec_file)
	java_cmd = 'java -Djava.awt.headless=true -cp "{0}" org.janelia.alignment.OptimizeMontageTransform --inputfile {1} --tilespecfile {2} --targetPath {3}'.format(\
		jar_file, corr_url, tiles_url, output_file)
	#print "Executing: {0}".format(java_cmd)
	call(java_cmd, shell=True) # w/o shell=True it seems that the env-vars are not set




def main():
	# Command line parser
	parser = argparse.ArgumentParser(description='Takes a correspondence list json file, \
		and optimizes the montage by perfroming the transform on each tile in the file.')
	parser.add_argument('correspondence_file', metavar='correspondence_file', type=str, 
	                	help='a correspondence_spec file')
	parser.add_argument('tilespec_file', metavar='tilespec_file', type=str, 
	                	help='a tilespec file containing all the tiles')
	parser.add_argument('output_file', metavar='output_file', type=str, 
	                	help='the output file')
	parser.add_argument('-j', '--jar_file', type=str, 
	                	help='the jar file that includes the render (default: ../target/render-0.0.1-SNAPSHOT.jar)',
	                	default='../target/render-0.0.1-SNAPSHOT.jar')

	args = parser.parse_args()

	#print args

	optimize_montage_transform(args.correspondence_file, args.tilespec_file, args.output_file, args.jar_file)

if __name__ == '__main__':
	main()

