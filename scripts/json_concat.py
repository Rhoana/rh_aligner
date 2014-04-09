# Takes a directory that has a json files in it, and concatenate them to a single file.
#

import sys
import os
import glob
import argparse
import json



def json_concat(files_dir, output_file):

	json_files = glob.glob(os.path.join(files_dir, '*'))

	# In order to avoid opening all the files and creating a large json in memory,
	# we hand build the output json file, and iteratively populate it with the
	# data from the files

	with open(output_file, 'w') as outfile:
		outfile.write('[\n')

		for i, json_file in enumerate(json_files):
			data = None
			with open(json_file, 'r') as data_file:
				data = json.load(data_file)
			if not data == None:
				json_data_str = json.dumps(data[0])
				outfile.write(json_data_str)
				if i < len(json_files) - 1:
					outfile.write(',')
				outfile.write('\n')
		outfile.write(']\n')





def main():
	# Command line parser
	parser = argparse.ArgumentParser(description='Takes a directory that has a json files in it, and concatenate them to a single file.')
	parser.add_argument('files_dir', metavar='files_dir', type=str, 
	                	help='a directory that contains json files to be concatenated')
	parser.add_argument('output_file', metavar='output_file', type=str, 
	                	help='the json file that will contain the output')

	args = parser.parse_args()

	#print args

	json_concat(args.files_dir, args.output_file)

if __name__ == '__main__':
	main()

