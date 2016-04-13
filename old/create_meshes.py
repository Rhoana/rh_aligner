# Creates the transformed meshes of a single layer and serializes them to files (one per tile).
#
# requires:
# - java (executed from the command line)
# - 

import sys
import os
import glob
import argparse
from subprocess import call
import utils


def create_meshes(tiles_fname, output_dir, jar_file, threads_num=4):

    tiles_url = utils.path2url(os.path.abspath(tiles_fname))
    # Compute the Sift features `for each tile in the tile spec file
    java_cmd = 'java -Xmx32g -XX:ParallelGCThreads=1 -Djava.awt.headless=true -cp "{0}" org.janelia.alignment.SaveMesh --targetDir {1} --threads {2} --inputfile {3}'.format(\
        jar_file, output_dir, threads_num, tiles_url)
    utils.execute_shell_command(java_cmd)




def main():
    # Command line parser
    parser = argparse.ArgumentParser(description='Creates the transformed meshes of a single layer and serializes them to files (one per tile).')
    parser.add_argument('tiles_fname', metavar='tiles_json', type=str, 
                        help='a tile_spec file that contains the images to precompute meshes for, in json format')
    parser.add_argument('-o', '--output_dir', type=str, 
                        help='an output directory, where the serialized meshes for the tiles will be stored (default: ./out_meshes)',
                        default='./out_meshes')
    parser.add_argument('-j', '--jar_file', type=str, 
                        help='the jar file that includes the render (default: ../target/render-0.0.1-SNAPSHOT.jar)',
                        default='../target/render-0.0.1-SNAPSHOT.jar')
    parser.add_argument('-t', '--threads_num', type=int, 
                        help='the number of threads to use (default: 1)',
                        default=1)


    args = parser.parse_args()

    #print args
    utils.create_dir(args.output_dir)

    create_meshes(args.tiles_fname, args.output_dir, args.jar_file, threads_num=args.threads_num)

if __name__ == '__main__':
    main()

