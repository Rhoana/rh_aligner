import sys
import os
import glob
import argparse
from subprocess import call
from bounding_box import BoundingBox
import json
import itertools
import utils

def export_mesh(jar_file, image_width, image_height, out_fname, conf=None):
    conf_args = utils.conf_args_from_file(conf, 'OptimizeLayersElastic')

    java_cmd = 'java -Xmx3g -XX:ParallelGCThreads=1 -Djava.awt.headless=true -cp "{0}" org.janelia.alignment.ExportMesh --imageWidth {1} --imageHeight {2} \
            --targetPath {3} {4}'.format(
        jar_file,
        int(image_width),
        int(image_height),
        out_fname,
        conf_args)
    utils.execute_shell_command(java_cmd)


def main():
    # Command line parser
    parser = argparse.ArgumentParser(description='Creates a mesh and exports it to a json file.')
    parser.add_argument('-W', '--image_width', type=float, 
                        help='the width of the image')
    parser.add_argument('-H', '--image_height', type=float, 
                        help='the height of the image')
    parser.add_argument('-o', '--output_file', type=str, 
                        help='an output mesh file (default: ./mesh.json)',
                        default='./mesh.json')
    parser.add_argument('-j', '--jar_file', type=str,
                        help='the jar file that includes the render (default: ../target/render-0.0.1-SNAPSHOT.jar)',
                        default='../target/render-0.0.1-SNAPSHOT.jar')
    parser.add_argument('-c', '--conf_file_name', type=str, 
                        help='the configuration file with the parameters for each step of the alignment process in json format (uses default parameters, if not supplied)',
                        default=None)


    args = parser.parse_args()

    export_mesh(args.jar_file, args.image_width, args.image_height,
        args.output_file, conf=args.conf_file_name)

if __name__ == '__main__':
    main()

