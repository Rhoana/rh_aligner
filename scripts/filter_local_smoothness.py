import argparse
from subprocess import call
import utils

def filter_local_smoothness(corr_file, out_fname, jar_file, conf=None):
    # When matching layers, no need to take bounding box into account
    conf_args = utils.conf_args_from_file(conf, 'FilterLocalSmoothness')

    java_cmd = 'java -Xmx3g -XX:ParallelGCThreads=1 -Djava.awt.headless=true -cp "{0}" org.janelia.alignment.FilterLocalSmoothness --inputfile {1} \
            --targetPath {2} {3}'.format(
        jar_file,
        utils.path2url(corr_file),
        out_fname,
        conf_args)
    utils.execute_shell_command(java_cmd)


def main():
    # Command line parser
    parser = argparse.ArgumentParser(description='Does the filter local smoothness stage of the alignment process.')
    parser.add_argument('corr_file', metavar='curr_file', type=str,
                        help='the json file of correspondence spec')
    parser.add_argument('-o', '--output_file', type=str, 
                        help='an output correspondent_spec file, that will include the sift features for each tile (default: ./corrs.json)',
                        default='./corrs.json')
    parser.add_argument('-j', '--jar_file', type=str,
                        help='the jar file that includes the render (default: ../target/render-0.0.1-SNAPSHOT.jar)',
                        default='../target/render-0.0.1-SNAPSHOT.jar')
    parser.add_argument('-c', '--conf_file_name', type=str, 
                        help='the configuration file with the parameters for each step of the alignment process in json format (uses default parameters, if not supplied)',
                        default=None)


    args = parser.parse_args()

    filter_local_smoothness(args.corr_file, args.output_file, args.jar_file, \
        conf=args.conf_file_name)

if __name__ == '__main__':
    main()

