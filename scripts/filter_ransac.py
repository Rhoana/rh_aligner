import argparse
from subprocess import call
import utils

# common functions

def filter_ransac(corr_file, compared_url, out_fname, jar_file, conf=None):
    # When matching layers, no need to take bounding box into account
    conf_args = utils.conf_args(conf, 'FilterRansac')

    java_cmd = 'java -Xmx4g -Djava.awt.headless=true -cp "{0}" org.janelia.alignment.FilterRansac --inputfile {1} --comparedUrl {2} \
            --targetPath {3} {4}'.format(
        jar_file,
        utils.path2url(corr_file),
        compared_url,
        out_fname,
        conf_args)
    utils.execute_shell_command(java_cmd)


def main():
    # Command line parser
    parser = argparse.ArgumentParser(description='Does the filter and ransac stage of the alignment process.')
    parser.add_argument('corr_file', metavar='curr_file', type=str,
                        help='the json file of correspondence spec')
    parser.add_argument('compared_url', metavar='compared_url', type=str,
                        help='the url of one of the tilespec files to compare all models against')
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

    filter_ransac(args.corr_file, args.compared_url, \
        args.output_file, args.jar_file, \
        conf=utils.conf_args_from_file(args.conf_file_name, "FilterRansac"))

if __name__ == '__main__':
    main()

