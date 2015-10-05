# A script for finding the missing sections given a multi-beam wafer directory
import os
import sys
import glob
import csv
import argparse

debug_input_dir = '/n/lichtmanfs2/SCS_2015-9-14_C1_W05_mSEM'

FOCUS_FAIL_STR = 'FOCUS_FAIL'
MISSING_SECTION_STR = 'MISSING_SECTION'

def has_inner_directories(folder):
    inner_files = glob.glob(os.path.join(folder, '*'))
    for inner_file in inner_files:
        if os.path.isdir(inner_file) and os.path.basename(inner_file).isdigit():
            return True
    return False

def find_missing_sections(wafer_dir):

    all_sections = {}
    # The directories are sorted by the timestamp and start with '_'
    for sub_folder in sorted(glob.glob(os.path.join(wafer_dir, '_*'))):
        if os.path.isdir(sub_folder):
            print("Parsing subfolder: {}".format(sub_folder))
            # Get all section folders in the sub-folder
            all_sections_folders = sorted(glob.glob(os.path.join(sub_folder, '*_*')))
            for section_folder in all_sections_folders:
                if os.path.isdir(section_folder):
                    # Found a section directory, now need to find out if it has a focus issue or not
                    # (if it has any sub-dir that is all numbers, it hasn't got an issue)
                    section_num = os.path.basename(section_folder).split('_')[0]
                    good_section = has_inner_directories(section_folder)
                    if good_section:
                        # The directories in the wafer directory are sorted by the timestamp, and so here we'll get the most recent scan of the section
                        all_sections[section_num] = section_folder
                    else:
                        all_sections[section_num] = FOCUS_FAIL_STR
    # Insert a missing section for each non-seen section number starting from 001 (and ending with the highest number)
    all_sections_keys = sorted(all_sections.keys())
    max_section = int(all_sections_keys[-1])
    prev_section = 0
    missing_sections = []
    focus_failed_sections = []
    for i in range(1, max_section + 1):
        section_num_str = str(i).zfill(3)
        if section_num_str in all_sections:
            # Found a section
            if all_sections[section_num_str] == FOCUS_FAIL_STR:
                focus_failed_sections.append(i)
        else:
            # Found a missing section
            missing_sections.append(i)
            all_sections[section_num_str] = MISSING_SECTION_STR
    return all_sections, missing_sections, focus_failed_sections

def find_and_save_missing_sections(wafer_dir, output_fname):
    all_sections, missing_sections, focus_failed_sections = find_missing_sections(wafer_dir)
    print("Found {} sections, {} missing sections, {} focus failed sections".format(len(all_sections), len(missing_sections), len(focus_failed_sections)))
    print("Missing sections: {}".format(missing_sections))
    print("Focus failed sections: {}".format(focus_failed_sections))

    # Output everything to csv
    print("Saving CSV file to: {}".format(output_fname))
    with open(output_fname, 'wb') as f:
        section_keys = sorted(all_sections.keys())
        w = csv.writer(f, delimiter=',')
        for section_num in section_keys:
            w.writerow([section_num, all_sections[section_num]])
    print("Done")


if __name__ == '__main__':
    # Command line parser
    parser = argparse.ArgumentParser(description='A script for finding the missing sections given a multi-beam wafer directory')
    parser.add_argument('wafer_dir', metavar='wafer_dir', type=str,
            help='a directory where the wafer sections are located (e.g., /n/lichtmanfs2/SCS_2015-9-14_C1_W05_mSEM)')
    parser.add_argument('-o', '--output_fname', type=str,
            help='an output CSV file (default: ./wafer_data.csv)',
            default='./wafer_data.csv')

    args = parser.parse_args()

    find_and_save_missing_sections(args.wafer_dir, args.output_fname)

