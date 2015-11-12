# A script for finding the missing sections given a multi-beam wafer directory
import os
import sys
import glob
import csv
import argparse
import scandir

debug_input_dir = '/n/lichtmanfs2/SCS_2015-9-14_C1_W05_mSEM'

FOCUS_FAIL_STR = 'FOCUS_FAIL'
MISSING_SECTION_STR = 'MISSING_SECTION'
MFOVS_MISSING_STR = 'MISSING_MFOVS'

def read_image_files(folder):
    # Yields non-thumbnail image files from the given folder
    for entry in scandir.scandir(folder):
        if entry.name.endswith('.bmp') and entry.is_file() and not entry.name.startswith('thumbnail') and entry.stat().st_size > 0:
            yield entry.name

def verify_mfov_folder(folder):
    # Make sure that there are 61 (non-zero) file images for the mfov
    img_counter = 0
    for img in read_image_files(folder):
        img_counter += 1
    return img_counter == 61

def read_region_metadata_csv_file(fname):
    #fname = '/n/lichtmanfs2/SCS_2015-9-21_C1_W04_mSEM/_20150930_21-50-13/090_S90R1/region_metadata.csv'
    if os.path.exists(fname):
        with open(fname, 'r') as f:
            reader = csv.reader(f, delimiter=';', quoting=csv.QUOTE_NONE)
            reader.next() # Skip the sep=';' row
            reader.next() # Skip the headers row
            for row in reader:
                # print row
                yield row

def verify_mfovs(folder):
    csv_reader = read_region_metadata_csv_file(os.path.join(folder, "region_metadata.csv"))
    max_mfov_num = 0
    mfovs = []
    for line in csv_reader:
        mfov_num = int(line[0])
        if mfov_num <= 0:
            print("Skipping mfov {} in folder {}".format(mfov_num, folder))
            continue

        # Read the mfov number
        max_mfov_num = max(max_mfov_num, mfov_num)
        mfov_folder = os.path.join(folder, str(mfov_num).zfill(6))
        if not os.path.exists(mfov_folder):
            print("Error: mfov folder {} not found".format(mfov_folder))
        elif not verify_mfov_folder(mfov_folder):
            print("Error: # of images in mfov directory {} is not 61".format(mfov_folder))
        else:
            mfovs.append(mfov_num)

    return mfovs, max_mfov_num

def find_missing_sections(wafer_dir):

    all_sections = {}
    # The directories are sorted by the timestamp and start with '_'
    for sub_folder in sorted(glob.glob(os.path.join(wafer_dir, '*'))):
        if os.path.isdir(sub_folder):
            print("Parsing subfolder: {}".format(sub_folder))
            # Get all section folders in the sub-folder
            all_sections_folders = sorted(glob.glob(os.path.join(sub_folder, '*_*')))
            for section_folder in all_sections_folders:
                if os.path.isdir(section_folder):
                    # Found a section directory, now need to find out if it has a focus issue or not
                    # (if it has any sub-dir that is all numbers, it hasn't got an issue)
                    section_num = os.path.basename(section_folder).split('_')[0]
                    relevant_mfovs, max_mfov_num = verify_mfovs(section_folder)
                    if len(relevant_mfovs) > 0:
                        # a good section
                        if min(relevant_mfovs) == 1 and max_mfov_num == len(relevant_mfovs):
                            # The directories in the wafer directory are sorted by the timestamp, and so here we'll get the most recent scan of the section
                            all_sections[section_num] = section_folder
                        else:
                            missing_mfovs = []
                            for i in range(0, max(relevant_mfovs)):
                                if i+1 not in relevant_mfovs:
                                    missing_mfovs.append(str(i+1))
                            all_sections[section_num] = MFOVS_MISSING_STR + ':"{}"'.format(','.join(missing_mfovs))
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

