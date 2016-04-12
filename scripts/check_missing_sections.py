# A script for finding the missing sections given a multi-beam wafer directory
import os
import sys
import glob
import stat
import csv
import argparse
import sqlite3
import re
import time

debug_input_dir = '/n/lichtmanfs2/SCS_2015-9-14_C1_W05_mSEM'

FOCUS_FAIL_STR = 'FOCUS_FAIL'
MISSING_SECTION_STR = 'MISSING_SECTION'
MFOVS_MISSING_STR = 'MISSING_MFOVS'

def create_db(db_fname):
    print("Using database file: {}".format(db_fname))
    db = sqlite3.connect(db_fname)
    db.isolation_level = None
    cursor = db.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS parsed_folders(id INTEGER PRIMARY KEY AUTOINCREMENT,
                                                                wafer_dir TEXT NOT NULL,
                                                                batch_dir TEXT NOT NULL,
                                                                dir TEXT NOT NULL,
                                                                section_num INTEGER NOT NULL,
                                                                errors TEXT)''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS final_folders(wafer_dir TEXT NOT NULL,
                                                               section_num INTEGER NOT NULL,
                                                               parsed_folder_id INTEGER NOT NULL,
                                                               FOREIGN KEY(parsed_folder_id) REFERENCES parsed_folders(id))''')
    cursor.execute('''CREATE UNIQUE INDEX IF NOT EXISTS unique_folders ON final_folders(wafer_dir, section_num)''')
    db.commit()

    return db, cursor

def get_wafers_dirs(cursor):
    cursor.execute("SELECT DISTINCT wafer_dir FROM parsed_folders")
    data = cursor.fetchall()
    return [entry[0] for entry in data]


def get_batch_dirs(cursor, wafer_dir):
    cursor.execute("SELECT DISTINCT batch_dir FROM parsed_folders WHERE wafer_dir=?",
                   (wafer_dir,))
    data = cursor.fetchall()
    return [entry[0] for entry in data]

def get_parsed_dirs(cursor, wafer_dir):
    cursor.execute("SELECT DISTINCT dir FROM parsed_folders WHERE wafer_dir=?",
                   (wafer_dir,))
    data = cursor.fetchall()
    return [entry[0] for entry in data]

def add_parsed_folder(cursor, db, wafer_dir, batch_dir, dir, section_num, errors):
    cursor.execute("INSERT INTO parsed_folders (wafer_dir, batch_dir, dir, section_num, errors) VALUES (?, ?, ?, ?, ?)",
                   (wafer_dir, batch_dir, dir, section_num, errors))
    db.commit()
    return cursor.lastrowid

def update_final_folder(cursor, db, wafer_dir, section_num, parsed_folder_id):
    cursor.execute("INSERT OR REPLACE INTO final_folders (wafer_dir, section_num, parsed_folder_id) VALUES (?, ?, ?)",
                   (wafer_dir, section_num, parsed_folder_id))
    db.commit()

def get_final_folder_id(cursor, wafer_dir, section_num):
    cursor.execute("SELECT parsed_folder_id FROM final_folders WHERE wafer_dir=? AND section_num=?",
                   (wafer_dir, section_num))
    data = cursor.fetchall()
    return [entry[0] for entry in data][0]

def get_wafer_final_folder_ids(cursor, wafer_dir):
    cursor.execute("SELECT parsed_folder_id FROM final_folders WHERE wafer_dir=?",
                   (wafer_dir,))
    data = cursor.fetchall()
    return [entry[0] for entry in data]

def get_wafer_final_folders(cursor, wafer_dir):
    cursor.execute("SELECT * FROM final_folders JOIN parsed_folders ON final_folders.parsed_folder_id = parsed_folders.id WHERE final_folders.wafer_dir=?",
                   (wafer_dir,))
    data = cursor.fetchall()
    returned_data = [{
                        'parsed_folder_id' : entry[3],
                        'wafer_dir' : entry[4],
                        'batch_dir' : entry[5],
                        'section_dir' : entry[6],
                        'section_num' : entry[7],
                        'errors' : entry[8]
                     } for entry in data]
    return returned_data


def normalize_path(dir):
    # Normalize the path name, in case we are in windows or given a relative path
    dir_normalized = os.path.abspath(dir).replace('\\','/')
    dir_normalized = dir_normalized[dir_normalized.find('/'):]
    return dir_normalized

def read_image_files(folder):
    # Yields non-thumbnail image files from the given folder
    for fname in glob.glob(os.path.join(folder, '*.bmp')):
        fstat = os.stat(fname)
        # Verify that it is a bmp file, non-thumbnail, that is actually a file, and non-empty
        if (not os.path.basename(fname).startswith('thumbnail')) and stat.S_ISREG(fstat.st_mode) and fstat.st_size > 0:
            yield fname

def verify_mfov_folder(folder):
    # Make sure that there are 61 (non-zero) file images for the mfov
    img_counter = 0
    for img in read_image_files(folder):
        img_counter += 1
    return img_counter == 61

def read_region_metadata_csv_file(fname):
    #fname = '/n/lichtmanfs2/SCS_2015-9-21_C1_W04_mSEM/_20150930_21-50-13/090_S90R1/region_metadata.csv'
    with open(fname, 'r') as f:
        reader = csv.reader(f, delimiter=';', quoting=csv.QUOTE_NONE)
        reader.next() # Skip the sep=';' row
        reader.next() # Skip the headers row
        for row in reader:
            # print row
            yield row

def verify_mfovs(folder):
    # if the region_metadata file doesn't exist, need to skip that folder
    if not os.path.exists(os.path.join(folder, "region_metadata.csv")):
       return -1, -1

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


def parse_batch_dir(batch_dir, wafer_dir, wafer_dir_normalized, prev_section_dirs, db, cursor):
    batch_section_data = {}
    last_section_folder = None
    print("Parsing batch dir: {}".format(batch_dir))
    # Get all section folders in the sub-folder
    all_sections_folders = sorted(glob.glob(os.path.join(batch_dir, '*_*')))
    for section_folder in all_sections_folders:
        # If already parsed that section dir
        if normalize_path(section_folder) in prev_section_dirs:
            print("Previously parsed section dir: {}, skipping...".format(section_folder))
            continue

        print("Parsing section dir: {}".format(section_folder))
        if os.path.isdir(section_folder):
            # Found a section directory, now need to find out if it has a focus issue or not
            # (if it has any sub-dir that is all numbers, it hasn't got an issue)
            section_num = os.path.basename(section_folder).split('_')[0]
            relevant_mfovs, max_mfov_num = verify_mfovs(section_folder)
            if relevant_mfovs == -1:
                # The folder doesn't have the "region_metadata.csv" file, need to skip it
                continue
            batch_section_data[section_num] = {'folder': section_folder}
            if len(relevant_mfovs) > 0:
                # a good section
                if min(relevant_mfovs) == 1 and max_mfov_num == len(relevant_mfovs):
                    # The directories in the wafer directory are sorted by the timestamp, and so here we'll get the most recent scan of the section
                    batch_section_data[section_num]['errors'] = None
                else:
                    missing_mfovs = []
                    for i in range(0, max(relevant_mfovs)):
                        if i+1 not in relevant_mfovs:
                            missing_mfovs.append(str(i+1))
                    batch_section_data[section_num]['errors'] = MFOVS_MISSING_STR + ':"{}"'.format(','.join(missing_mfovs))
            else:
                batch_section_data[section_num]['errors'] = FOCUS_FAIL_STR

        
    # No need to verify that the last section is not being imaged at the moment
    # because we only consider sections that have the "region_metadata.csv", and if it is not there,
    # the folder is skipped
    
            
    # Insert all parsed section folders to the database
    for section_num in batch_section_data:
        row_id = add_parsed_folder(cursor, db, wafer_dir_normalized, normalize_path(batch_dir), normalize_path(batch_section_data[section_num]['folder']),
                                   section_num, batch_section_data[section_num]['errors'])
        batch_section_data[section_num]['parsed_folder_id'] = row_id

    return batch_section_data



def find_missing_sections(wafer_dir, wafer_dir_normalized, db, cursor):

    all_sections = {}
    # update all_sections with data from the previously parsed folders
    prev_final_data = get_wafer_final_folders(cursor, wafer_dir_normalized)
    for entry in prev_final_data:
        section_num_str = str(entry['section_num']).zfill(3)
        all_sections[section_num_str] = {
                                            'folder': entry['section_dir'],
                                            'errors': entry['errors'],
                                            'parsed_folder_id': entry['parsed_folder_id']
                                        }

    # Fetch the previously parsed dirs
    prev_section_dirs = get_parsed_dirs(cursor, wafer_dir_normalized)
    print("prev_dirs:", prev_section_dirs)

    # The batch directories are sorted by the timestamp in the directory name. Need to store it in a hashtable for sorting
    all_batch_files = glob.glob(os.path.join(wafer_dir, '*'))
    all_batch_dirs = []
    dir_to_time = {}
    for folder in all_batch_files:
        # Assuming folder names similar to: scs_20151217_19-45-07   (scs can be changed to any other name)
        if os.path.isdir(folder):
            m = re.match('.*_([0-9]{8})_([0-9]{2})-([0-9]{2})-([0-9]{2})$', folder)
            if m is not None:
                dir_to_time[folder] = "{}_{}-{}-{}".format(m.group(1), m.group(2), m.group(3), m.group(4))
                all_batch_dirs.append(folder)

    # Parse the batch directories
    for sub_folder in sorted(all_batch_dirs, key=lambda folder: dir_to_time[folder]):
        if os.path.isdir(sub_folder):
            batch_section_data = parse_batch_dir(sub_folder, wafer_dir, wafer_dir_normalized, prev_section_dirs, db, cursor)

            # Update the sections that were parsed during this execution
            all_sections.update(batch_section_data)

            
    # Insert a missing section for each non-seen section number starting from 001 (and ending with the highest number)
    all_sections_keys = sorted(all_sections.keys())
    max_section = int(all_sections_keys[-1])
    prev_section = 0
    missing_sections = []
    focus_failed_sections = []
    missing_mfovs_sections = []

    cursor.execute("begin")
    for i in range(1, max_section + 1):
        section_num_str = str(i).zfill(3)
        if section_num_str in all_sections:
            # Found a section
            if all_sections[section_num_str]['errors'] == FOCUS_FAIL_STR:
                focus_failed_sections.append(i)
            elif all_sections[section_num_str]['errors'] is not None:
                missing_mfovs_section.append(i)
            # Add the section to the final folders in the db
            update_final_folder(cursor, db, wafer_dir_normalized, i, all_sections[section_num_str]['parsed_folder_id'])
        else:
            # Found a missing section
            missing_sections.append(i)
            all_sections[section_num_str] = {
                'folder': MISSING_SECTION_STR,
                'errors': MISSING_SECTION_STR }
    db.commit()

    return all_sections, missing_sections, focus_failed_sections, missing_mfovs_sections

def find_and_save_missing_sections(wafer_dir, db_fname, output_fname):
    db, cursor = create_db(db_fname)
    all_sections, missing_sections, focus_failed_sections, missing_mfovs_sections = find_missing_sections(wafer_dir, normalize_path(wafer_dir), db, cursor)
    print("Found {} sections, {} missing sections, {} focus failed sections, {} sections with missing mfovs".format(len(all_sections), len(missing_sections), len(focus_failed_sections), len(missing_mfovs_sections)))
    print("Missing sections: {}".format(missing_sections))
    print("Focus failed sections: {}".format(focus_failed_sections))
    print("Missing mfovs sections: {}".format(missing_mfovs_sections))

    # Output everything to csv
    print("Saving CSV file to: {}".format(output_fname))
    with open(output_fname, 'wb') as f:
        section_keys = sorted(all_sections.keys())
        w = csv.writer(f, delimiter=',')
        for section_num in section_keys:
            w.writerow([section_num, all_sections[section_num]['folder'], all_sections[section_num]['errors']])
    print("Done")


if __name__ == '__main__':
    # Command line parser
    parser = argparse.ArgumentParser(description='A script for finding the missing sections given a multi-beam wafer directory')
    parser.add_argument('wafer_dir', metavar='wafer_dir', type=str,
            help='a directory where the wafer sections are located (e.g., /n/lichtmanfs2/SCS_2015-9-14_C1_W05_mSEM)')
    parser.add_argument('-o', '--output_fname', type=str,
            help='an output CSV file (default: ./wafer_data.csv)',
            default='./wafer_data.csv')
    parser.add_argument('-d', '--db_fname', type=str,
            help='a db file that stores all the parsed directories (default: ./parsed_folders.db)',
            default='./parsed_folders.db')

    args = parser.parse_args()

    find_and_save_missing_sections(args.wafer_dir, args.db_fname, args.output_fname)

