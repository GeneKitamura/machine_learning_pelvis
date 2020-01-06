import pydicom
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import re
import shutil

from glob import glob

MAIN_PATH = './images/PelvisImagesDICOM/'
R60_DF = pd.read_excel('./old_xlsx/R60_cases_anon.xlsx') # all requested cases
IMG_MAP_DF = pd.read_excel('./old_xlsx/R60_IMAGE_MAPPING_anon.xlsx')
NOTE_MAP_DF = pd.read_excel('./old_xlsx/R60_NOTE_MAPPING_anon.xlsx')
PELVIC_FX_DF = pd.read_excel('./old_xlsx/pelvic_final_anon.xlsx') #actual useful 6263 cases

non_pelvis_list = ['./images/PelvisImagesDICOM/Patient_7888715990/Study_chest_82594812']

def count_dirs(c_path=MAIN_PATH):
    dir_count = 0
    multi_pt_list = []
    all_pt_list = []
    single_pt_list = []
    study_acc_dir_list = []
    unique_acc_dir_list = []

    for pt in os.listdir(c_path):
        # print(next(os.walk(os.path.join(MAIN_PATH, i)))[1])
        subdir_path = next(os.walk(os.path.join(c_path, pt)))[1]
        for pt_acc in subdir_path:
            c_acc = re.search(r'_(\d{5,})', pt_acc).group(1)
            study_acc_dir_list.append(c_acc)

            if c_acc not in unique_acc_dir_list:
                unique_acc_dir_list.append(c_acc)

        subdir_size = len(subdir_path)
        dir_count += subdir_size
        pt_num = pt.split('_')[1]
        all_pt_list.append(pt_num)
        if subdir_size != 1:
            multi_pt_list.append(pt_num)
        else:
            single_pt_list.append(pt_num)

    print('total subdirectories/accessions: ', dir_count)
    print('single subfolder pt: ', len(single_pt_list))
    print('multiple subfolder pt: ', len(multi_pt_list))
    print('study_acc_dirs: ', len(study_acc_dir_list))  # there are 7 duplicates (one accession with 2 folders).  So in total, 48 missing cases, congruent with # of unassigned accession num.
    print('unique_study_acc_dirs: ', len(unique_acc_dir_list))

    return study_acc_dir_list, multi_pt_list, single_pt_list

def process_accession_dirs(c_path=MAIN_PATH, return_dups_paths=False):
    study_acc_dir, multi_pt_list, single_pt_list = count_dirs()

    study_acc_series = pd.Series(study_acc_dir, name='acc').map(lambda x: int(x))
    study_acc_dir_dups_list = study_acc_series[study_acc_series.duplicated()].tolist()

    dups_acc_path_list = []
    dups_series_path_list = []

    for pt in os.listdir(c_path):
        acc_path = next(os.walk(os.path.join(c_path, pt)))[1]
        for pt_acc in acc_path:
            acc_num = int(re.search(r'_(\d{5,})', pt_acc).group(1))

            if acc_num in study_acc_dir_dups_list:
                dups_acc_path_list.append(os.path.join(c_path, pt, pt_acc))

    for acc in dups_acc_path_list:
        dups_series = next(os.walk(acc))[1]

        for pt_series in dups_series:
            dups_series_path_list.append(os.path.join(acc, pt_series))

    unique_study_acc_based_on_dirs_list = study_acc_series[~study_acc_series.duplicated()].sort_values().tolist()

    if return_dups_paths:
        return dups_acc_path_list, dups_series_path_list

    else:
        return unique_study_acc_based_on_dirs_list, multi_pt_list, single_pt_list

def process_R60_request(R60_anon_df=R60_DF):
    R60_ID = R60_anon_df['ID'].tolist()
    R60_acc = R60_anon_df['Accession Number'].tolist()
    R60_dups_idx = R60_anon_df[R60_anon_df['Accession Number'].duplicated()].index #is 0; no dups here.
    R60_unique = R60_anon_df['Accession Number'].nunique()

    return R60_ID, R60_acc, R60_dups_idx, R60_unique

def process_img_map(unique_accession_dirs_list, multi_pt_list, single_pt_list, pelvic_fx_df, img_map_df):

    img_map_copy = img_map_df.copy()
    filled_img_map = img_map_copy.fillna(999)

    filled_img_map[['PT_STUDY_ID', 'ACCESSION_STUDY_ID', 'SERIES_NUMBER_DICOM']] = \
        filled_img_map[['PT_STUDY_ID', 'ACCESSION_STUDY_ID', 'SERIES_NUMBER_DICOM']].applymap(lambda x: int(x))
    filled_img_map['has_dir'] = filled_img_map['ACCESSION_STUDY_ID']\
        .isin(unique_accession_dirs_list)  # correlating the db to the dirs, showing missing 48 cases with undefined values.

    img_map_nan = filled_img_map[filled_img_map["ACCESSION_STUDY_ID"] == 999]
    non_dup_img_map = filled_img_map[~filled_img_map['Accession Number'].duplicated()]  # equal to 7677 cases, same as R60.  48 have undefined ACCESSION_STUDY_ID.

    img_map_valid = filled_img_map[filled_img_map['ACCESSION_STUDY_ID'] != 999].copy()
    valid_non_dups_imgs = img_map_valid[~img_map_valid['Accession Number'].duplicated()]  # 7629 cases, excluded 48 with undefined ACCESSION_STUDY_ID.
    img_map_unique_acc_list = valid_non_dups_imgs['Accession Number'].nunique()

    valid_multi_pt = valid_non_dups_imgs[valid_non_dups_imgs['PT_STUDY_ID'].isin(multi_pt_list)] # pts with multiple accessions
    unique_valid_mult_pt = valid_multi_pt[~valid_multi_pt['PT_STUDY_ID'].duplicated()]
    valid_single_pt = valid_non_dups_imgs[valid_non_dups_imgs['PT_STUDY_ID'].isin(single_pt_list)]

    pelvic_fx_df_copy = pelvic_fx_df.copy()

    joined_df = non_dup_img_map.join(pelvic_fx_df_copy.set_index('Accession Number'), on='Accession Number', lsuffix='_img', rsuffix='_fx').fillna(999)
    joined_df[['ID_fx', 'consold_label', 'sep_label']] = joined_df[['ID_fx', 'consold_label', 'sep_label']].applymap(lambda x: int(x))
    joined_df = joined_df.rename(columns={'ID_img': 'ID'}).drop(columns=['ID_fx', 'STUDY_DATE_DICOM'])

    joined_df['valid_map'] = joined_df['Accession Number'].isin(valid_non_dups_imgs['Accession Number'].tolist())  # is the required image valid/exist
    joined_df['valid_map'] = joined_df['valid_map'].map(lambda x: bool(x))

    return joined_df

def process_note_map(note_map_df=NOTE_MAP_DF):
    note_map_copy = note_map_df.copy()
    filled_note_map = note_map_copy.fillna(999)

    filled_note_map[['STUDY_ID', 'NOTE_ID', 'NOTE_CSN_ID', 'NOTE_CSN_STUDY_ID']] = filled_note_map[
        ['STUDY_ID', 'NOTE_ID', 'NOTE_CSN_ID', 'NOTE_CSN_STUDY_ID']].applymap(lambda x: int(x))

    note_map_nan = filled_note_map[filled_note_map["STUDY_ID"] == 999]
    note_map_nan_accession = note_map_nan['Accession Number'].tolist()  # 1021
    note_map_accession = filled_note_map['Accession Number'].tolist()
    note_map_duplicated = filled_note_map[filled_note_map['Accession Number'].duplicated()]
    non_dup_note_map = filled_note_map[~filled_note_map['Accession Number'].duplicated()]
    non_dup_note_map_accessions = filled_note_map[~filled_note_map['Accession Number'].duplicated()]['Accession Number'].tolist()

    filled_note_in_nan = filled_note_map[filled_note_map['Accession Number'].isin(note_map_nan_accession)]
    filled_note_in_nan_dups = filled_note_in_nan[filled_note_in_nan['Accession Number'].duplicated()]

    note_map_valid = filled_note_map[filled_note_map['STUDY_ID'] != 999]
    note_map_unique = note_map_valid['Accession Number'].nunique()

    print(note_map_nan.shape, filled_note_in_nan.shape, filled_note_in_nan_dups.shape, note_map_unique)

def get_final_df():
    unique_acc_dirs_list, multi_pt_list, single_pt_list = process_accession_dirs()
    final_df = process_img_map(unique_acc_dirs_list, multi_pt_list, single_pt_list, PELVIC_FX_DF, IMG_MAP_DF)

    return final_df

def rm_nonpelvis():
    for study_acc in non_pelvis_list:
        shutil.rmtree(study_acc)

def consolidate_dup_acc_dirs(modify_files=False):
    dups_acc, dups_series = process_accession_dirs(c_path=MAIN_PATH, return_dups_paths=True)

    study_ids_list = []
    prev_acc = './previous_accession'
    to_del_dups = []

    if not modify_files:
        print('Not modifying the files')

    for current_acc in dups_acc:

        study_id = (re.search(r'/.*_(\d{5,})$', current_acc).group(1))

        if study_id in study_ids_list:
            print('curr_acc: ', current_acc)
            print('prev_acc_path: ', prev_acc)
            to_del_dups.append(current_acc)
            series_dir = glob(os.path.join(current_acc, '*'))
            for single_series in series_dir:
                c_series = 'moved_' + re.search(r'/(Series.*$)', single_series).group(1)
                new_path = os.path.join(prev_acc, c_series)
                if modify_files:
                    shutil.copytree(single_series, new_path)

        study_ids_list.append(study_id)
        # study_path = re.search(r'(.*Patient_\d*)', current_acc).group(1)
        prev_acc = current_acc

    for i in to_del_dups:
        if modify_files:
            shutil.rmtree(i)

def place_SB_files(modify_files=False, del_error_files=False, save_excel=False, output_txt=False):
    final_df = get_final_df()

    valid_df = final_df[final_df['valid_map']]
    study_ids = valid_df['PT_STUDY_ID']
    accessions_ids = valid_df['ACCESSION_STUDY_ID']

    #Make sure new study_ids and accession_ids are fully above or below the current values
    #print(study_ids.min(), study_ids.max()) #7882157104 7888999786
    #print(accessions_ids.min(), accessions_ids.max()) #2000900 93697957

    new_sb_paths = []

    if not modify_files:
        print('not modifying files')

    for orig_id in os.listdir('./SB_images/'):
        orig_id = int(orig_id)
        new_pt_id = 5000000000 + orig_id
        new_accession = 1000000 + orig_id

        c_idx = final_df[final_df['ID'] == orig_id].index
        final_df.loc[c_idx, 'PT_STUDY_ID'] = new_pt_id
        final_df.loc[c_idx, 'ACCESSION_STUDY_ID'] = new_accession
        c_path = './images/PelvisImagesDICOM/Patient_{}/Study_{}/series_none/'.format(new_pt_id, new_accession)
        orig_file_path = os.path.join('./SB_images/', str(orig_id))
        new_sb_paths.append(c_path)
        if modify_files:
            shutil.copytree(orig_file_path, c_path)

    #TODO: Check new_sb_paths; probably should do before copying to other folder
    error_dcm_files = []
    path_is_dir = []

    for i in new_sb_paths:
        series_files = glob(os.path.join(i, '*'))
        for each_dcm_file in series_files:
            if os.path.isdir(each_dcm_file):
                path_is_dir.append(each_dcm_file)
                continue
            try:
                img = pydicom.dcmread(each_dcm_file).pixel_array
                # plt.imshow(img, 'gray')
                # plt.show()
            except:
                error_dcm_files.append(each_dcm_file)

    # Make sure that the failed and dir paths are saved
    with open('./post_move_fail_paths.txt', 'w') as f:
        for i in error_dcm_files:
            f.write(i + '\n')

    with open('./post_move_path_is_dirs.txt', 'w') as f:
        for i in path_is_dir:
            f.write(i + '\n')

    if del_error_files:
        for error_file in error_dcm_files:
            os.remove(error_file)

    if save_excel:
        final_df.to_excel('./after_SB_placement.xlsx')

    return final_df