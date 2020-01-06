import pydicom
import numpy as np
import skimage
import cv2
import os
import re
import pandas as pd

from glob import glob
from skimage import exposure, transform
from sklearn import preprocessing

# Inception scaling can be done at time of batch dicom resizing or augmentation pipeline

def mod_sb_placed_df(save_it=False):
    placed_final_df = pd.read_excel('./old_xlsx/after_SB_placement.xlsx')
    useful_df = placed_final_df[placed_final_df['PT_STUDY_ID'] != 999].copy()
    useful_df['acc_path'] = "placeholder"
    useful_df['full_acc_path'] = "placeholder"

    def create_acc_path(row):
        pt_id = row['PT_STUDY_ID']
        study_acc = row['ACCESSION_STUDY_ID']
        pt_dir = './images/PelvisImagesDICOM/Patient_{}'.format(pt_id)

        for dir_acc in os.listdir(pt_dir):
            if re.search(str(study_acc), dir_acc):
                return dir_acc, os.path.join(pt_dir, dir_acc)

    useful_df['acc_path'], useful_df['full_acc_path'] = zip(*useful_df.apply(lambda row: create_acc_path(row), axis=1))
    useful_df = useful_df[['ID', 'PT_STUDY_ID', 'ACCESSION_STUDY_ID',
                           'consold_label', 'sep_label', 'acc_path',
                           'full_acc_path']]

    if save_it:
        useful_df.to_excel('./compressed_useful_df.xlsx')

    return useful_df

def get_full_dicom_path(row):
    pt_id = row['PT_STUDY_ID']
    study_acc = row['ACCESSION_STUDY_ID']
    series_type = row['Series']
    dicom_num = row['dicom']
    pt_dir = './images/PelvisImagesDICOM/Patient_{}'.format(pt_id)

    for dir_acc in os.listdir(pt_dir):
        if re.search(str(study_acc), dir_acc):
            acc_path = os.path.join(pt_dir, dir_acc)

            for a_series in os.listdir(acc_path):
                if re.search(str(series_type), a_series):
                    series_path = os.path.join(acc_path, a_series)

                    for s_dicom in os.listdir(series_path):
                        short_dicom = str(dicom_num)[:8]
                        if re.search(short_dicom, s_dicom):
                            return os.path.join(series_path, s_dicom)

def create_img_from_compressed_df(inp_df, n_classes=5, preprocess='inception', np_save_file=None, conv_to_uint8=False, min_max_exposure=False, output_size=238):
    print('check_statement')

    c_df = inp_df.copy()  # copy to prevent modifying original df, use compressed_useful_df.xlsx

    images_list = []
    label_list = []
    series_list = []
    id_list = []
    out_dicom_list = []
    study_id_list = []
    accession_study_list = []

    view_list = []
    derivative_list = []

    for idx, row in c_df.iterrows():
        acc_path = row['full_acc_path']
        c_id = row['ID']
        study_id = row['PT_STUDY_ID']
        accession_study_id = row['ACCESSION_STUDY_ID']
        c_series_list = glob(os.path.join(acc_path, '*'))
        for one_series in c_series_list:
            series_name = re.search(r'/.*_(.*)$', one_series).group(1)
            dicom_list = glob(os.path.join(one_series, '*'))

            for dicom_file in dicom_list:

                a = pydicom.dcmread(dicom_file)

                try:
                    if a.DerivationDescription:  # derivative view
                        derivative_list.append(dicom_file)
                        continue

                except AttributeError: # no derivative view attribute, so move on to image creation
                    pass

                try:
                    c_view = a.ViewPosition
                    view_list.append(c_view)

                except AttributeError: # no view position info
                    pass

                # collect individual image information
                #series_list.append(str(c_id) + ': ' + series_name)

                dicom_end = re.search(r'/.*/.*/.*/.*/.*/(.*)$', dicom_file).group(1)
                series_list.append(series_name)
                out_dicom_list.append(dicom_end)
                study_id_list.append(study_id)
                accession_study_list.append(accession_study_id)

                _img = a.pixel_array
                bs = a.BitsStored

                _img = erase_borders(_img)

                if min_max_exposure:

                    _img = rescale(_img, 0, 1, np.min(_img), np.max(_img)).astype(np.float32)

                    if a.PhotometricInterpretation == 'MONOCHROME1':
                        _img = skimage.util.invert(_img, signed_float=False)

                    # All pixels ABOVE 99th percentile set TO 99th percentile (getting around super bright labels)
                    _img[_img > np.percentile(_img, 99)] = np.percentile(_img, 99)

                    # Rescaling again after pixel cutt-off
                    _img = rescale(_img, 0, 1, np.min(_img), np.max(_img)).astype(np.float32)

                    if conv_to_uint8:
                        _img = skimage.img_as_uint(_img)

                else:
                    _img = exposure.rescale_intensity(_img, in_range=('uint' + str(bs)))  # most, if not all, are of type uint16

                    if conv_to_uint8:
                        _img = skimage.img_as_uint(_img)

                    if a.PhotometricInterpretation == 'MONOCHROME1':
                        _img = cv2.bitwise_not(_img)

                _img = transform.resize(_img, (output_size, output_size), mode='reflect', anti_aliasing=True)  # img_as_float
                _img = np.stack([_img, _img, _img], axis=-1)
                images_list.append(_img)
                id_list.append(c_id)

    image_array = np.array(images_list, np.float32)
    id_array = np.array(id_list, np.int64)

    if preprocess == 'inception':
        image_array = (image_array - 0.5) * 2 # get values [-1, 1].  Shift and scale.

    # single labels as one-hot
    label_list = preprocessing.MultiLabelBinarizer(np.arange(n_classes)).fit_transform(label_list)
    label_array = np.array(label_list, np.float32)

    expanded_df = pd.DataFrame({'ID': id_list, 'PT_STUDY_ID': study_id_list, 'ACCESSION_STUDY_ID': accession_study_list, 'Series': series_list, 'dicom': out_dicom_list})

    if np_save_file is not None:
        np.savez(np_save_file, image_array=image_array, id_array=id_array)
        #np.savez('useful_globus_arrays', image_array=image_array, id_array=id_array)
        # with open('series_text_file', 'w') as f:
        #     for one_series in series_list:
        #         f.write(one_series + '\n')
        expanded_df.to_excel('./expanded_df.xlsx')

    return image_array, id_array, expanded_df

def create_img_from_full_dicom_df(inp_df, preprocess='inception', np_save_file=None, conv_to_uint8=False, min_max_exposure=False, output_size=238):

    c_df = inp_df.copy()  # copy to prevent modifying original df, use expanded_useful_df.xlsx

    images_list = []
    index_list = []
    id_list = []
    orig_idx_list = []

    view_list = []
    derivative_list = []

    for idx, row in c_df.iterrows():
        dicom_path = row['full_path']
        c_id = row['ID']
        orig_idx = row['orig_idx']
        a = pydicom.dcmread(dicom_path)

        try:
            if a.DerivationDescription:  # derivative view
                derivative_list.append(dicom_path)
                continue

        except AttributeError: # no derivative view attribute, so move on to image creation
            pass

        try:
            c_view = a.ViewPosition
            view_list.append(c_view)

        except AttributeError: # no view position info
            pass

        _img = a.pixel_array
        bs = a.BitsStored

        _img = erase_borders(_img)

        if min_max_exposure:

            _img = rescale(_img, 0, 1, np.min(_img), np.max(_img)).astype(np.float32)

            if a.PhotometricInterpretation == 'MONOCHROME1':
                _img = skimage.util.invert(_img, signed_float=False)

            # All pixels ABOVE 99th percentile set TO 99th percentile (getting around super bright labels)
            _img[_img > np.percentile(_img, 99)] = np.percentile(_img, 99)

            # Rescaling again after pixel cutt-off
            _img = rescale(_img, 0, 1, np.min(_img), np.max(_img)).astype(np.float32)

            if conv_to_uint8:
                _img = skimage.img_as_uint(_img)

        else:
            _img = exposure.rescale_intensity(_img, in_range=('uint' + str(bs)))  # most, if not all, are of type uint16

            if conv_to_uint8:
                _img = skimage.img_as_uint(_img)

            if a.PhotometricInterpretation == 'MONOCHROME1':
                _img = cv2.bitwise_not(_img)

        _img = transform.resize(_img, (output_size, output_size), mode='reflect', anti_aliasing=True)  # img_as_float
        _img = np.stack([_img, _img, _img], axis=-1)
        images_list.append(_img)
        index_list.append(idx)
        id_list.append(c_id)
        orig_idx_list.append(orig_idx)

    image_array = np.array(images_list, np.float32)
    index_array = np.array(index_list, np.int64)
    id_array = np.array(id_list, np.int64)
    orig_idx_array = np.array(orig_idx_list, np.int64)

    if preprocess == 'inception':
        image_array = (image_array - 0.5) * 2 # get values [-1, 1].  Shift and scale.

    if np_save_file is not None:
        np.savez(np_save_file, image_array=image_array, id_array=id_array, orig_idx_array=orig_idx_array)

    return image_array, id_array, orig_idx_array

def rescale(array, new_min, new_max, old_min, old_max):
    return (array - old_min) * (new_max - new_min) / (old_max - old_min) + new_min

def erase_borders(current_image):

    # checking the rows from the top
    for i in range (current_image.shape[0]):
        pixel_check = current_image[i][0]

        # Need to break out of loop when pixels are no longer homogeneous across row
        if np.mean(current_image[i] == pixel_check) != 1:
            current_image = current_image[i:]
            break

    # checking rows from the bottom
    for i in range(current_image.shape[0]-1, 0, -1): # subtract one at starting point to prevent IndexError
        pixel_check = current_image[i][0]

        if np.mean(current_image[i] == pixel_check) != 1:
            current_image = current_image[:i]
            break

    # checking columns from one side
    for i in range(current_image.shape[1]):
        pixel_check = current_image[0][i]

        # Need to break out of loop when pixels are not longer homogeneous through column
        if np.mean(current_image[:, i] == pixel_check) != 1:
            current_image = current_image[:, i:]
            break

    # checking columns from the other side
    for i in range(current_image.shape[1] - 1, 0, -1):
        pixel_check = current_image[0][i]

        if np.mean(current_image[:, i] == pixel_check) != 1:
            current_image = current_image[:, :i]
            break

    return current_image

def consolidate_positions():
    positions_df = pd.read_excel('./unlabeled_positions.xlsx').fillna('negative')

    hardware_list = ['RF', 'LAR', 'LF', 'BAR', 'RAR', 'Pelvic', 'EX', 'BIM', 'IM', 'Multiple']  # no LUM or PUMP
    arthro_list = ['LAR', 'BAR', 'RAR', 'Multiple']
    pelvic_pos = ['AP', 'LAO', 'RAO', 'OUT', 'IN']
    lao_pos = ['LAO']
    rao_pos = ['RAO']
    hip_pos = ['RH', 'RF', 'LH', 'CTL', 'LF']
    ctl_pos = ['CTL']
    right_hip = ['RH', 'RF']
    left_hip = ['LH', 'LF']
    fail_pos = ['FAIL', 'FEM']
    exclude_pos = ['EXCLUDE']

    positions_df['has_hardware'] = positions_df['Hardware'].isin(hardware_list)
    positions_df['arthro'] = positions_df['Hardware'].isin(arthro_list)
    positions_df['is_pelvis'] = positions_df['Label'].isin(pelvic_pos)
    positions_df['lao'] = positions_df['Label'].isin(lao_pos)
    positions_df['rao'] = positions_df['Label'].isin(rao_pos)
    positions_df['is_hip'] = positions_df['Label'].isin(hip_pos)
    positions_df['right_hip'] = positions_df['Label'].isin(right_hip)
    positions_df['left_hip'] = positions_df['Label'].isin(left_hip)
    positions_df['ctl_pos'] = positions_df['Label'].isin(ctl_pos)
    positions_df['fail_pos'] = positions_df['Label'].isin(fail_pos)
    positions_df['exclude'] = positions_df['Label'].isin(exclude_pos)

    return positions_df

def load_df_and_npz(check_intersection=True, save_filtered=None):
    full_positions_labeled = pd.read_excel('./full_labeled_positions.xlsx')

    #Easier to work with after excluding items
    df_exclude_idx = full_positions_labeled[full_positions_labeled['exclude']].index.tolist()
    more_excludes_orig_idx = [838, 1415, 2299, 6779, 6876, 7208, 7452, 9019, 9087, 9168, 9170, 9351, 9639, 10278]
    more_excludes_idx  = full_positions_labeled[full_positions_labeled['orig_idx'].isin(more_excludes_orig_idx)].index.tolist()
    full_excludes_idx = df_exclude_idx + more_excludes_idx

    filtered_df = full_positions_labeled.loc[~full_positions_labeled.index.isin(full_excludes_idx)].copy()
    filtered_df = filtered_df.reset_index(drop=True)

    hardware_idx = filtered_df[filtered_df['has_hardware']].index.tolist()
    nonhardware_idx = filtered_df[~filtered_df['has_hardware']].index.tolist()
    arthro_idx = filtered_df[filtered_df['arthro']].index.tolist()
    pelvis_idx = filtered_df[filtered_df['is_pelvis']].index.tolist()
    lao_idx = filtered_df[filtered_df['lao']].index.tolist()
    rao_idx = filtered_df[filtered_df['rao']].index.tolist()
    hip_idx = filtered_df[filtered_df['is_hip']].index.tolist()
    right_hip_idx = filtered_df[filtered_df['right_hip']].index.tolist()
    left_hip_idx = filtered_df[filtered_df['left_hip']].index.tolist()
    ctl_idx = filtered_df[filtered_df['ctl_pos']].index.tolist()
    fail_idx = filtered_df[filtered_df['fail_pos']].index.tolist()

    filtered_df['position_label'] = 'placeholder'
    filtered_df['hardware_label'] = 'placeholder'
    filtered_df.loc[pelvis_idx, 'position_label'] = 0
    filtered_df.loc[hip_idx, 'position_label'] = 1
    filtered_df.loc[fail_idx, 'position_label'] = 2
    filtered_df.loc[nonhardware_idx, 'hardware_label'] = 0
    filtered_df.loc[hardware_idx, 'hardware_label'] = 1

    pelvis_len = len(pelvis_idx)
    pelvis_train_n = np.floor(pelvis_len * .7).astype(np.int64)
    pelvis_val_n = pelvis_len - pelvis_train_n

    hip_len = len(hip_idx)
    hip_train_n = np.floor(hip_len * .7).astype(np.int64)
    hip_val_n = hip_len - hip_train_n

    fail_len = len(fail_idx)
    fail_train_n = np.floor(fail_len * .7).astype(np.int64)
    fail_val_n = fail_len - fail_train_n

    hardware_len = len(hardware_idx)
    hardware_train_n = np.floor(hardware_len * .7).astype(np.int64)
    hardware_val_n = hardware_len - hardware_train_n

    nonhardware_len = len(nonhardware_idx)
    nonhardware_train_n = np.floor(nonhardware_len * .7).astype(np.int64)
    nonhardware_val_n = nonhardware_len - nonhardware_train_n

    pelvis_train_idx = pelvis_idx[:pelvis_train_n]
    pelvis_val_idx = pelvis_idx[pelvis_train_n:]
    hip_train_idx = hip_idx[:hip_train_n]
    hip_val_idx = hip_idx[hip_train_n:]
    fail_train_idx = fail_idx[:fail_train_n]
    fail_val_idx = fail_idx[fail_train_n:]

    hardware_train_idx = hardware_idx[:hardware_train_n]
    hardware_val_idx = hardware_idx[hardware_train_n:]
    nonhardware_train_idx = nonhardware_idx[:nonhardware_train_n]
    nonhardware_val_idx = nonhardware_idx[nonhardware_train_n:]

    concat_position_train_idx = pelvis_train_idx + hip_train_idx + fail_train_idx
    concat_position_val_idx = pelvis_val_idx + hip_val_idx + fail_val_idx
    concat_hardware_train_idx = hardware_train_idx + nonhardware_train_idx
    concat_hardware_val_idx = hardware_val_idx + nonhardware_val_idx

    filtered_df['pelvis_train'] = filtered_df.index.isin(pelvis_train_idx)
    filtered_df['pelvis_val'] = filtered_df.index.isin(pelvis_val_idx)
    filtered_df['hip_train'] = filtered_df.index.isin(hip_train_idx)
    filtered_df['hip_val'] = filtered_df.index.isin(hip_val_idx)
    filtered_df['fail_train'] = filtered_df.index.isin(fail_train_idx)
    filtered_df['fail_val'] = filtered_df.index.isin(fail_val_idx)

    filtered_df['position_train'] = filtered_df.index.isin(concat_position_train_idx)
    filtered_df['position_val'] = filtered_df.index.isin(concat_position_val_idx)
    filtered_df['hardware_train'] = filtered_df.index.isin(concat_hardware_train_idx)
    filtered_df['hardware_val'] = filtered_df.index.isin(concat_hardware_val_idx)

    # shortened_df = filtered_df[['orig_idx', 'ACCESSION_STUDY_ID', 'ID', 'PT_STUDY_ID', 'Series',
    #                             'dicom', 'unfiltered_idx', 'has_hardware', 'is_pelvis',
    #                             'is_hip', 'fail_pos', 'position_train', 'position_val', 'hardware_train',
    #                             'hardware_val', 'position_label', 'hardware_label']].copy()

    shortened_df = filtered_df

    if check_intersection:
        c_train_idx = shortened_df[shortened_df['position_train']].index
        c_val_idx = shortened_df[shortened_df['position_val']].index
        print(c_train_idx.intersection(c_val_idx))

        d_train_idx = shortened_df[shortened_df['hardware_train']].index
        d_val_idx = shortened_df[shortened_df['hardware_val']].index
        print(d_train_idx.intersection(d_val_idx))



    shortened_df['full_path'] = shortened_df.apply(get_full_dicom_path, axis=1)
    assert shortened_df[shortened_df['full_path'].map(lambda x: len(x)) < 30].shape != 0, 'DICOM path not full for some paths'

    if save_filtered is not None:
        print('saving files')
        shortened_df.to_excel('post_exclude_filtered_df.xlsx')

    return shortened_df