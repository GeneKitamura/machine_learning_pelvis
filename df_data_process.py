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

from dicom_process import load_df_and_npz, create_img_from_full_dicom_df
from image_tools import rescale_img
from load_data import Data

def precreate_npz():
    final_filtered_df = pd.read_excel('./post_exclude_filtered_df.xlsx')

    pos_train_df = final_filtered_df[final_filtered_df['position_train']]
    c_images, c_ids, c_orig_idx = create_img_from_full_dicom_df(pos_train_df, output_size=238, np_save_file='./238_pos_train')
    pos_train_labels = pos_train_df['position_label'].values
    np.save('./238_pos_train_labels', pos_train_labels)

    pos_val_df = final_filtered_df[final_filtered_df['position_val']]
    c_images, c_ids, c_orig_idx = create_img_from_full_dicom_df(pos_val_df, output_size=224, np_save_file='./224_pos_val')
    pos_val_labels = pos_val_df['position_label'].values
    np.save('./224_pos_val_labels', pos_val_labels)

    hardware_train_df = final_filtered_df[final_filtered_df['hardware_train']]
    c_images, c_ids, c_orig_idx = create_img_from_full_dicom_df(hardware_train_df, output_size=238, np_save_file='./238_hardware_train')
    hardware_train_labels = hardware_train_df['hardware_label'].values
    np.save('./238_hardware_train_labels', hardware_train_labels)

    hardware_val_df = final_filtered_df[final_filtered_df['hardware_val']]
    c_images, c_ids, c_orig_idx = create_img_from_full_dicom_df(hardware_val_df, output_size=224, np_save_file='./224_hardware_val')
    hardware_val_labels = hardware_val_df['hardware_label'].values
    np.save('./224_hardware_val_labels', hardware_val_labels)

    pos_and_hardware_val = final_filtered_df[final_filtered_df['position_val'] & final_filtered_df['hardware_val']]
    c_images, c_ids, c_orig_idx = create_img_from_full_dicom_df(pos_and_hardware_val, output_size=224, np_save_file='./224_combo_pos_and_hardware_val')
    combo_pos_val_labels = pos_and_hardware_val['position_label'].values
    combo_hardware_val_labels = pos_and_hardware_val['hardware_label'].values
    np.savez('./224_combo_pos_and_hardware_val_lables', pos_labels=combo_pos_val_labels, hardware_labels=combo_hardware_val_labels)


def consolidate_imgs_from_df():

    with np.load('./238_pos_train.npz') as f:
        pos_train_images = f['image_array']
        pos_train_ids = f['id_array']
        pos_train_orig_idx = f['orig_idx_array']
    pos_train_labels = np.load('./238_pos_train_labels.npy')

    with np.load('./224_pos_val.npz') as f:
        pos_val_images_224 = f['image_array']
        pos_val_ids_224 = f['id_array']
        pos_val_orig_idx_224 = f['orig_idx_array']
    pos_val_labels_224 = np.load('./224_pos_val_labels.npy')

    with np.load('./238_hardware_train.npz') as f:
        hardware_train_images = f['image_array']
        hardware_train_ids = f['id_array']
        hardware_train_orig_idx = f['orig_idx_array']
    hardware_train_labels = np.load('./238_hardware_train_labels.npy')

    with np.load('./224_hardware_val.npz') as f:
        hardware_val_images_224 = f['image_array']
        hardware_val_ids_224 = f['id_array']
        hardware_val_orig_idx_224 = f['orig_idx_array']
    hardware_val_labels_224 = np.load('./224_hardware_val_labels.npy')

    with np.load('./224_combo_pos_and_hardware_val.npz') as f:
        pos_and_hardware_val_images_224 = f['image_array']
        pos_and_hardware_val_ids_224 = f['id_array']
        pos_and_hardware_val_orig_idx_224 = f['orig_idx_array']
    with np.load('./224_combo_pos_and_hardware_val_lables.npz') as f:
        combo_pos_val_labels_224 = f['pos_labels']
        combo_hardware_val_labels_224 = f['hardware_labels']

    n_cxr = 100

    with np.load('./densenet_sized_arrays/ptx_cxr_images.npz') as f:
        # 400 cases each
        cxr_238_cal = f['kolo_238_cal'][:n_cxr]
        cxr_238_cal = rescale_img(cxr_238_cal, new_min=-1, new_max=1)
        cxr_224_val = f['kolo_224_val'][:n_cxr]
        cxr_224_val = rescale_img(cxr_224_val, new_min=-1, new_max=1)

    cxr_238_cal_labels = np.repeat([3], n_cxr)
    cxr_224_val_labels = np.repeat([3], n_cxr)

    total_pos_train_images = np.concatenate((pos_train_images, cxr_238_cal), axis=0)
    total_pos_train_labels = np.concatenate((pos_train_labels, cxr_238_cal_labels), axis=0)
    total_pos_val_images = np.concatenate((pos_val_images_224, cxr_224_val), axis=0)
    total_pos_val_labels = np.concatenate((pos_val_labels_224, cxr_224_val_labels), axis=0)

    np.savez('./pos_and_cxr_train', image_array=total_pos_train_images, label_array=total_pos_train_labels)
    np.savez('./pos_and_cxr_val', image_array=total_pos_val_images, label_array=total_pos_val_labels)


def read_precreated_npz():
    total_pos_labeler = preprocessing.MultiLabelBinarizer(np.arange(4)) # pelvis, hip, fail, CXR
    total_hardware_labeler = preprocessing.MultiLabelBinarizer(np.arange(2)) #yes, no

    with np.load('./pos_and_cxr_train.npz') as f:
        total_pos_train_images = f['image_array']
        total_pos_train_labels = f['label_array']
    hot_total_pos_train_labels = total_pos_labeler.fit_transform(total_pos_train_labels.reshape(total_pos_train_labels.shape[0], 1))
    np.savez('./hot_pos_and_cxr_train', image_array=total_pos_train_images, label_array=hot_total_pos_train_labels)

    with np.load('./pos_and_cxr_val.npz') as f:
        total_pos_val_images = f['image_array']
        total_pos_val_labels = f['label_array']
    hot_total_pos_val_labels = total_pos_labeler.fit_transform(total_pos_val_labels.reshape(total_pos_val_labels.shape[0], 1))
    np.savez('./hot_pos_and_cxr_val', image_array=total_pos_val_images, label_array=hot_total_pos_val_labels)

    with np.load('./238_hardware_train.npz') as f:
        hardware_train_images = f['image_array']
        hardware_train_ids = f['id_array']
        hardware_train_orig_idx = f['orig_idx_array']
    hardware_train_labels = np.load('./238_hardware_train_labels.npy')
    hot_hardware_train_labels = total_hardware_labeler.fit_transform(hardware_train_labels.reshape(hardware_train_labels.shape[0], 1))
    np.savez('hot_238_hardware_train', image_array=hardware_train_images, label_array=hot_hardware_train_labels)

    with np.load('./224_hardware_val.npz') as f:
        hardware_val_images_224 = f['image_array']
        hardware_val_ids_224 = f['id_array']
        hardware_val_orig_idx_224 = f['orig_idx_array']
    hardware_val_labels_224 = np.load('./224_hardware_val_labels.npy')
    hot_hardware_val_labels = total_hardware_labeler.fit_transform(hardware_val_labels_224.reshape(hardware_val_labels_224.shape[0], 1))
    np.savez('hot_224_hardware_val', image_array=hardware_val_images_224, label_array=hot_hardware_val_labels)

    with np.load('./224_combo_pos_and_hardware_val.npz') as f:
        pos_and_hardware_val_images_224 = f['image_array']
        pos_and_hardware_val_ids_224 = f['id_array']
        pos_and_hardware_val_orig_idx_224 = f['orig_idx_array']
    with np.load('./224_combo_pos_and_hardware_val_lables.npz') as f:
        combo_pos_val_labels_224 = f['pos_labels']
        combo_hardware_val_labels_224 = f['hardware_labels']
    hot_combo_pos_val_labels = total_pos_labeler.fit_transform(combo_pos_val_labels_224.reshape(combo_pos_val_labels_224.shape[0], 1))
    hot_combo_hardware_val_labels = total_hardware_labeler.fit_transform(combo_hardware_val_labels_224.reshape(combo_hardware_val_labels_224.shape[0], 1))
    np.savez('./hot_224_combo_pos_val', image_array=pos_and_hardware_val_images_224, label_array=hot_combo_pos_val_labels)
    np.savez('./hot_224_combo_hardware_val', image_array=pos_and_hardware_val_images_224, label_array=hot_combo_hardware_val_labels)

def read_hot_npz(c_path):
    with np.load(c_path) as f:
        images = f['image_array']
        labels = f['label_array']

    return Data(images, labels)
