import pandas as pd
import numpy as np
import re
import tensorflow as tf

from sklearn import preprocessing

from dicom_process import get_full_dicom_path, create_img_from_full_dicom_df
from utils import array_print
from custom_functions import custom_tile_alt_imshow
from image_tools import double_array_tile

keras = tf.keras
Layers = keras.layers.Layer
Model = keras.models.Model
Metrics = keras.metrics

def prepare_fx_excel(print_shapes=False):
    expanded_df = pd.read_excel('./expanded_useful_df.xlsx')
    sep_labels =  {'neg': 0, 'only_pubis': 1, 'pos_pelvis': 2, 'femur': 4, 'ace_disloc': 5, 'complex': 6}
    consold_labels = {'neg': 0, 'pelvic_ring': 3, 'femur': 4, 'ace_disloc': 5, 'complex': 6}

    expanded_df['full_path'] = expanded_df.apply(lambda x: get_full_dicom_path(x), axis=1)
    assert expanded_df[expanded_df['full_path'].map(lambda x: len(x)) < 30].shape != 0, 'DICOM path not full for some paths'

    expanded_df['orig_idx'] = expanded_df.index

    classified_df = expanded_df[expanded_df['sep_label'] != 999].copy()

    zero_idx = classified_df[classified_df['sep_label'] == 0].index.tolist()
    one_idx = classified_df[classified_df['sep_label'] == 1].index.tolist()
    two_idx = classified_df[classified_df['sep_label'] == 2].index.tolist()
    four_idx = classified_df[classified_df['sep_label'] == 4].index.tolist()
    five_idx = classified_df[classified_df['sep_label'] == 5].index.tolist()
    six_idx = classified_df[classified_df['sep_label'] == 6].index.tolist()

    zero_len = len(zero_idx)
    one_len = len(one_idx)
    two_len = len(two_idx)
    four_len = len(four_idx)
    five_len = len(five_idx)
    six_len = len(six_idx)

    zero_train_n = np.floor(zero_len * .7).astype(np.int64)
    one_train_n = np.floor(one_len * .7).astype(np.int64)
    two_train_n = np.floor(two_len * .7).astype(np.int64)
    four_train_n = np.floor(four_len * .7).astype(np.int64)
    five_train_n = np.floor(five_len * .7).astype(np.int64)
    six_train_n = np.floor(six_len * .7).astype(np.int64)

    zero_train = zero_idx[:zero_train_n]
    zero_test = zero_idx[zero_train_n:]
    one_train = one_idx[:one_train_n]
    one_test = one_idx[one_train_n:]
    two_train = two_idx[:two_train_n]
    two_test = two_idx[two_train_n:]
    four_train = four_idx[:four_train_n]
    four_test = four_idx[four_train_n:]
    five_train = five_idx[:five_train_n]
    five_test = five_idx[five_train_n:]
    six_train = six_idx[:six_train_n]
    six_test = six_idx[six_train_n:]

    concat_train_idx = zero_train + one_train + two_train + four_train + five_train + six_train
    concat_test_idx = zero_test + one_test + two_test + four_test + five_test + six_test

    classified_df['train'] = classified_df.index.isin(concat_train_idx)
    classified_df['test'] = classified_df.index.isin(concat_test_idx)

    if print_shapes:
        n = [0,1,2,4,5,6]
        for i in n:
            print('Class {}'.format(i))
            array_print(classified_df[classified_df['sep_label'] == i], classified_df[(classified_df['sep_label'] == i ) & classified_df['train']], classified_df[(classified_df['sep_label'] == i ) & classified_df['test']])

        print('Train')
        array_print(classified_df[classified_df['train']])
        print('Test')
        array_print(classified_df[classified_df['test']])

    return classified_df

def create_FX_npz():
    classified_df = prepare_fx_excel()

    a, b, c = create_img_from_full_dicom_df(classified_df[classified_df['train']], output_size=224, np_save_file='class_224_train')
    a, b, c = create_img_from_full_dicom_df(classified_df[classified_df['train']], output_size=238, np_save_file='class_238_train')
    a, b, c = create_img_from_full_dicom_df(classified_df[classified_df['test']], output_size=224, np_save_file='class_224_test')
    a, b, c = create_img_from_full_dicom_df(classified_df[classified_df['test']], output_size=238, np_save_file='class_238_test')

def label_npzs(file_list, save_it=True):
    binarizer = preprocessing.MultiLabelBinarizer(classes=np.arange(7))

    classified_df = prepare_fx_excel()
    label_category = 'sep_label'
    id_labeler = classified_df[['ID', label_category]].set_index('ID').to_dict()[label_category]
    v_map = np.vectorize(lambda x: id_labeler[x])

    for i in file_list:
        print(i)
        with np.load(i) as f:
            image_array = f['image_array']
            id_array = f['id_array']
            orig_idx_array = f['orig_idx_array']

        label_array = v_map(id_array)
        hot_label_array = binarizer.fit_transform(label_array.reshape(label_array.shape[0], 1))

        f_name = re.search(r'(.*).npz', i).group(1) + "_hotlabel"
        if save_it:
            np.savez(f_name, image_array=image_array, id_array=id_array, label_array=hot_label_array, orig_idx_array=orig_idx_array)

def read_npz_hotlabel(file_name):
    with np.load(file_name) as f:
        image_array = f['image_array']
        id_array = f['id_array']
        label_array = f['label_array']
        orig_idx_array = f['orig_idx_array']

    return image_array, id_array, label_array, orig_idx_array

def clean_with_pos_model():
    a_imgs, a_ids, a_labels, a_idx = read_npz_hotlabel('./class_238_train_hotlabel.npz')
    # a_imgs, a_ids, a_labels, a_idx = read_npz_hotlabel('./class_224_train_hotlabel.npz')

    pos_model = keras.models.load_model('./pos_and_hardware_models/pos_train/final_dense_model')
    model = pos_model

    trunc_imgs = a_imgs[:, 7:231, 7:231, :] #truncate to model size of 224
    #use cropped 238 to filter non-pelvis to exclude more borderline cases.

    preds = model.predict(trunc_imgs)
    preds_max = np.argmax(preds, axis=1)
    pred_bool = (preds_max == 0) #only pelvis x-rays

    array_print(a_imgs, a_imgs[pred_bool], a_ids[pred_bool], a_labels[pred_bool], a_idx[pred_bool])

    np.savez('pelvis_only_238_train_hot.npz', image_array=a_imgs[pred_bool], id_array=a_ids[pred_bool], label_array=a_labels[pred_bool], orig_idx_array=a_idx[pred_bool])
    # np.savez('pelvis_only_224_train_hot.npz', image_array=a_imgs[pred_bool], id_array=a_ids[pred_bool], label_array=a_labels[pred_bool], orig_idx_array=a_idx[pred_bool])

    #check images
    start_i = 0
    step_n = 25
    eval_hardware=False

    custom_tile_alt_imshow(a_imgs[pred_bool][start_i: (start_i + step_n)], prob_array=preds[pred_bool][start_i: (start_i + step_n)], hard_bool_array=([eval_hardware]*len(a_imgs[pred_bool]))[start_i: (start_i + step_n)])

def check_hardware_with_model(train=True):
    # Use uncropped images for this

    hard_model = keras.models.load_model('./pos_and_hardware_models/hardware_train/final_dense_model')
    model = hard_model

    # First to trains, then the test dataset

    if train:
        cases_224 = './pelvis_only_224_train_hot.npz'
        cases_238 = './pelvis_only_238_train_hot.npz'

    else:
        cases_224 = './pelvis_only_224_test_hot.npz'
        cases_238 = './pelvis_only_238_test_hot.npz'

    a_imgs, a_ids, a_labels, a_idx = read_npz_hotlabel(cases_224)
    b_imgs, b_ids, b_labels, b_idx = read_npz_hotlabel(cases_238)

    preds = model.predict(a_imgs)
    preds_max = np.argmax(preds, axis=1)
    pred_bool = (preds_max == 0)
    else_pred_bool = (preds_max != 0)

    array_print(a_imgs, a_imgs[pred_bool], a_ids[pred_bool], a_labels[pred_bool], a_idx[pred_bool])
    a_new_name = re.search(r'(.*).npz', cases_224).group(1) + '_nonhardware'
    b_new_name = re.search(r'(.*).npz', cases_238).group(1) + '_nonhardware'

    # check to make sure 224 and 238 hardware cases are the same
    double_array_tile(b_imgs[else_pred_bool], a_imgs[else_pred_bool])

    np.savez(a_new_name, image_array=a_imgs[pred_bool], id_array=a_ids[pred_bool], label_array=a_labels[pred_bool], orig_idx_array=a_idx[pred_bool])
    np.savez(b_new_name, image_array=b_imgs[pred_bool], id_array=b_ids[pred_bool], label_array=b_labels[pred_bool], orig_idx_array=b_idx[pred_bool])


    #check images
    start_i = 0
    step_n = 25
    eval_hardware=True

    custom_tile_alt_imshow(a_imgs[pred_bool][start_i: (start_i + step_n)], prob_array=preds[pred_bool][start_i: (start_i + step_n)], hard_bool_array=([eval_hardware]*len(a_imgs[pred_bool]))[start_i: (start_i + step_n)])

def change_labels(change_6_to_3=False, grouped=False, fem_neck_only=False, consold=False, fem_vs_nonfem=False):

    # NL vs. ant_pelvis vs. pos_pelvis vs. ace_disloc vs. femur vs. complex (6 classes)
    if change_6_to_3:
        #change class 6 to class 3 to "fill gap"
        binarizer = preprocessing.MultiLabelBinarizer(classes=np.arange(6))
        for i in ['./pelvis_only_224_test_hot.npz', 'pelvis_only_224_train_hot.npz', 'pelvis_only_238_test_hot.npz', 'pelvis_only_238_train_hot.npz']:
            image_array, id_array, hot_label_array, orig_idx_array = read_npz_hotlabel(i)
            label_maxed = np.argmax(hot_label_array, axis=1)
            new_label = np.where(label_maxed == 6, 3, label_maxed)
            new_label = binarizer.fit_transform(new_label.reshape(new_label.shape[0], 1))

            np.savez(i, image_array=image_array, id_array=id_array, label_array=new_label, orig_idx_array=orig_idx_array)

        a_imgs, a_ids, a_labels, a_idx = read_npz_hotlabel('./pelvis_only_224_train_hot.npz')
        b_imgs, b_ids, b_labels, b_idx = read_npz_hotlabel('./pelvis_only_224_test_hot.npz')

        for i in range(7):
            print(i, a_labels[np.argmax(a_labels, axis=1) == i].shape)

            # 0(2185, 6)
            # 1(515, 6)
            # 2(82, 6)
            # 3(544, 6) # Originally class 6, but change to class 3 (complex cases) since class 3 is empty for us
            # 4(1331, 6)
            # 5(292, 6)

        for i in range(7):
            print(i, b_labels[np.argmax(b_labels, axis=1) == i].shape)

            # 0(1305, 6)
            # 1(230, 6)
            # 2(41, 6)
            # 3(229, 6) # Originally class 6, but change to class 3 (complex cases) since class 3 is empty for us
            # 4(639, 6)
            # 5(127, 6)

        a_imgs, a_ids, a_labels, a_idx = read_npz_hotlabel('./pelvis_only_238_train_hot_nonhardware.npz')
        b_imgs, b_ids, b_labels, b_idx = read_npz_hotlabel('./pelvis_only_224_test_hot_nonhardware.npz')

        for i in range(7):
            print(i, a_labels[np.argmax(a_labels, axis=1) == i].shape)

            # 0(2154, 6)
            # 1(495, 6)
            # 2(82, 6)
            # 3(536, 6)
            # 4(1282, 6)
            # 5(285, 6)

        for i in range(7):
            print(i, b_labels[np.argmax(b_labels, axis=1) == i].shape)

            # 0(1274, 6)
            # 1(218, 6)
            # 2(41, 6)
            # 3(225, 6)
            # 4(620, 6)
            # 5(125, 6)

    # NL vs. pelvic ring vs. ace_fem_disloc vs. complex (4 classes)
    elif grouped:
        binarizer = preprocessing.MultiLabelBinarizer(classes=np.arange(4))

        # to get the new labels based on IDs
        recolored_df = pd.read_excel('./old_xlsx/new_recolored_df.xlsx')
        id_labeler = recolored_df[['ID', 'grouped_label']].set_index('ID', drop=True).to_dict()['grouped_label']
        v_map = np.vectorize(lambda x: id_labeler[x])

        for i in ['./pelvis_only_224_test_hot_nonhardware.npz', 'pelvis_only_238_train_hot_nonhardware.npz']:
            image_array, id_array, hot_label_array, orig_idx_array = read_npz_hotlabel(i)

            new_label_array = v_map(id_array)
            hot_new_label = binarizer.fit_transform(new_label_array.reshape(new_label_array.shape[0], 1))

            new_name = re.search(r'(.*).npz', i).group(1) + '_grouped'
            np.savez(new_name, image_array=image_array, id_array=id_array, label_array=hot_new_label,orig_idx_array=orig_idx_array)

        a_imgs, a_ids, a_labels, a_idx = read_npz_hotlabel('./pelvis_only_224_train_hot_nonhardware_grouped.npz')
        b_imgs, b_ids, b_labels, b_idx = read_npz_hotlabel('./pelvis_only_224_test_hot_nonhardware_grouped.npz')

        for i in range(4):
            print(i, a_labels[np.argmax(a_labels, axis=1) == i].shape)

            # 0(2154, 4)
            # 1(898, 4)
            # 2(1585, 4)
            # 3(197, 4)

        for i in range(4):
            print(i, b_labels[np.argmax(b_labels, axis=1) == i].shape)

            # 0(1274, 4)
            # 1(391, 4)
            # 2(760, 4)
            # 3(78, 4)

    # NL vs. pelvic_ring vs. fem vs. ace_disloc vs. complex (5 classes)
    elif consold:
        binarizer = preprocessing.MultiLabelBinarizer(classes=np.arange(5))

        # to get the new labels based on IDs
        recolored_df = pd.read_excel('./old_xlsx/new_recolored_df.xlsx')
        id_labeler = recolored_df[['ID', 'consold_label']].set_index('ID', drop=True).to_dict()['consold_label']
        v_map = np.vectorize(lambda x: id_labeler[x])

        for i in ['./pelvis_only_224_test_hot_nonhardware.npz', 'pelvis_only_238_train_hot_nonhardware.npz']:
            image_array, id_array, hot_label_array, orig_idx_array = read_npz_hotlabel(i)

            new_label_array = v_map(id_array)

            # Need to change labels to go from 0 to 4, since they are currently [0, 3, 4, 5, 6]
            new_label_array = np.where(new_label_array == 3, 1, new_label_array) # pelvic ring
            new_label_array = np.where(new_label_array == 4, 2, new_label_array) # femur
            new_label_array = np.where(new_label_array == 5, 3, new_label_array) # ace_dislocation
            new_label_array = np.where(new_label_array == 6, 4, new_label_array) # complex
            hot_new_label = binarizer.fit_transform(new_label_array.reshape(new_label_array.shape[0], 1))

            new_name = re.search(r'(.*).npz', i).group(1) + '_consold'
            np.savez(new_name, image_array=image_array, id_array=id_array, label_array=hot_new_label,
                     orig_idx_array=orig_idx_array)

        a_imgs, a_ids, a_labels, a_idx = read_npz_hotlabel('./pelvis_only_238_train_hot_nonhardware_consold.npz')
        b_imgs, b_ids, b_labels, b_idx = read_npz_hotlabel('./pelvis_only_224_test_hot_nonhardware_consold.npz')

        for i in range(5):
            print(i, a_labels[np.argmax(a_labels, axis=1) == i].shape)

            # 0(2154, 5)
            # 1(898, 5)
            # 2(1282, 5)
            # 3(285, 5)
            # 4(215, 5)


        for i in range(5):
            print(i, b_labels[np.argmax(b_labels, axis=1) == i].shape)

            # 0(1274, 5)
            # 1(391, 5)
            # 2(620, 5)
            # 3(125, 5)
            # 4(93, 5)

    # NL vs. fem vs. non_fem vs. complex (4 classes)
    elif fem_vs_nonfem:
        binarizer = preprocessing.MultiLabelBinarizer(classes=np.arange(4))

        # to get the new labels based on IDs
        recolored_df = pd.read_excel('./old_xlsx/new_recolored_df.xlsx')
        id_labeler = recolored_df[['ID', 'fem_vs_nonfem']].set_index('ID', drop=True).to_dict()['fem_vs_nonfem']
        v_map = np.vectorize(lambda x: id_labeler[x])

        for i in ['./pelvis_only_224_test_hot_nonhardware.npz', 'pelvis_only_238_train_hot_nonhardware.npz']:
            image_array, id_array, hot_label_array, orig_idx_array = read_npz_hotlabel(i)

            new_label_array = v_map(id_array)
            hot_new_label = binarizer.fit_transform(new_label_array.reshape(new_label_array.shape[0], 1))

            new_name = re.search(r'(.*).npz', i).group(1) + '_femNonfem'
            np.savez(new_name, image_array=image_array, id_array=id_array, label_array=hot_new_label,orig_idx_array=orig_idx_array)

        a_imgs, a_ids, a_labels, a_idx = read_npz_hotlabel('./pelvis_only_238_train_hot_nonhardware_femNonfem.npz')
        b_imgs, b_ids, b_labels, b_idx = read_npz_hotlabel('./pelvis_only_224_test_hot_nonhardware_femNonfem.npz')

        for i in range(4):
            print(i, a_labels[np.argmax(a_labels, axis=1) == i].shape)

            # 0(2154, 4)
            # 1(1282, 4)
            # 2(1316, 4)
            # 3(82, 4)

        for i in range(4):
            print(i, b_labels[np.argmax(b_labels, axis=1) == i].shape)

            # 0(1274, 4)
            # 1(620, 4)
            # 2(575, 4)
            # 3(34, 4)

    # Only NL vs fem fracture (2 classes)
    elif fem_neck_only:
        binarizer = preprocessing.MultiLabelBinarizer(classes=np.arange(2))
        for i in ['./pelvis_only_224_test_hot_nonhardware.npz', 'pelvis_only_238_train_hot_nonhardware.npz']:
            image_array, id_array, hot_label_array, orig_idx_array = read_npz_hotlabel(i)
            label_maxed = np.argmax(hot_label_array, axis=1)
            label_bool = (label_maxed == 4) | (label_maxed == 0)

            image_array = image_array[label_bool]
            id_array = id_array[label_bool]
            label_maxed = label_maxed[label_bool]
            orig_idx_array = orig_idx_array[label_bool]

            new_label = np.where(label_maxed == 4, 1, label_maxed)
            new_label = binarizer.fit_transform(new_label.reshape(new_label.shape[0], 1))

            new_name = re.search(r'(.*).npz', i).group(1) + '_FemNeck'
            np.savez(new_name, image_array=image_array, id_array=id_array, label_array=new_label,orig_idx_array=orig_idx_array)

        a_imgs, a_ids, a_labels, a_idx = read_npz_hotlabel('./pelvis_only_238_train_hot_nonhardware_FemNeck.npz')
        b_imgs, b_ids, b_labels, b_idx = read_npz_hotlabel('./pelvis_only_224_test_hot_nonhardware_FemNeck.npz')

        for i in range(2):
            print(i, a_labels[np.argmax(a_labels, axis=1) == i].shape)

            # 0(2154, 2)
            # 1(1282, 2)

        for i in range(2):
            print(i, b_labels[np.argmax(b_labels, axis=1) == i].shape)

            # 0(1274, 2)
            # 1(620, 2)

    # NL vs. any fracture (2 classes).
    else:
        binarizer = preprocessing.MultiLabelBinarizer(classes=np.arange(2))
        for i in ['./pelvis_only_224_test_hot_nonhardware.npz', 'pelvis_only_224_train_hot_nonhardware.npz',
                  'pelvis_only_238_test_hot_nonhardware.npz', 'pelvis_only_238_train_hot_nonhardware.npz']:
            image_array, id_array, hot_label_array, orig_idx_array = read_npz_hotlabel(i)
            label_maxed = np.argmax(hot_label_array, axis=1)
            new_label = np.where(label_maxed != 0, 1, label_maxed)
            new_label = binarizer.fit_transform(new_label.reshape(new_label.shape[0], 1))

            new_name = re.search(r'(.*).npz', i).group(1) + '_binary'
            np.savez(new_name, image_array=image_array, id_array=id_array, label_array=new_label,orig_idx_array=orig_idx_array)

        a_imgs, a_ids, a_labels, a_idx = read_npz_hotlabel('./pelvis_only_224_train_hot_nonhardware_binary.npz')
        b_imgs, b_ids, b_labels, b_idx = read_npz_hotlabel('./pelvis_only_224_test_hot_nonhardware_binary.npz')

        for i in range(2):
            print(i, a_labels[np.argmax(a_labels, axis=1) == i].shape)

            # With hardware
            # 0(2185, 2)
            # 1(2764, 2)

            # Without hardware
            # 0(2154, 2)
            # 1(2680, 2)

        for i in range(2):
            print(i, b_labels[np.argmax(b_labels, axis=1) == i].shape)

            # With hardware
            # 0(1305, 2)
            # 1(1266, 2)

            # Without hardware
            # 0(1274, 2)
            # 1(1229, 2)