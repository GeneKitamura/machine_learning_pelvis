import numpy as np
import math
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import statsmodels.api as sm

from sklearn import metrics

from model_inferring import show_metrics
from utils import array_print
from FX_dicom_preparation import read_npz_hotlabel

keras = tf.keras
Layers = keras.layers.Layer
Model = keras.models.Model
Metrics = keras.metrics

# If evaluating each model's performance for femoral fx vs normals:
# this method doesn't work well with the grouped_model since fem and ace are combined.
# ROC curves (along with AUC and thresholds) constructed and interpreted in context of the validation data.
# Using a val data of only NL and fem fractures will create a new ROC, but will allow direct comparison of models on performance of the val_data.
# In that case, load the model, and use "infer_from_NL_and_fem_fx" method rather than get_out_values
def get_out_values(cur_npz_dir='./inferrence_npz/nonhardware_fx_sep_out.npz', class_int=[0, 4], class_str=['Normal', 'Prox femur']):
    with np.load(cur_npz_dir) as f:
        image_array = f['image_array']
        id_array = f['id_array']
        label_array = f['label_array']
        orig_idx_array = f['orig_idx_array']
        predictions = f['predictions']
        binary_preds = f['binary_preds']
        binary_labels = f['binary_labels']

    ind_labels = np.argmax(label_array, axis=1)
    ind_bin_lables = np.argmax(binary_labels, axis=1)
    ind_predictions = np.argmax(predictions, axis=1)
    ind_bin_preds = np.argmax(binary_preds, axis=1)

    n = predictions.shape[0]
    out_lib = {}

    # This method looks at class of interest vs. all others (4 vs [0,1,2,3,5])
    for i, j in zip(class_int, class_str):
        class_indicator = np.where(ind_labels == i, 1, 0)
        prediction_class_indicator = np.where(ind_predictions == i, 1, 0)
        fpr, tpr, thresholds = metrics.roc_curve(y_true=class_indicator, y_score=predictions[..., i])
        n_pos = class_indicator[class_indicator == 1].shape[0]
        n_others = class_indicator[class_indicator == 0].shape[0]

        accuracy = metrics.accuracy_score(y_true=class_indicator, y_pred=prediction_class_indicator)
        c_CI95 = math.sqrt((accuracy * (1 - accuracy)) / n)
        print(j, 'accuracy', accuracy, ' +- ', c_CI95)
        auc_val = metrics.auc(fpr, tpr)
        q0 = auc_val * (1 - auc_val)
        q1 = auc_val / (2 - auc_val) - (auc_val ** 2)
        q2 = ((2 * auc_val ** 2) / (1 + auc_val)) - (auc_val ** 2)
        se = math.sqrt((q0 + (n_pos - 1) * q1 + (n_others - 1) * q2) / (n_pos * n_others))
        auc_95_CI = 1.96 * se
        print(j, 'auc is', auc_val, '+-', auc_95_CI)

        thresh_val = show_metrics(predictions[..., i], class_indicator, thresholds)

        post_threshold = np.array(predictions[..., i] > thresh_val, np.int)

        class_filter = (ind_labels == i)

        i_name = str(i)
        out_lib[i_name + '_out'] = post_threshold[class_filter]
        out_lib[i_name + '_ids'] = id_array[class_filter]
        out_lib[i_name + '_auc'] = auc_val
        out_lib[i_name + '_auc_SE'] = se

    return out_lib

# Can get threshold values from the whole validation data, and input into man_thresh_val as a list for class 0 and fem_fx
# sep:  man_thresh_val=[0.46868384, 0.5437456]
# consold: man_thresh_val=[0.71493226, 0.03784376]
# grouped: man_thresh_val=[0.19280155, 0.8270937]
# fem_vs_nonfem: man_thresh_val=[0.32999614, 0.10996514]
# fem_only: man_thresh_val=[0.9664069, 0.034231238]
def infer_from_NL_and_fem_fx(model, npz_path='./pelvis_only_224_test_hot_nonhardware.npz', class_int = 2, man_thresh_val=None):
    imgs, ids, labels, _ = read_npz_hotlabel(npz_path)

    ind_labels = np.argmax(labels, axis=1)
    labels_filter = (ind_labels == 4) | (ind_labels == 0) # only NL and fem fx
    filtered_ind_labels = ind_labels[labels_filter]
    remapped_ind_labels = np.where(filtered_ind_labels == 4, class_int, 0) #changing sep fem class 4 into model fem label
    filtered_imgs = imgs[labels_filter]
    filtered_ids = ids[labels_filter]

    # model = keras.models.load_model(model_path)
    predictions = model.predict(filtered_imgs)
    ind_predictions = np.argmax(predictions, axis=1)

    tot_accuracy = metrics.accuracy_score(y_true=remapped_ind_labels, y_pred=ind_predictions)
    print('tot accuracy:', tot_accuracy)

    n = predictions.shape[0]
    out_lib = {}

    class_l = [0] + [class_int]

    # This method is ONLY look at ONE class vs ONE other since we filter out the rest.
    for c_n, i in enumerate(class_l):
        class_indicator = np.where(remapped_ind_labels == i, 1, 0)
        prediction_class_indicator = np.where(ind_predictions == i, 1, 0)
        fpr, tpr, thresholds = metrics.roc_curve(y_true=class_indicator, y_score=predictions[..., i])
        n_pos = class_indicator[class_indicator == 1].shape[0]
        n_others = class_indicator[class_indicator == 0].shape[0]

        accuracy = metrics.accuracy_score(y_true=class_indicator, y_pred=prediction_class_indicator)
        c_CI95 = math.sqrt((accuracy * (1 - accuracy)) / n)
        print('class', i, 'accuracy', accuracy, ' +- ', c_CI95)

        auc_val = metrics.auc(fpr, tpr)
        q0 = auc_val * (1 - auc_val)
        q1 = auc_val / (2 - auc_val) - auc_val ** 2
        q2 = 2 * auc_val ** 2 / (1 + auc_val) - auc_val ** 2
        se = math.sqrt((q0 + (n_pos - 1) * q1 + (n_others - 1) * q2) / (n_pos * n_others))
        auc_95_CI = 1.96 * se
        print('class', i, 'auc is', auc_val, '+-', auc_95_CI)

        thresh_val = show_metrics(predictions[..., i], class_indicator, thresholds, single_man_thresh_val=man_thresh_val[c_n])

        post_threshold = np.array(predictions[..., i] > thresh_val, np.int)
        tn, fp, fn, tp = metrics.confusion_matrix(class_indicator, post_threshold).ravel()

        class_filter = (remapped_ind_labels == i)

        i_name = str(i)
        out_lib[i_name + '_out'] = post_threshold[class_filter]
        out_lib[i_name + '_ids'] = filtered_ids[class_filter]
        out_lib[i_name + '_auc'] = auc_val
        out_lib[i_name + '_auc_SE'] = se
        out_lib[i_name + '_whole_out'] = post_threshold
        out_lib[i_name + '_whole_labels'] = class_indicator
        out_lib[i_name + '_confusion'] = (tn, fp, fn, tp)

    return out_lib


def sep_vs_fem(sep_lib=None, fem_only_lib=None, sep_class_int = 4, fem_class_int = 1, neg_corr=None, pos_corr=None):
    if sep_lib is None:
        print('Sep_lib')
        sep_lib = get_out_values(cur_npz_dir ='./inferrence_npz/nonhardware_fx_sep_out.npz', class_int = [0] + [sep_class_int], class_str = ['Normal', 'Prox femur'])

    if fem_only_lib is None:
        print('fem_only_lib')
        fem_only_lib = get_out_values(cur_npz_dir ='./inferrence_npz/FemNeck_out.npz', class_int = [0] + [fem_class_int], class_str = ['Normal', 'Femoral'])

    sep_id_name = str(sep_class_int) + '_ids'
    sep_out_name = str(sep_class_int) + '_out'
    fem_id_name = str(fem_class_int) + '_ids'
    fem_out_name = str(fem_class_int) + '_out'
    sep_auc_name = str(sep_class_int) + '_auc'
    fem_auc_name = str(fem_class_int) + '_auc'
    sep_auc_se = str(sep_class_int) + '_auc_SE'
    fem_auc_se = str(fem_class_int) + '_auc_SE'
    sep_whole_labels = str(sep_class_int) + '_whole_labels'
    fem_whole_labels = str(fem_class_int) + '_whole_labels'
    sep_whole_out = str(sep_class_int) + '_whole_out'
    fem_whole_out = str(fem_class_int) + '_whole_out'
    sep_confusion = str(sep_class_int) + '_confusion'
    fem_confusion = str(fem_class_int) + '_confusion'

    sep_0_labels = sep_lib['0_whole_labels']
    sep_pos_labels = sep_lib[sep_whole_labels]
    fem_0_labels = fem_only_lib['0_whole_labels']
    fem_pos_labels= fem_only_lib[fem_whole_labels]

    sep_0_outs = sep_lib['0_whole_out']
    sep_pos_outs = sep_lib[sep_whole_out]
    fem_0_outs = fem_only_lib['0_whole_out']
    fem_pos_outs = fem_only_lib[fem_whole_out]

    sep_0_confusion = sep_lib['0_confusion']
    sep_pos_confusion = sep_lib[sep_confusion]
    fem_0_confusion = fem_only_lib['0_confusion']
    fem_pos_confusion = fem_only_lib[fem_confusion]

    #PPV and NPV comparisons for class 0
    tn1, fp1, fn1, tp1 = sep_0_confusion
    tn2, fp2, fn2, tp2 = fem_0_confusion
    tot_pos_1 = tp1 + fp1
    tot_pos_2 = tp2 + fp2
    ppv1 = tp1 / tot_pos_1
    ppv2 = tp2 / tot_pos_2
    print('class 0 ppv1 {} and ppv2 {}'.format(ppv1, ppv2))
    pooled_ppv = (tp1 + tp2) / (tot_pos_1 + tot_pos_2)
    both_tp = np.sum(((sep_0_outs == 1) & (fem_0_outs == 1)) & (sep_0_labels == 1))
    both_fp = np.sum(((sep_0_outs == 1) & (fem_0_outs == 1)) & (sep_0_labels == 0))
    pos_C = (both_tp*(1 - pooled_ppv)**2 + (both_fp*pooled_ppv**2)) / (tot_pos_1 + tot_pos_2)
    ppv_chi_square_test_value = (ppv1 - ppv2)**2 / ((pooled_ppv * (1 - pooled_ppv) - 2 * pos_C) * (1/tot_pos_1 + 1/tot_pos_2))

    tot_neg_1 = tn1 + fn1
    tot_neg_2 = tn2 + fn2
    npv1 = tn1 / tot_neg_1
    npv2 = tn2 / tot_neg_2
    print('class 0 npv1 {} and npv2 {}'.format(npv1, npv2))
    pooled_npv = (tn1 + tn2) / (tot_neg_1 + tot_neg_2)
    both_tn = np.sum(((sep_0_outs == 0) & (fem_0_outs == 0)) & (sep_0_labels == 0))
    both_fn = np.sum(((sep_0_outs == 0) & (fem_0_outs == 0)) & (sep_0_labels == 1))
    neg_C = (both_tn*(1 - pooled_npv)**2 + (both_fn*pooled_npv**2))/ (tot_neg_1 + tot_neg_2)
    npv_chi_square_test_value = (npv1 - npv2)**2 / ((pooled_npv * (1 - pooled_npv) - 2 * neg_C) * (1/tot_neg_1 + 1/tot_neg_2))

    print('Class 0 ppv chisquare: {}.  npv chisquare: {}'.format(ppv_chi_square_test_value, npv_chi_square_test_value))

    # PPV and NPV comparisons for positive class
    tn1, fp1, fn1, tp1 = sep_pos_confusion
    tn2, fp2, fn2, tp2 = fem_pos_confusion
    tot_pos_1 = tp1 + fp1
    tot_pos_2 = tp2 + fp2
    ppv1 = tp1 / tot_pos_1
    ppv2 = tp2 / tot_pos_2
    print('class 1 ppv1 {} and ppv2 {}'.format(ppv1, ppv2))
    pooled_ppv = (tp1 + tp2) / (tot_pos_1 + tot_pos_2)
    both_tp = np.sum(((sep_pos_outs == 1) & (fem_pos_outs == 1)) & (sep_pos_labels == 1))
    both_fp = np.sum(((sep_pos_outs == 1) & (fem_pos_outs == 1)) & (sep_pos_labels == 0))
    pos_C = (both_tp * (1 - pooled_ppv) ** 2 + (both_fp * pooled_ppv ** 2)) / (tot_pos_1 + tot_pos_2)
    ppv_chi_square_test_value = (ppv1 - ppv2) ** 2 / ((pooled_ppv * (1 - pooled_ppv) - 2 * pos_C) * (1 / tot_pos_1 + 1 / tot_pos_2))

    tot_neg_1 = tn1 + fn1
    tot_neg_2 = tn2 + fn2
    npv1 = tn1 / tot_neg_1
    npv2 = tn2 / tot_neg_2
    print('class 1 npv1 {} and npv2 {}'.format(npv1, npv2))
    pooled_npv = (tn1 + tn2) / (tot_neg_1 + tot_neg_2)
    both_tn = np.sum(((sep_pos_outs == 0) & (fem_pos_outs == 0)) & (sep_pos_labels == 0))
    both_fn = np.sum(((sep_pos_outs == 0) & (fem_pos_outs == 0)) & (sep_pos_labels == 1))
    neg_C = (both_tn * (1 - pooled_npv) ** 2 + (both_fn * pooled_npv ** 2)) / (tot_neg_1 + tot_neg_2)
    npv_chi_square_test_value = (npv1 - npv2) ** 2 / ((pooled_npv * (1 - pooled_npv) - 2 * neg_C) * (1 / tot_neg_1 + 1 / tot_neg_2))

    print('Class Pos ppv chisquare: {}.  npv chisquare: {}'.format(ppv_chi_square_test_value, npv_chi_square_test_value))

    print('\nMake sure labels between the groups are identical')
    pos_labels_overlap = fem_pos_labels[fem_pos_labels != sep_pos_labels]
    neg_labels_overlap = fem_0_labels[fem_0_labels != sep_0_labels]
    array_print(pos_labels_overlap, neg_labels_overlap)

    # For class 0 cases
    both_tn = np.sum(((sep_0_outs == 0) & (fem_0_outs == 0)) & (sep_0_labels == 0))
    both_tp = np.sum(((sep_0_outs == 1) & (fem_0_outs == 1)) & (sep_0_labels == 1))
    both_fn = np.sum(((sep_0_outs == 0) & (fem_0_outs == 0)) & (sep_0_labels == 1))
    both_fp = np.sum(((sep_0_outs == 1) & (fem_0_outs == 1)) & (sep_0_labels == 0))
    sep_tn_fem_fp = np.sum(((sep_0_outs == 0) & (fem_0_outs == 1)) & (sep_0_labels == 0))
    sep_fp_fem_tn = np.sum(((sep_0_outs == 1) & (fem_0_outs == 0)) & (sep_0_labels == 0))
    sep_tp_fem_fn = np.sum(((sep_0_outs == 1) & (fem_0_outs == 0)) & (sep_0_labels == 1))
    sep_fn_fem_tp = np.sum(((sep_0_outs == 0) & (fem_0_outs == 1)) & (sep_0_labels == 1))

    # sens = tp / (tp + fn + eps)
    # spec = tn / (tn + fp + eps)
    # ppv = tp / (tp + fp + eps)
    # npv = tn / (tn + fn + eps)

    sens_cont_table = [[both_tp, sep_fn_fem_tp], [sep_tp_fem_fn, both_fn]]
    spec_cont_table = [[both_tn, sep_fp_fem_tn], [sep_tn_fem_fp, both_fp]]
    ppv_cont_table = [[both_tp, ], []]
    npv_cont_table = [[], []]

    print('For 0 class: ')
    print(both_tn, both_tp, both_fn, both_fp, sep_tn_fem_fp, sep_fp_fem_tn, sep_tp_fem_fn, sep_fn_fem_tp)
    print('sens: ', sm.stats.mcnemar(sens_cont_table))
    print('spec: ', sm.stats.mcnemar(spec_cont_table))

    # For pos class
    both_tn = np.sum(((sep_pos_outs == 0) & (fem_pos_outs == 0)) & (sep_pos_labels == 0))
    both_tp = np.sum(((sep_pos_outs == 1) & (fem_pos_outs == 1)) & (sep_pos_labels == 1))
    both_fn = np.sum(((sep_pos_outs == 0) & (fem_pos_outs == 0)) & (sep_pos_labels == 1))
    both_fp = np.sum(((sep_pos_outs == 1) & (fem_pos_outs == 1)) & (sep_pos_labels == 0))
    sep_tn_fem_fp = np.sum(((sep_pos_outs == 0) & (fem_pos_outs == 1)) & (sep_pos_labels == 0))
    sep_fp_fem_tn = np.sum(((sep_pos_outs == 1) & (fem_pos_outs == 0)) & (sep_pos_labels == 0))
    sep_tp_fem_fn = np.sum(((sep_pos_outs == 1) & (fem_pos_outs == 0)) & (sep_pos_labels == 1))
    sep_fn_fem_tp = np.sum(((sep_pos_outs == 0) & (fem_pos_outs == 1)) & (sep_pos_labels == 1))

    sens_cont_table = [[both_tp, sep_fn_fem_tp], [sep_tp_fem_fn, both_fn]]
    spec_cont_table = [[both_tn, sep_fp_fem_tn], [sep_tn_fem_fp, both_fp]]
    ppv_cont_table = [[both_tp, ], []]
    npv_cont_table = [[], []]

    print('\nFor Pos class: ')
    print(both_tn, both_tp, both_fn, both_fp, sep_tn_fem_fp, sep_fp_fem_tn, sep_tp_fem_fn, sep_fn_fem_tp)
    print('sens: ', sm.stats.mcnemar(sens_cont_table))
    print('spec: ', sm.stats.mcnemar(spec_cont_table))

    print('\nMake sure there is no ID overlap')
    neg_id_overlap = fem_only_lib['0_ids'][(fem_only_lib['0_ids'] != sep_lib['0_ids'])]
    pos_id_overlap = fem_only_lib[fem_id_name][(fem_only_lib[fem_id_name] != sep_lib[sep_id_name])]
    print('Difference in positive and negative outputs between the 2 models')
    neg_out_overlap = fem_only_lib['0_out'][(fem_only_lib['0_out'] != sep_lib['0_out'])]
    pos_out_overlap = fem_only_lib[fem_out_name][(fem_only_lib[fem_out_name] != sep_lib[sep_out_name])]

    array_print(neg_id_overlap, pos_id_overlap, neg_out_overlap, pos_out_overlap)

    pos_R = metrics.matthews_corrcoef(sep_lib[sep_out_name], fem_only_lib[fem_out_name])
    neg_R = metrics.matthews_corrcoef(sep_lib['0_out'], fem_only_lib['0_out'])

    avg_R = (pos_R + neg_R) / 2
    avg_0_AUC = (sep_lib['0_auc'] + fem_only_lib['0_auc']) / 2
    avg_pos_AUC = (sep_lib[sep_auc_name] + fem_only_lib[fem_auc_name]) / 2

    print('avg_R:', avg_R, 'avg_0_auc:', avg_0_AUC, 'avg_pos_auc:', avg_pos_AUC)

    # From Table in manuscript.  Use avg_0_AUC for neg_corr and avg_pos_AUC for pos_corr.
    if neg_corr is None:
        neg_corr = 0.37
    if pos_corr is None:
        pos_corr = 0.33

    both_auc_SE = math.sqrt(sep_lib['0_auc_SE'] ** 2 + fem_only_lib['0_auc_SE'] ** 2 - 2 * neg_corr * sep_lib['0_auc_SE'] * fem_only_lib['0_auc_SE'])
    neg_z = (sep_lib['0_auc'] - fem_only_lib['0_auc']) / both_auc_SE
    non_cor_neg_z = (sep_lib['0_auc'] - fem_only_lib['0_auc']) / (math.sqrt(sep_lib['0_auc_SE'] ** 2 + fem_only_lib['0_auc_SE'] ** 2))

    pos_both_auc_SE = math.sqrt(sep_lib[sep_auc_se] ** 2 + fem_only_lib[fem_auc_se] ** 2 - 2 * pos_corr * sep_lib[sep_auc_se] * fem_only_lib[fem_auc_se])
    pos_z = (sep_lib[sep_auc_name] - fem_only_lib[fem_auc_name]) / pos_both_auc_SE
    non_cor_pos_z = (sep_lib[sep_auc_name] - fem_only_lib[fem_auc_name]) / math.sqrt(sep_lib[sep_auc_se] ** 2 + fem_only_lib[fem_auc_se] ** 2)

    print('non_cor_neg_z: {} non_cor_pos_z: {}'.format(non_cor_neg_z, non_cor_pos_z))
    print('neg_z:', neg_z, 'pos_z:', pos_z)

def calculate_z_values_between_models():
    group_model = keras.models.load_model('./fx_models/grouped_fx/final_dense_model')
    sep_model = keras.models.load_model('./fx_models/nonhardware_fx_sep/final_dense_model')
    femNeck_model = keras.models.load_model('./fx_models/fem_Neck_only/final_dense_model')
    consold_model = keras.models.load_model('./fx_models/consold_fx/final_dense_model')
    fem_vs_nonfem_model = keras.models.load_model('./fx_models/fem_vs_nonfem/final_dense_model')

    sep_lib = infer_from_NL_and_fem_fx(sep_model, class_int=4)
    fem_vs_nonfem_lib = infer_from_NL_and_fem_fx(fem_vs_nonfem_model, class_int=1)
    consold_lib = infer_from_NL_and_fem_fx(consold_model, class_int=2)
    grouped_lib = infer_from_NL_and_fem_fx(group_model, class_int=2)
    fem_only_lib = infer_from_NL_and_fem_fx(femNeck_model, class_int=1)

    #Change sep_lib to different model libs, while keeping the fem_only_lib the same to compare model performance

    sep_vs_fem(sep_lib=fem_vs_nonfem_lib, fem_only_lib=fem_only_lib, pos_corr=0.31, neg_corr=0.34, sep_class_int=1)
    # sep_vs_fem(sep_lib=consold_lib, fem_only_lib=fem_only_lib, pos_corr=0.33, neg_corr=0.35, sep_class_int=2)
    # sep_vs_fem(sep_lib=grouped_lib, fem_only_lib=fem_only_lib, pos_corr=0.35, neg_corr=0.35, sep_class_int=2)