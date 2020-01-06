import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from FX_dicom_preparation import read_npz_hotlabel

from sklearn import preprocessing, metrics
from collections import namedtuple

keras = tf.keras
Layers = keras.layers.Layer
Model = keras.models.Model
Metrics = keras.metrics

def custom_eval_models(val_predicted, val_labels, save_auc=None, top_corner=False):
    ind_predictions = np.argmax(val_predicted, axis=1)
    ind_labels = np.argmax(val_labels, axis=1)

    synopsis = metrics.classification_report(y_true=ind_labels, y_pred=ind_predictions)
    accuracy = metrics.accuracy_score(y_true=ind_labels, y_pred=ind_predictions)

    print(synopsis)
    print("Accuracy is {}".format(accuracy))

    fig, ax = plt.subplots()
    line_styles = [':', '-.', '--', '-', ':', '-.']
    colors = ['red', 'blue', 'yellow', 'green', 'cyan', 'magenta']
    color = 'black'

    n_classes = pd.Series(ind_labels).nunique()

    for i in range(n_classes):
        class_indicator = np.where(ind_labels == i, 1, 0)
        fpr, tpr, _ = metrics.roc_curve(y_true=class_indicator, y_score=val_predicted[..., i])
        auc_val = metrics.auc(fpr, tpr)
        # class_list = ['Pelvis', 'Hip', '_', 'CXR']
        class_list = ['0', '1', '2', '3', '4', '5', '6']
        print('AUC is {} for {} class'.format(auc_val, class_list[i]))
        ax.plot(fpr, tpr, label=(class_list[i] + ' class' + ' with AUC of %.2f' % auc_val), markevery=20, linestyle=line_styles[i], color=colors[i])

    ax.plot([0, 1], [0, 1], linestyle='-', color='black')
    ax.set_xlabel("False Positive rate", color=color)
    ax.set_ylabel("True Positive rate", color=color)
    ax.set_xlim(left=-0.01, right=1.01)
    ax.set_ylim(bottom=0, top=1.01)
    if top_corner:
        ax.set_xlim(left=-0.01, right=0.1)
        ax.set_ylim(bottom=0.9, top=1.01)
    ax.tick_params(color=color, labelcolor=color)
    ax.set_title('Receiver Operator Curve', color=color)
    ax.legend(loc=4)
    if save_auc is not None:
        fig.savefig(save_auc, dpi=400, format='eps')
    plt.show()

def model_inferring(val_path='./pelvis_only_224_test_hot_nonhardware.npz', model_path='./fx_models/nonhardware_fx_sep/final_dense_model'):
    image_array, id_array, label_array, orig_idx_array = read_npz_hotlabel(val_path)
    model = keras.models.load_model(model_path)
    predictions = model.predict(image_array)

    custom_eval_models(predictions, label_array)

    class_0_preds = predictions[:, 0].reshape(predictions.shape[0], 1)
    non_fx_preds = np.sum(predictions[:, 1:], axis=1)
    non_fx_preds = non_fx_preds.reshape(non_fx_preds.shape[0], 1)
    binary_preds = np.concatenate([class_0_preds, non_fx_preds], axis=1)

    ind_labels = np.argmax(label_array, axis=1)
    binary_labels = np.where(ind_labels != 0, 1, ind_labels)
    binary_labels = preprocessing.MultiLabelBinarizer(np.arange(2)).fit_transform(binary_labels.reshape(binary_labels.shape[0], 1))

    custom_eval_models(binary_preds, binary_labels)

    np.savez('something',
             image_array=image_array,
             id_array=id_array,
             label_array=label_array,
             orig_idx_array=orig_idx_array,
             predictions=predictions,
             binary_preds=binary_preds,
             binary_labels=binary_labels
             )

def infer_from_out_npz(npz_file):
    with np.load(npz_file) as f:
        image_array = f['image_array']
        id_array = f['id_array']
        label_array = f['label_array']
        orig_idx_array = f['orig_idx_array']
        predictions = f['predictions']
        binary_preds = f['binary_preds']
        binary_labels = f['binary_labels']

    custom_eval_models(predictions, label_array)

    custom_eval_models(binary_preds, binary_labels)

def custom_consold_eval_models(class_str_LIST_lists, class_int_LIST_list, npz_list, title_list, save_auc=None, width=36, height=12):

    fig, axes = plt.subplots(2, 5, figsize=(width, height))
    fig.subplots_adjust(hspace=.1, wspace=0)


    line_styles = [':', '-.', '--', '-', ':', '-.']
    colors = ['red', 'blue', 'yellow', 'green', 'cyan', 'magenta']
    color = 'black'
    half_tot_num = np.arange(4)

    flat_axes = axes.flatten()
    binary_str = ['Normal', 'Abnormal']

    for c_num in half_tot_num:
        cur_npz_dir = npz_list[c_num]
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

        print(title_list[c_num])
        accuracy = metrics.accuracy_score(y_true=ind_labels, y_pred=ind_predictions)
        bin_accuracy = metrics.accuracy_score(y_true=ind_bin_lables, y_pred=ind_bin_preds)
        print("Accuracy is {}".format(accuracy))
        print('Binary accuracy is ', bin_accuracy)

        c_upper_ax = flat_axes[c_num]
        c_lower_ax = flat_axes[c_num + 5]
        class_str = class_str_LIST_lists[c_num]
        class_int = class_int_LIST_list[c_num]

        for i in class_int:
            class_indicator = np.where(ind_labels == i, 1, 0)
            prediction_class_indicator = np.where(ind_predictions == i, 1, 0)
            fpr, tpr, _ = metrics.roc_curve(y_true=class_indicator, y_score=predictions[..., i])

            accuracy = metrics.accuracy_score(y_true=class_indicator, y_pred=prediction_class_indicator)
            print(class_str[i], 'accuracy', accuracy)
            auc_val = metrics.auc(fpr, tpr)
            c_upper_ax.plot(fpr, tpr, label=(class_str[i] + ': AUC of %.2f' % auc_val), markevery=20, linestyle=line_styles[i], color=colors[i])

        print('\n')

        c_upper_ax.plot([0, 1], [0, 1], linestyle='-', color='black')
        # c_upper_ax.set_xlabel("False Positive rate", color=color)
        # c_upper_ax.set_ylabel("True Positive rate", color=color)
        c_upper_ax.set_xlim(left=-0.01, right=1.01)
        c_upper_ax.set_ylim(bottom=0, top=1.01)
        c_upper_ax.tick_params(color=color, labelcolor=color)
        c_upper_ax.set_xticks([], [])
        if c_num != 0:
            c_upper_ax.set_yticks([], [])
        c_upper_ax.set_title(title_list[c_num], color=color)
        c_upper_ax.legend(loc=4)

        # print('Binary version:')
        for i in [0, 1]: # for the binary versions
            bin_class_indicator = np.where(ind_bin_lables == i, 1, 0)
            bin_pred_indicator = np.where(ind_bin_preds == i, 1, 0)
            # accuracy = metrics.accuracy_score(y_true=bin_class_indicator, y_pred=bin_pred_indicator)
            # print('class', i, 'accuracy is', accuracy)

            fpr, tpr, _ = metrics.roc_curve(y_true=bin_class_indicator, y_score=binary_preds[..., i])
            auc_val = metrics.auc(fpr, tpr)
            c_lower_ax.plot(fpr, tpr, label=(binary_str[i] + ': AUC of %.2f' % auc_val), markevery=20, linestyle=line_styles[i], color=colors[i])

        c_lower_ax.plot([0, 1], [0, 1], linestyle='-', color='black')
        # c_lower_ax.set_xlabel("False Positive rate", color=color)
        # c_lower_ax.set_ylabel("True Positive rate", color=color)
        c_lower_ax.set_xlim(left=-0.01, right=1.01)
        c_lower_ax.set_ylim(bottom=0, top=1.01)
        c_lower_ax.tick_params(color=color, labelcolor=color)
        if c_num != 0:
            c_lower_ax.set_xticks([], [])
            c_lower_ax.set_yticks([], [])
        c_lower_ax.set_title(title_list[c_num] + ' as binary output', color=color)
        c_lower_ax.legend(loc=4)

    # One plot of Fem vs. Non-fem
    with np.load('./inferrence_npz/FemNeck_out.npz') as f:
        image_array = f['image_array']
        id_array = f['id_array']
        label_array = f['label_array']
        orig_idx_array = f['orig_idx_array']
        predictions = f['predictions']
        binary_preds = f['binary_preds']
        binary_labels = f['binary_labels']

    ind_labels = np.argmax(label_array, axis=1)
    ind_bin_preds = np.argmax(predictions, axis=1)

    accuracy = metrics.accuracy_score(y_true=ind_labels, y_pred=ind_bin_preds)
    print('fem_vs_nonfem accuracy', accuracy)

    fem_fx_only_str = ['Normal', 'Femoral fracture']
    bin_ax = flat_axes[4]
    for i in [0, 1]:
        class_indicator = np.where(ind_labels == i, 1, 0)
        fpr, tpr, _ = metrics.roc_curve(y_true=class_indicator, y_score=predictions[..., i])
        auc_val = metrics.auc(fpr, tpr)
        bin_ax.plot(fpr, tpr, label=(fem_fx_only_str[i] + ': AUC of %.2f' % auc_val), markevery=20, linestyle=line_styles[i], color=colors[i])

    bin_ax.plot([0, 1], [0, 1], linestyle='-', color='black')
    # bin_ax.set_xlabel("False Positive rate", color=color)
    # bin_ax.set_ylabel("True Positive rate", color=color)
    bin_ax.set_xlim(left=-0.01, right=1.01)
    bin_ax.set_ylim(bottom=0, top=1.01)
    bin_ax.tick_params(color=color, labelcolor=color)
    bin_ax.set_xticks([], [])
    bin_ax.set_yticks([], [])
    bin_ax.set_title("Normal vs. Femoral fracture only", color=color)
    bin_ax.legend(loc=4)

    # One plot of binary
    with np.load('./inferrence_npz/binary_fx_out.npz') as f:
        image_array = f['image_array']
        id_array = f['id_array']
        label_array = f['label_array']
        orig_idx_array = f['orig_idx_array']
        predictions = f['predictions']
        binary_preds = f['binary_preds']
        binary_labels = f['binary_labels']

    ind_labels = np.argmax(label_array, axis=1)
    ind_bin_lables = np.argmax(binary_labels, axis=1)
    ind_bin_preds = np.argmax(predictions, axis=1)

    accuracy = metrics.accuracy_score(y_true=ind_labels, y_pred=ind_bin_preds)
    print('Binary trained accuracy', accuracy)

    bin_ax = flat_axes[9]
    for i in [0, 1]:
        class_indicator = np.where(ind_labels == i, 1, 0)
        fpr, tpr, _ = metrics.roc_curve(y_true=class_indicator, y_score=predictions[..., i])
        auc_val = metrics.auc(fpr, tpr)
        bin_ax.plot(fpr, tpr, label=(binary_str[i] + ': AUC of %.2f' % auc_val), markevery=20, linestyle=line_styles[i], color=colors[i])

    bin_ax.plot([0, 1], [0, 1], linestyle='-', color='black')
    # bin_ax.set_xlabel("False Positive rate", color=color)
    # bin_ax.set_ylabel("True Positive rate", color=color)
    bin_ax.set_xlim(left=-0.01, right=1.01)
    bin_ax.set_ylim(bottom=0, top=1.01)
    bin_ax.tick_params(color=color, labelcolor=color)
    bin_ax.set_xticks([], [])
    bin_ax.set_yticks([], [])
    bin_ax.set_title("Model trained as binary split", color=color)
    bin_ax.legend(loc=4)

    if save_auc is not None:
        fig.savefig(save_auc, dpi=400, format='eps')
    plt.show()