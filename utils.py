import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn import metrics

from df_data_process import read_hot_npz

keras = tf.keras
Layers = keras.layers.Layer
Model = keras.models.Model
Metrics = keras.metrics

def load_model_data(keras_model_path, val_data_path):
    k_model = keras.models.load_model(keras_model_path)
    val_data_tup = read_hot_npz(val_data_path)

    return k_model, val_data_tup

def eval_models(keras_model, val_data_tup, save_auc=None):
    k_model = keras_model
    val_data = val_data_tup
    val_predicted = k_model.predict(val_data.images)

    ind_true = np.argmax(val_data.labels, axis=1)
    ind_prediction = np.argmax(val_predicted, axis=1)
    n_classes = len(set(ind_true))

    synopsis = metrics.classification_report(y_true=ind_true, y_pred=ind_prediction)
    accuracy = metrics.accuracy_score(y_true=ind_true, y_pred=ind_prediction)

    print(synopsis)
    print("Accuracy is {}".format(accuracy))

    fig, ax = plt.subplots()
    line_styles = [':', '-.', '--', '-']
    colors = ['red', 'blue', 'yellow', 'green']
    color = 'black'

    for i in range(n_classes):
        class_indicator = np.where(np.argmax(val_data.labels, axis=1) == i, 1, 0)
        fpr, tpr, _ = metrics.roc_curve(y_true=class_indicator, y_score=val_predicted[..., i])
        auc_val = metrics.auc(fpr, tpr)
        print('AUC is {} for class {}'.format(auc_val, i))
        ax.plot(fpr, tpr, label=('Class ' + str(i) + ' with AUC of %.2f' % auc_val), markevery=20, linestyle=line_styles[i], color=colors[i])

    ax.plot([0, 1], [0, 1], linestyle='-', color='black')
    ax.set_xlabel("False Positive rate", color=color)
    ax.set_ylabel("True Positive rate", color=color)
    ax.set_xlim(left=-0.01, right=1.01)
    ax.set_ylim(bottom=0, top=1.01)
    ax.tick_params(color=color, labelcolor=color)
    ax.set_title('Receiver Operator Curve', color=color)
    ax.legend(loc=4)
    if save_auc is not None:
        fig.savefig(save_auc, dpi=400, format='png')
    plt.show()


def array_print(*args):
    for i in args:
        print(i.shape)

def array_min_max(*args):
    for i in args:
        print(np.min(i), np.max(i))