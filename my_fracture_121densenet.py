#!/usr/bin/env python
import tensorflow as tf
import numpy as np
import sklearn
import os
import argparse

from sklearn import metrics

from load_data import get_npz_structured_iterator

keras = tf.keras
Layers = keras.layers.Layer
Model = keras.models.Model
Metrics = keras.metrics

'-------------To change-------------------'
CLASS_N = 6
PRECROP = './pelvis_only_238_train_hot_nonhardware.npz'
VALIDATION = './pelvis_only_224_test_hot_nonhardware.npz'

SAVE_DIR = './nonhardware_fx_evals/'

N_TRAIN = 4834

'-------------Leave alone-------------------'
EPOCHS = None # useless because indefinite repeat will be on with this model

MODEL_SAVE = True

TRAIN_BATCH_SZ = 20
TEST_BATCH_SZ = 20
BUFF_SIZE = 3000
INP_IMG_SIZE=[238, 238, 3]
OUT_IMG_SIZE=[224, 224, 3]
LABEL_SIZE = [CLASS_N]

POOL_3_CUTOFF = 140
POOL_4_CUTOFF = 312

LAST_ONLY = 100
POOL_4_EPOCHS = 300
POOL_3_EPOCHS = 300
ALL_LAYERS_EPOCHS = 1300
SAVE_PERIOD = 5000

# LAST_ONLY = 1
# POOL_4_EPOCHS = 1
# POOL_3_EPOCHS = 1
# ALL_LAYERS_EPOCHS = 1
# SAVE_PERIOD = 5000

def main(load_path):

    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    fpr = {}
    tpr = {}
    auc_val = {}
    all_synopsis = {}
    accuracy = {}

    graph = tf.get_default_graph()
    sess = tf.Session(graph=graph)
    keras.backend.set_session(sess)

    base_model = keras.applications.densenet.DenseNet121(include_top=False, input_shape=(224, 224, 3), pooling='avg', weights='imagenet')
    x = base_model.output
    #x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Dense(CLASS_N, 'softmax')(x) # with keras, must add kwarg of activation='softmax'
    model = Model(inputs=base_model.input, outputs=x)

    if load_path is not None:
        print('Models weights being loaded from : {}'.format(load_path))
        model.load_weights(load_path)

    # for i, layer in enumerate(model.layers):
    #     print(i, layer.name)
    # layer 52 for pool2_pool, 140 for pool3_pool, 312 for pool4_pool for Densenet121
    # layer 57 for pool2_pool, 150 for pool3_pool, 327 for pool4_pool for pool_only_cat
    # layer 87 for pool2_pool, 240 for pool3_pool, 537 for pool4_pool for conv_pool_cat


    for layer in base_model.layers:
        layer.trainable = False

    my_metrics = [
        Metrics.categorical_accuracy
    ]

    model.compile(optimizer=keras.optimizers.Adam(), loss='categorical_crossentropy', metrics=my_metrics)

    my_callbacks = [
        keras.callbacks.TerminateOnNaN(),
        keras.callbacks.EarlyStopping(monitor='loss', patience=100),
        keras.callbacks.ReduceLROnPlateau(monitor='loss', patience=50, min_lr=1e-6)
    ]

    train_precrop_iter = get_npz_structured_iterator(sess, EPOCHS, TRAIN_BATCH_SZ, PRECROP, BUFF_SIZE, INP_IMG_SIZE,
                                             OUT_IMG_SIZE, LABEL_SIZE, return_initalizer=False)

    train_end_iter, train_end_init, train_end_dataset = get_npz_structured_iterator(sess, 1, TEST_BATCH_SZ, PRECROP, BUFF_SIZE,
                                                                  INP_IMG_SIZE, OUT_IMG_SIZE, LABEL_SIZE,
                                                                  indefinite_repeat=False, aug_crop=True)

    val_iter, val_init, val_dataset = get_npz_structured_iterator(sess, 1, TEST_BATCH_SZ, VALIDATION, BUFF_SIZE, OUT_IMG_SIZE,
                                                                  OUT_IMG_SIZE, LABEL_SIZE, indefinite_repeat=False,
                                                                  aug_crop=False)
    next_train_end = train_end_iter.get_next()
    next_val = val_iter.get_next()

    steps_per_epoch = np.floor(N_TRAIN / TRAIN_BATCH_SZ).astype(np.int32)



    print('---------LAST LAYER TRAINING----------')
    m_history = model.fit(train_precrop_iter, epochs=LAST_ONLY, callbacks=my_callbacks, shuffle=False, steps_per_epoch=steps_per_epoch)

    predict_list = np.empty((0, CLASS_N), np.float32)
    true_label_list = np.empty((0, CLASS_N), np.float32)

    while True:
        try:
            imgs, labels = sess.run(next_train_end)
            c_out = model.predict(imgs)
            predict_list = np.concatenate([predict_list, c_out], axis=0)
            true_label_list = np.concatenate([true_label_list, labels], axis=0)
        except tf.errors.OutOfRangeError:
            break

    ind_prediction = np.argmax(predict_list, axis=1)
    ind_true = np.argmax(true_label_list, axis=1)

    if CLASS_N == 2:
        fpr['last_train'], tpr['last_train'], _ = metrics.roc_curve(y_score=predict_list[:,1], y_true=ind_true)
        auc_val['last_train'] = metrics.auc(fpr['last_train'], tpr['last_train'])
    synopsis = metrics.classification_report(y_true=ind_true, y_pred=ind_prediction)
    all_synopsis['last_train'] = synopsis
    accuracy['last_train'] = metrics.accuracy_score(y_true=ind_true, y_pred=ind_prediction)

    predict_list = np.empty((0, CLASS_N), np.float32)
    true_label_list = np.empty((0, CLASS_N), np.float32)

    while True:
        try:
            imgs, labels = sess.run(next_val)
            c_out = model.predict(imgs)
            predict_list = np.concatenate([predict_list, c_out], axis=0)
            true_label_list = np.concatenate([true_label_list, labels], axis=0)
        except tf.errors.OutOfRangeError:
            break


    ind_prediction = np.argmax(predict_list, axis=1)
    ind_true = np.argmax(true_label_list, axis=1)

    if CLASS_N == 2:
        fpr['last_validation'], tpr['last_validation'], _ = metrics.roc_curve(y_score=predict_list[:,1], y_true=ind_true)
        auc_val['last_validation'] = metrics.auc(fpr['last_validation'], tpr['last_validation'])
    synopsis = metrics.classification_report(y_true=ind_true, y_pred=ind_prediction)
    all_synopsis['last_validation'] = synopsis
    accuracy['last_validation'] = metrics.accuracy_score(y_true=ind_true, y_pred=ind_prediction)



    print('---------POOL_4 LAYER TRAINING----------')
    my_callbacks = [
        keras.callbacks.TerminateOnNaN(),
        keras.callbacks.EarlyStopping(monitor='loss', patience=100),
        keras.callbacks.ReduceLROnPlateau(monitor='loss', patience=50, min_lr=1e-6)
    ]

    for layer in model.layers[:POOL_4_CUTOFF]:
        layer.trainable = False
    for layer in model.layers[POOL_4_CUTOFF:]:
        layer.trainable = True

    model.compile(optimizer=keras.optimizers.Adam(), loss='categorical_crossentropy', metrics=my_metrics)

    m_history = model.fit(train_precrop_iter, epochs=POOL_4_EPOCHS, callbacks=my_callbacks, shuffle=False, steps_per_epoch=steps_per_epoch)

    sess.run(train_end_init, feed_dict=train_end_dataset.feed_dict)
    sess.run(val_init, feed_dict=val_dataset.feed_dict)
    predict_list = np.empty((0, CLASS_N), np.float32)
    true_label_list = np.empty((0, CLASS_N), np.float32)

    while True:
        try:
            imgs, labels = sess.run(next_train_end)
            c_out = model.predict(imgs)
            predict_list = np.concatenate([predict_list, c_out], axis=0)
            true_label_list = np.concatenate([true_label_list, labels], axis=0)
        except tf.errors.OutOfRangeError:
            break

    ind_prediction = np.argmax(predict_list, axis=1)
    ind_true = np.argmax(true_label_list, axis=1)

    if CLASS_N == 2:
        fpr['pool4_train'], tpr['pool4_train'], _ = metrics.roc_curve(y_score=predict_list[:, 1], y_true=ind_true)
        auc_val['pool4_train'] = metrics.auc(fpr['pool4_train'], tpr['pool4_train'])
    synopsis = metrics.classification_report(y_true=ind_true, y_pred=ind_prediction)
    all_synopsis['pool4_train'] = synopsis
    accuracy['pool4_train'] = metrics.accuracy_score(y_true=ind_true, y_pred=ind_prediction)

    predict_list = np.empty((0, CLASS_N), np.float32)
    true_label_list = np.empty((0, CLASS_N), np.float32)

    while True:
        try:
            imgs, labels = sess.run(next_val)
            c_out = model.predict(imgs)
            predict_list = np.concatenate([predict_list, c_out], axis=0)
            true_label_list = np.concatenate([true_label_list, labels], axis=0)
        except tf.errors.OutOfRangeError:
            break

    ind_prediction = np.argmax(predict_list, axis=1)
    ind_true = np.argmax(true_label_list, axis=1)

    if CLASS_N == 2:
        fpr['pool4_validation'], tpr['pool4_validation'], _ = metrics.roc_curve(y_score=predict_list[:, 1], y_true=ind_true)
        auc_val['pool4_validation'] = metrics.auc(fpr['pool4_validation'], tpr['pool4_validation'])
    synopsis = metrics.classification_report(y_true=ind_true, y_pred=ind_prediction)
    all_synopsis['pool4_validation'] = synopsis
    accuracy['pool4_validation'] = metrics.accuracy_score(y_true=ind_true, y_pred=ind_prediction)

    if MODEL_SAVE:
        model.save(SAVE_DIR + 'pool4_dense_model')

    print('---------POOL_3 LAYER TRAINING----------')
    my_callbacks = [
        keras.callbacks.TerminateOnNaN(),
        keras.callbacks.EarlyStopping(monitor='loss', patience=100),
        keras.callbacks.ReduceLROnPlateau(monitor='loss', patience=50, min_lr=1e-6),
        keras.callbacks.ModelCheckpoint(SAVE_DIR + 'pool_3', save_best_only=True, period=SAVE_PERIOD, monitor='loss')
    ]

    for layer in model.layers[:POOL_3_CUTOFF]:
        layer.trainable = False
    for layer in model.layers[POOL_3_CUTOFF:]:
        layer.trainable = True

    model.compile(optimizer=keras.optimizers.Adam(lr=0.0005), loss='categorical_crossentropy', metrics=my_metrics)

    m_history = model.fit(train_precrop_iter, epochs=POOL_3_EPOCHS, callbacks=my_callbacks, shuffle=False,
                          steps_per_epoch=steps_per_epoch)

    sess.run(train_end_init, feed_dict=train_end_dataset.feed_dict)
    sess.run(val_init, feed_dict=val_dataset.feed_dict)
    predict_list = np.empty((0, CLASS_N), np.float32)
    true_label_list = np.empty((0, CLASS_N), np.float32)

    while True:
        try:
            imgs, labels = sess.run(next_train_end)
            c_out = model.predict(imgs)
            predict_list = np.concatenate([predict_list, c_out], axis=0)
            true_label_list = np.concatenate([true_label_list, labels], axis=0)
        except tf.errors.OutOfRangeError:
            break

    ind_prediction = np.argmax(predict_list, axis=1)
    ind_true = np.argmax(true_label_list, axis=1)

    if CLASS_N == 2:
        fpr['pool3_train'], tpr['pool3_train'], _ = metrics.roc_curve(y_score=predict_list[:, 1], y_true=ind_true)
        auc_val['pool3_train'] = metrics.auc(fpr['pool3_train'], tpr['pool3_train'])
    synopsis = metrics.classification_report(y_true=ind_true, y_pred=ind_prediction)
    all_synopsis['pool3_train'] = synopsis
    accuracy['pool3_train'] = metrics.accuracy_score(y_true=ind_true, y_pred=ind_prediction)

    predict_list = np.empty((0, CLASS_N), np.float32)
    true_label_list = np.empty((0, CLASS_N), np.float32)

    while True:
        try:
            imgs, labels = sess.run(next_val)
            c_out = model.predict(imgs)
            predict_list = np.concatenate([predict_list, c_out], axis=0)
            true_label_list = np.concatenate([true_label_list, labels], axis=0)
        except tf.errors.OutOfRangeError:
            break

    ind_prediction = np.argmax(predict_list, axis=1)
    ind_true = np.argmax(true_label_list, axis=1)

    if CLASS_N == 2:
        fpr['pool3_validation'], tpr['pool3_validation'], _ = metrics.roc_curve(y_score=predict_list[:, 1],y_true=ind_true)
        auc_val['pool3_validation'] = metrics.auc(fpr['pool3_validation'], tpr['pool3_validation'])
    synopsis = metrics.classification_report(y_true=ind_true, y_pred=ind_prediction)
    all_synopsis['pool3_validation'] = synopsis
    accuracy['pool3_validation'] = metrics.accuracy_score(y_true=ind_true, y_pred=ind_prediction)

    if MODEL_SAVE:
        model.save(SAVE_DIR + 'pool3_dense_model')

    print('---------ALL LAYER TRAINING----------')
    my_callbacks = [
        keras.callbacks.TerminateOnNaN(),
        keras.callbacks.EarlyStopping(monitor='loss', patience=100),
        keras.callbacks.ReduceLROnPlateau(monitor='loss', patience=50, min_lr=1e-6),
        keras.callbacks.ModelCheckpoint(SAVE_DIR + 'all_layers', save_best_only=True, period=SAVE_PERIOD, monitor='loss')
    ]

    for layer in model.layers:
        layer.trainable = True

    model.compile(optimizer=keras.optimizers.Adam(lr=0.0005), loss='categorical_crossentropy', metrics=my_metrics)

    m_history = model.fit(train_precrop_iter, epochs=ALL_LAYERS_EPOCHS, callbacks=my_callbacks, shuffle=False,steps_per_epoch=steps_per_epoch)

    sess.run(train_end_init, feed_dict=train_end_dataset.feed_dict)
    sess.run(val_init, feed_dict=val_dataset.feed_dict)
    predict_list = np.empty((0, CLASS_N), np.float32)
    true_label_list = np.empty((0, CLASS_N), np.float32)

    while True:
        try:
            imgs, labels = sess.run(next_train_end)
            c_out = model.predict(imgs)
            predict_list = np.concatenate([predict_list, c_out], axis=0)
            true_label_list = np.concatenate([true_label_list, labels], axis=0)
        except tf.errors.OutOfRangeError:
            break

    ind_prediction = np.argmax(predict_list, axis=1)
    ind_true = np.argmax(true_label_list, axis=1)

    if CLASS_N == 2:
        fpr['all_train'], tpr['all_train'], _ = metrics.roc_curve(y_score=predict_list[:, 1], y_true=ind_true)
        auc_val['all_train'] = metrics.auc(fpr['all_train'], tpr['all_train'])
    synopsis = metrics.classification_report(y_true=ind_true, y_pred=ind_prediction)
    all_synopsis['all_train'] = synopsis
    accuracy['all_train'] = metrics.accuracy_score(y_true=ind_true, y_pred=ind_prediction)

    predict_list = np.empty((0, CLASS_N), np.float32)
    true_label_list = np.empty((0, CLASS_N), np.float32)

    while True:
        try:
            imgs, labels = sess.run(next_val)
            c_out = model.predict(imgs)
            predict_list = np.concatenate([predict_list, c_out], axis=0)
            true_label_list = np.concatenate([true_label_list, labels], axis=0)
        except tf.errors.OutOfRangeError:
            break

    ind_prediction = np.argmax(predict_list, axis=1)
    ind_true = np.argmax(true_label_list, axis=1)

    if CLASS_N == 2:
        fpr['all_validation'], tpr['all_validation'], _ = metrics.roc_curve(y_score=predict_list[:, 1],y_true=ind_true)
        auc_val['all_validation'] = metrics.auc(fpr['all_validation'], tpr['all_validation'])
    synopsis = metrics.classification_report(y_true=ind_true, y_pred=ind_prediction)
    all_synopsis['all_validation'] = synopsis
    accuracy['all_validation'] = metrics.accuracy_score(y_true=ind_true, y_pred=ind_prediction)

    print('-----------------DONE TRAINING-------------------')
    for i in ['last_train', 'last_validation', 'pool4_train', 'pool4_validation', 'pool3_train',
              'pool3_validation', 'all_train', 'all_validation']:
        print(i)
        print(all_synopsis[i])
        if CLASS_N == 2:
            print('AUC is :', auc_val[i])
        print('Accuracy is: ', accuracy[i])

    if MODEL_SAVE:
        print('Saving final model')
        model.save(SAVE_DIR + 'final_dense_model')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--load_path',
        type=str,
    )

    FLAGS, _ = parser.parse_known_args()
    main(FLAGS.load_path)