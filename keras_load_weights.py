import keras
import numpy as np

from FX_dicom_preparation import read_npz_hotlabel
from model_inferring import custom_eval_models
from utils import array_print

Layers = keras.layers.Layer
Model = keras.models.Model
Metrics = keras.metrics

from vis import visualization
from vis.utils import utils as vis_utils

def load_weights_and_save(load_path, save_path, CLASS_N):

    base_model = keras.applications.densenet.DenseNet121(include_top=False, input_shape=(224, 224, 3), pooling='avg', weights='imagenet')
    x = base_model.output
    # x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Dense(CLASS_N, activation='softmax')(x)  # with keras, must add kwarg of activation='softmax'
    model = Model(inputs=base_model.input, outputs=x)

    model.load_weights(load_path)

    model.save(save_path)

def keras_visualization():
    val_path = './pelvis_only_224_train_hot_nonhardware_consold.npz'
    model_path = './fx_models/consold_fx/keras_weighted_model'

    image_array, id_array, label_array, orig_idx_array = read_npz_hotlabel(val_path)
    model = keras.models.load_model(model_path)
    predictions = model.predict(image_array)

    # Just checking to make sure the loaded model is working on training_224
    custom_eval_models(predictions, label_array)

    chosen_layers = ['conv5_block16_2_conv', 'conv5_block16_concat', 'bn', 'relu', 'avg_pool', 'dense_1']

    for i in chosen_layers: #essentially enumerating through the layers and getting the idx
        print(vis_utils.find_layer_idx(model, i), i)

    max_label = np.argmax(label_array, axis=1)
    label_choice = 2
    label_bool = (max_label == label_choice)
    choice_images = image_array[label_bool]

    image_holder = np.empty((0, 224, 224, 3))
    hmap_holder = np.empty((0, 224, 224))
    sal_holder = np.empty((0, 224, 224))

    for i in range(10):
        c_n = i
        one_train_img = choice_images[c_n]
        hmap = visualization.visualize_cam(model, layer_idx=428, filter_indices=[label_choice], seed_input=one_train_img, penultimate_layer_idx=426)
        # if penultimate layer not specified, it selects the last Conv layer, which is 423 for DenseNet121 named conv5_block16_2_conv
        sal = visualization.visualize_saliency(model, layer_idx=428, filter_indices=[label_choice],seed_input=one_train_img, backprop_modifier='guided')
        one_train_img = np.expand_dims(one_train_img, axis=0)
        hmap = np.expand_dims(hmap, axis=0)
        sal = np.expand_dims(sal, axis=0)
        image_holder = np.concatenate((image_holder, one_train_img), axis=0)
        hmap_holder = np.concatenate((hmap_holder, hmap), axis=0)
        sal_holder = np.concatenate((sal_holder, sal), axis=0)

    array_print(image_holder, hmap_holder, sal_holder)

    np.savez('cam_out_npz/consold_train_class_1_i_to_10.npz', image_holder=image_holder, hmap_holder=hmap_holder, sal_holder=sal_holder)

def load_cam_out(c_path='./cam_out_npz/nonhardware_train_class_2_i_to_20_penult_426.npz'):
    with np.load(c_path) as f:
        image_holder = f['image_holder']
        hmap_holder = f['hmap_holder']
        sal_holder = f['sal_holder']

    return image_holder, hmap_holder, sal_holder