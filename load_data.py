import tensorflow as tf
import numpy as np

from sklearn import preprocessing
from collections import namedtuple, defaultdict

Data = namedtuple('Data', ['images', 'labels'])


#DO NOT put if statements inside the map functions, as they will result in issues with the iterator
class Dataset_Class:
    def __init__(self, epochs, batch_sz, prefetch_n, file_name, shuffle_buff, inp_img_size, out_img_size, label_size, preprocess, indefinite_repeat):
        self.epochs = epochs
        self.batch_sz = batch_sz
        self.prefetch_n = prefetch_n
        self.file_name = file_name
        self.shuffle_buff = shuffle_buff
        self.inp_img_size = inp_img_size
        self.out_img_size = out_img_size
        self.label_size = label_size
        self.preprocess = preprocess
        self.indefinite_repeat = indefinite_repeat

    def standardize_data(self, datatup):
        images = datatup.images
        labels = datatup.labels

        print('Inception preprocessing for: ', self.file_name)
        images = (images - tf.reduce_min(images)) * (1.0 - 0.0) / (tf.reduce_max(images) - tf.reduce_min(images)) + 0.0
        images = (images - 0.5) * 2 # get values [-1, 1].  Shift and scale. For Resnet V2 and all tf models.

        return Data(images, labels)


class Memory_Dataset(Dataset_Class):
    def __init__(self, epochs, batch_sz, prefetch_n, file_name, shuffle_buff, inp_img_size, out_img_size, label_size,
                 preprocess=True, indefinite_repeat=False):
        super().__init__(epochs, batch_sz, prefetch_n, file_name, shuffle_buff, inp_img_size, out_img_size, label_size,
                         preprocess, indefinite_repeat)
        self.images_PH = tf.placeholder(np.float32, [None] + inp_img_size)
        self.labels_PH = tf.placeholder(np.float32, [None] + label_size)

    # MUST do BEFORE batching
    def augment(self, Data_tup):

        img  = Data_tup.images
        label = Data_tup.labels

        img = tf.image.random_brightness(img, 0.3)
        img = tf.image.random_contrast(img, 0.7, 1.3)
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_crop(img, size=self.out_img_size)

        return Data(img, label)

    def shuf_rep_map_batch_fetch(self, dataset, buffer_size, epochs, batch_sz, map_func):
        if self.indefinite_repeat:
            dataset = dataset.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=buffer_size))

        elif epochs > 1: #if epoch == 1, then no repeat.
            dataset = dataset.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=buffer_size, count=epochs))
            #ok to shuffle after parallel-interleave and before mapping.

        # TF bug: Cannot use new map_func (even if lambda) if tf.dataset not previously declared/created before tf Session started.
        if map_func is not None:
            dataset = dataset.apply(tf.data.experimental.map_and_batch(map_func=map_func, batch_size=batch_sz, num_parallel_calls=2))
        else:
            dataset = dataset.batch(batch_sz)

        if self.preprocess:
            dataset = dataset.map(self.standardize_data)

        dataset = dataset.prefetch(self.prefetch_n)

        return dataset

    def make_memory_dataset(self, aug_crop=True):

        dataset = tf.data.Dataset.from_tensor_slices(Data(self.images_PH, self.labels_PH))
        c_map_func = None
        if aug_crop:
            c_map_func = self.augment

        self.dataset = self.shuf_rep_map_batch_fetch(dataset, self.shuffle_buff, self.epochs, self.batch_sz, c_map_func)

    def make_short_dataset(self):
        self.dataset = tf.data.Dataset.from_tensor_slices(Data(self.images_PH, self.labels_PH))

    def make_memory_feed_dict(self):
        with np.load(self.file_name) as c_file:
            images = c_file['image_array']
            labels = c_file['label_array']

        self.feed_dict = {self.images_PH: images,
                          self.labels_PH: labels}


class Dataset_iterator:
    def __init__(self, dataset):
        self.structured_iterator = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)

    def get_el(self):
        return self.structured_iterator.get_next()

    def return_initialized_iterator(self, dataset):
        return self.structured_iterator.make_initializer(dataset)


def get_npz_structured_iterator(sess, epochs, batch_size, file_name, shuffle_buff, inp_img_size, out_img_size, label_size, indefinite_repeat=True, return_initalizer=True, aug_crop=True):

    mem_dataset = Memory_Dataset(epochs, batch_size, 1, file_name, shuffle_buff, inp_img_size, out_img_size, label_size, indefinite_repeat=indefinite_repeat)
    mem_dataset.make_memory_dataset(aug_crop=aug_crop)
    mem_dataset.make_memory_feed_dict()

    d_iterator = Dataset_iterator(mem_dataset.dataset)
    iterator_init = d_iterator.return_initialized_iterator(mem_dataset.dataset)
    sess.run(iterator_init, feed_dict=mem_dataset.feed_dict)

    if return_initalizer:
        return d_iterator.structured_iterator, iterator_init, mem_dataset
    else:
        return d_iterator.structured_iterator