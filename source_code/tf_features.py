""" <Local Varying Anomaly Detection>
    Copyright (C) <2022>  <Wen-Yan Lin>
    daniellin@smu.edu.sg

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>."""

import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications.resnet import ResNet101, ResNet50
from tensorflow.keras.layers import Input, GlobalAveragePooling2D
from tensorflow.keras.models import Model

def get_indices_from_keras_generator(gen, batch_size):
    """
    Given a keras data generator, it returns the indices and the filepaths
    corresponding the current batch. 
    :param gen: keras generator.
    :param batch_size: size of the last batch generated.
    :return: tuple with indices and filenames
    """

    idx_left = (gen.batch_index - 1) * batch_size
    idx_right = idx_left + gen.batch_size if idx_left >= 0 else None
    indices = gen.index_array[idx_left:idx_right]
    filenames = [gen.filenames[i] for i in indices]
    return indices, filenames



def folders2generator(train_folder, 
            im_size = (224, 224),
            batch_size = 16,
            target_class = None,
            use_one_hot = True,
            ):
        
    if use_one_hot:
        class_mode = 'categorical'
    else:
        class_mode = 'sparse'

    # create tensorflow generators from folders
    sub_folders = list(np.sort(os.listdir(train_folder)))
    if target_class is None:
        class2train = sub_folders
    else:
        class2train = [sub_folders[j] for j in target_class]

    im_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                            fill_mode='bilinear', 
                            validation_split = 0,
                            )
    data_gen = im_datagen.flow_from_directory(train_folder,
                                            classes = class2train,
                                            target_size = im_size,
                                            batch_size = batch_size,
                                            subset = 'training',
                                            class_mode = class_mode) 


    return data_gen



def generator2feature(model, generator):
    # convert a generators images into features and labels
    
    num_sets = len(generator)
    out_im = []
    out_gt = []
    filenames = []
    for i in range(0, num_sets):
        (im, gt) = generator.next()
        indices, filename = get_indices_from_keras_generator(generator, im.shape[0])

        if model is None:
            feat = im
        else:
            feat = np.array(model(im, training=False))
            
        if len(gt.shape) == 2: # one-hot encoding
            gt = np.argmax(gt, axis=1)
        out_im.append(feat)
        out_gt.append(gt)
        filenames.append(filename)
    
    return out_im, out_gt, filenames

def download_traditional_network(base_net= ResNet101(weights='imagenet', include_top=False), 
                input_shape = (224, 224, 3,),
                add_softmax_top = False,
                num_class = None, # only for use with softmax top
                ):
    X_input = Input(input_shape)
    X = base_net(X_input)
    X = GlobalAveragePooling2D()(X)
    if add_softmax_top:
        X = tf.keras.layers.Dense(num_class, activation = 'softmax', name='final')(X)
    return Model(inputs = X_input, outputs = X)
