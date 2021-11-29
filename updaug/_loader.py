# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import tensorflow as tf


def _random_rotate(x):
    theta = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
    return tf.image.rot90(x, theta)


def distort(x, outputshape=(128,128)):
    """
    Wrapper for tf.raw_ops.ImageProjectiveTransform; scales, shears,
    and shifts an image.
    
    :x: (H,W,3) tensor for the input image
    :outputshape: tuple for image output size
    """
    scale_x = tf.random.uniform((), minval=0.8, maxval=1.2)
    scale_y = tf.random.uniform((), minval=0.8, maxval=1.2)
    shear_x = tf.random.uniform((), minval=-0.1, maxval=0.1)
    shear_y = tf.random.uniform((), minval=-0.1, maxval=0.1)
    dxmax = x.shape[1]*scale_x/10
    dymax = x.shape[0]*scale_y/10
    dx = tf.random.uniform((), minval=-dxmax, maxval=dxmax)
    dy = tf.random.uniform((), minval=-dymax, maxval=dymax)

    tfm = tf.stack([scale_x, shear_x, dx, shear_y, scale_y, dy, 
                     tf.constant(0.0, dtype=tf.float32), 
                     tf.constant(0.0, dtype=tf.float32)], axis=0)
    tfm = tf.reshape(tfm, [1,8])
    distorted = tf.raw_ops.ImageProjectiveTransformV2(images=tf.expand_dims(x,0), transforms=tfm, 
                                      output_shape=outputshape, interpolation="BILINEAR")[0]
    return tf.reshape(distorted, (outputshape[0], outputshape[1], 3))


def dataset_generator(filepaths, labels, pairs_per_epoch, num_parallel_calls=6,
                      outputshape=(128,128), filetype="png", batch_size=64,
                      crop=False, flip=True, rot=False, seed=False):
    """
    Build a generator that yields tensorflow datasets, each generating pairs of 
    distorted images with corresponding domains.
    
    :filepaths: list of strings; paths to all images
    :labels: list; domain labels for each file. assumes they're numbered 0 
        to num_domains-1
    :pairs_per_epoch:
    :num_parallel_calls: number of cores for loading/augmenting images
    :ouputshape: tuple; size of output images
    :filetype: whether to look for jpg or png images
    :batch_size:
    :crop: if True, pull a random crop from each image instead of resizing
    :flip:
    :rot:
    """
    # organize files by domain- first 
    df = pd.DataFrame({"filepath":filepaths, "label":labels})
    num_domains = df.label.max()+1
    
    subsets = [df[df.label == i].filepath.values for i in range(num_domains)]
    
    def _load_and_distort(x):
        loaded = tf.io.read_file(x)
        if filetype == "png":
            decoded = tf.io.decode_png(loaded)
        elif filetype == "jpg":
            decoded = tf.io.decode_jpeg(loaded)
        else:
            assert False, "don't know this file type"
        if crop:
            resized = tf.image.random_crop(decoded, [outputshape[0], outputshape[1], 3])
        else:
            resized = tf.image.resize(decoded, outputshape)
        if flip:
            resized = tf.image.random_flip_left_right(resized)
        if rot:
            resized = _random_rotate(resized)
        cast = tf.cast(resized, tf.float32)/255
        return distort(cast, outputshape)
    
    def _prep(p):
        img0 = _load_and_distort(p["file0"])
        img1 = _load_and_distort(p["file1"])
        return img0, tf.cast(tf.one_hot(p["label0"], num_domains), tf.float32), \
            img1, tf.cast(tf.one_hot(p["label1"], num_domains), tf.float32)
    
    
    while True:
        pairs = {"file0":[], "label0":[], "file1":[], "label1":[]}
        if seed:
            np.random.seed(seed)
        for _ in range(pairs_per_epoch):
            pairchoice = np.random.choice(np.arange(num_domains), replace=False, size=2)
            file0 = np.random.choice(subsets[pairchoice[0]])
            file1 = np.random.choice(subsets[pairchoice[1]])
            pairs["file0"].append(file0)
            pairs["file1"].append(file1)
            pairs["label0"].append(pairchoice[0])
            pairs["label1"].append(pairchoice[1])
    
        ds = tf.data.Dataset.from_tensor_slices(pairs)
        ds = ds.map(_prep, num_parallel_calls=num_parallel_calls)
        ds = ds.batch(batch_size)
        ds = ds.prefetch(1)
        yield ds
        
        