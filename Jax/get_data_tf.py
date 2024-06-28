### We get data from either torch or tensorflow3
## Let's go with the old pal!!!

 
import tensorflow_datasets as tfds
from tensorflow_datasets.core.utils import gcs_utils
gcs_utils._is_gcs_disabled = True
import tensorflow_models as tfm
import tensorflow as tf 
import numpy as np
import functools

tf.config.experimental.set_visible_devices([], 'GPU')

## CIFAR10
## MEAN = [0.49139968, 0.48215827 ,0.44653124]
## STD = [0.24703233, 0.24348505, 0.26158768]
## CIFAR100


MEAN = [0.5071, 0.4867, 0.4408]
STD = [0.2675, 0.2565, 0.2761]


ds_train, info = tfds.load('cifar100', split='train', shuffle_files=False, as_supervised=True, with_info = True)
ds_test, info = tfds.load('cifar100', split='test', shuffle_files=False, as_supervised=True, with_info = True)

tf.random.set_seed(0)


def return_train_set(ds_train = ds_train, **kwargs):

    tf.random.set_seed(0)
    erase_ = tfm.vision.augment.RandomErasing(probability = 0.25,
                    min_area = 0.02,
                    max_area = 0.1,
                    min_aspect = 0.3,
                    min_count=1,
                    max_count=1,
                    trials=10)


    rand_aug_ = tfm.vision.augment.RandAugment(num_layers = 1, 
                            magnitude = 8.0)

    mixUp = tfm.vision.augment.MixupAndCutmix(num_classes = 100,
                                      mixup_alpha = 0.8,
                                      cutmix_alpha = 1.0,
                                      prob = 1.0,
                                      switch_prob = 0.5,
                                      label_smoothing = 0.1)
    
    def combine(img, label):
        img = tfm.vision.preprocess_ops.color_jitter(img, brightness = 0.1, contrast = 0.1, saturation = 0.1)
        #img = tfm.vision.preprocess_ops.random_crop_image(img, (3/4,3/4), area_range = (0.08,1.0))
        img = tfm.vision.preprocess_ops.resize_and_crop_image(img, (32,32), (32, 32))[0]
        img = tfm.vision.preprocess_ops.random_horizontal_flip(img)[0]
        img = rand_aug_.distort(img)
        img  = tf.cast(img, tf.float32)/255.0
        img = tfm.vision.preprocess_ops.normalize_image(img, MEAN, STD)
        img = erase_.distort(img)
        return img, label
    
    def prepare(ds) -> np.ndarray:
        batch_size = kwargs["batch_size"]
        ds = ds.map(combine, tf.data.AUTOTUNE)
        ds = ds.shuffle(5000)
        ds = ds.batch(batch_size, True)
        ds = ds.map(mixUp, tf.data.AUTOTUNE)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return tfds.as_numpy(ds)
    return prepare(ds_train)

"""
for x, y in return_train_set(**{"batch_size":16}):
    print(x.shape, y.shape)
   
"""

def return_test_set(ds_test = ds_test, **kwargs):

    tf.random.set_seed(0)
    
    def normalize_img(x, label):
    ## augomet_mage add here!!
        x  = tf.cast(x, tf.float32)/255.0
        return tfm.vision.preprocess_ops.normalize_image(x, MEAN, STD), label
    
    def prepare(ds)->np.ndarray:
        batch_size = kwargs["batch_size"]
        ds = ds.map(normalize_img, tf.data.AUTOTUNE)
        ds = ds.cache()
        ds = ds.batch(batch_size)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return tfds.as_numpy(ds)
    
    return prepare(ds_test)
"""
for x, y in return_test_set(**{"batch_size":16}):
    print(x.shape, y.shape)

"""
"""

train, val = return_train_set(**{"batch_size":10})

for x,y in train:
    print(x.shape, y.shape)"""

# I hate TF, be honest!
#.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
"""

import tensorflow as tf
import numpy as np

import torch
class take:
    def __init__(self):
        self.q = np.random.randn(10000)
    def __getitem__(self, i):
        if i >= self.__len__():
            raise IndexError
        return self.q[i:i+100],2
    def __len__(self):
        return 1000-100-1

data = tf.data.Dataset.from_generator(take()).shuffle(200).batch(32)


for i in data:
    print(i)
i.shape

if __name__ == '__main__':
    pass
"""