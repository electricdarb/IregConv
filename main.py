import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import *
from tensorflow.keras.layers import *
from tensorflow.keras import regularizers
import tensorflow_datasets as tfds
from time import time
import resnet50
from IregConv2D import IregConv2D
import datetime
import os

def get_dataset(batch_size, is_training=True):
    split = 'train' if is_training else 'test'
    dataset, info = tfds.load(name='cifar10', split=split, with_info=True, as_supervised=True, try_gcs=True)
    # Normalize the input data.
    def scale(image, label):
        image = tf.cast(image, tf.float32)
        image /= 255.0
        return image, label

    dataset = dataset.map(scale)

    if is_training:
        dataset = dataset.shuffle(10000)

    dataset = dataset.batch(batch_size)
    return dataset


if __name__ == "__main__":
    batch_size = 128
    init_lr = 0.1
    max_epochs = 1

    def make_cbs(name: str):
        logdir = os.path.join("logs", name, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        tbcb = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=0)
        cpcb = tf.keras.callbacks.ModelCheckpoint(filepath= os.path.join('./checkpoint', name),
                                                    save_weights_only=True,
                                                    monitor='val_accuracy',
                                                    mode='max',
                                                    save_best_only=True)
        lrcb = tf.keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss', patience = 5, factor = 0.1)

        return [tbcb, cpcb, lrcb]

    train_dataset = get_dataset(batch_size, is_training=True)
    test_dataset = get_dataset(batch_size, is_training=False)

    base_model = resnet50.ResNet50([32, 32, 3], classes = 10)
    test_model = resnet50.ResNet50([32, 32, 3], classes = 10, Conv2D=IregConv2D)

    test_model.compile(loss='sparse_categorical_crossentropy', 
            optimizer=Adam(init_lr),
            metrics=['accuracy'],
            )

    base_model.compile(loss='sparse_categorical_crossentropy',
            optimizer=Adam(init_lr),
            metrics=['accuracy'],
            )

    hist_test = test_model.fit(train_dataset, epochs = max_epochs, 
                    validation_data=test_dataset,callbacks = make_cbs('iregular'),verbose=1)

    hist_base = base_model.fit(train_dataset, epochs = max_epochs, 
                    validation_data=test_dataset,callbacks = make_cbs('regular'),verbose=1)