import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import *
from tensorflow.keras.layers import *
from tensorflow.keras import regularizers
import tensorflow_datasets as tfds
import resnet50
from IregConv2D import IregConv2D
import datetime
import os
import sys

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
    max_epochs = 164
    init_lr = .1

    def make_cbs(name: str):
        logdir = os.path.join("logs", name, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        def schedule(epoch, lr):
            if epoch < 1:
                return 0.01
            elif epoch < 82:
                return 0.1
            elif epoch < 123:
                return .01
            return 0.001

        cpcb = tf.keras.callbacks.ModelCheckpoint(filepath= os.path.join('./checkpoint', name),
                                                    save_weights_only=True,
                                                    monitor='val_accuracy',
                                                    mode='max',
                                                    save_best_only=True,
                                                    save_freq="epoch")
        lrcb = tf.keras.callbacks.LearningRateScheduler(schedule, verbose=0)

        return [cpcb, lrcb]

    train_dataset = get_dataset(batch_size, is_training=True)
    test_dataset = get_dataset(batch_size, is_training=False)
    if len(sys.argv) > 1:
        conv = IregConv2D
        name = 'Ireg'
    else: 
        conv = Conv2D
        name = 'Reg'
    
    model = resnet50.ResNet50([32, 32, 3], 
                    classes = 10, 
                    reg = regularizers.L2(0.0001),
                    Conv2D=conv)
    model.compile(loss='sparse_categorical_crossentropy',
            optimizer=Adam(init_lr),
            metrics=['accuracy'],
            )

    history = model.fit(train_dataset, epochs = max_epochs, 
                    validation_data=test_dataset, callbacks = make_cbs(name),verbose=1,
                    steps_per_epoch=1, validation_steps= 1)

    np.save(f'{name}_history.npy', history.history)

