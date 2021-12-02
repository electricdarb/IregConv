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
import boto3
from shutil import make_archive

KEY_ID = "hold" 
SECRET_KEY = "hold"

def get_dataset(batch_size, is_training=True):
    """ gets and returns the Cifar-10 datasetS
    """
    split = 'train' if is_training else 'test'
    dataset, info = tfds.load(name='cifar10', split=split, with_info=True, as_supervised=True, try_gcs=True,  shuffle_files=True)
    # Normalize the input data
    def scale(image, label):
        image = tf.cast(image, tf.float32)
        image /= 255.0
        return image, label

    dataset = dataset.map(scale)

    if is_training:
        dataset = dataset.shuffle(50000)

    dataset = dataset.batch(batch_size).cache()
    dataset = dataset.repeat()
    return dataset

if __name__ == "__main__":

    batch_size = 128
    max_epochs = 65
    init_lr = .1

    if len(sys.argv) > 0:
        conv = IregConv2D
        wpk = 3 # int(sys.argv[1])
        name = f'Ireg_wkp_{wpk}'
    else: 
        conv = Conv2D
        name = 'Reg'
        wpk = None

    logdir = os.path.join("logs", name, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    def schedule(epoch, lr):
        if epoch < 15:
            return 0.100
        elif epoch < 25:
            return 0.030
        elif epoch < 40:
            return 0.010
        elif epoch < 50:
            return 0.003
        return 0.001

    weight_folder = f"checkpoint_{name}"

    cpcb = tf.keras.callbacks.ModelCheckpoint(filepath= os.path.join(weight_folder, name),
                                                save_weights_only=True,
                                                monitor='val_accuracy',
                                                mode='max',
                                                save_best_only=True,
                                                save_freq="epoch")
    lrcb = tf.keras.callbacks.LearningRateScheduler(schedule, verbose=0)

    cbs = [cpcb, lrcb]

    train_dataset = get_dataset(batch_size, is_training=True)
    test_dataset = get_dataset(batch_size, is_training=False)

    print("conv type", name)

    model_ = resnet50.ResNet50([32, 32, 3], 
                    classes = 10, 
                    reg = regularizers.L2(0.0001),
                    Conv2D=conv, 
                    weights_per_kernel = wpk)

    input_tensor = Input(shape = [32, 32, 3],)
    x = tf.keras.layers.RandomTranslation(.125, .125)(input_tensor)
    y = model_(x)
    model = tf.keras.models.Model(inputs=input_tensor, outputs=y, name="ResNet50")
    
    model.compile(loss='sparse_categorical_crossentropy',
            optimizer=SGD(init_lr, 0.9),
            metrics=['accuracy'],
            )

    history = model.fit(train_dataset, epochs = max_epochs, 
                    validation_data=test_dataset, callbacks = cbs ,verbose=1,
                    steps_per_epoch = 390, validation_steps = 78)
    
    history_filename = f'{name}_history.npy'
    np.save(history_filename, history.history)

    session = boto3.Session(
            aws_access_key_id=KEY_ID,
            aws_secret_access_key=SECRET_KEY)

    s3 = session.resource('s3')
    bucketname = 'sparse-resnet-bgill'
    result = s3.Bucket(bucketname).upload_file(os.path.join('./', history_filename), history_filename)

    make_archive(weight_folder, 'zip', root_dir=None, base_dir=None) 
    result = s3.Bucket(bucketname).upload_file(os.path.join('./', f"{weight_folder}.zip"), f"{weight_folder}.zip")
