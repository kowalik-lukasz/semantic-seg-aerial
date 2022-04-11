# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 20:11:37 2022

@author: Łukasz Kowalik
"""
import os
# os.add_dll_directory('C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v10.1\\bin')
import numpy as np
import random
import segmentation_models as sm
import tensorflow as tf
from skimage import io
from matplotlib import pyplot as plt
from preprocessing.utils import clear_and_ttv_split, preprocess_data, generate_train_data

if __name__ == '__main__':
    """
    Clear previous content of the train/test/val dirs,
    then train/test/val split of the patched data
    """
    # clear_and_ttv_split('potsdam_rgb', 256)


    """
    Initial sanity check of the data
    """
    train_img_dir = os.path.join('..', 'data', 'potsdam_rgb', 'train', 'images')
    train_label_dir = os.path.join('..', 'data', 'potsdam_rgb', 'train', 'labels')

    img_list = os.listdir(train_img_dir)
    label_list = os.listdir(train_label_dir)
    n_images = len(img_list)

    img_index = random.randint(0, n_images-1)
    rand_img = io.imread(os.path.join(train_img_dir,img_list[img_index]))
    rand_label = io.imread(os.path.join(train_label_dir,label_list[img_index]))
    print('Random image: ' + os.path.join(train_img_dir,img_list[img_index]))
    print('Random label: ' + os.path.join(train_label_dir,label_list[img_index]))

    plt.figure(figsize=(12, 8))
    plt.subplot(121)
    plt.imshow(rand_img)
    plt.title('Image')
    plt.subplot(122)
    plt.imshow(rand_label)
    plt.title('Ground Truth')
    plt.show()


    """
    Image generator for reading data directly from the drive
    with data augmentation (horizontal + vertical flip methods)
    """
    seed = 1998
    batch_size = 14
    n_classes = 6
    epochs = 25

    backbone = 'resnet34'
    # backbone_input = sm.get_preprocessing(backbone)

    train_aug_img_dir = os.path.join('..', 'data', 'potsdam_rgb', 'data_for_augmentation', 'train_images')
    train_aug_label_dir = os.path.join('..', 'data', 'potsdam_rgb', 'data_for_augmentation', 'train_labels')
    val_aug_img_dir = os.path.join('..', 'data', 'potsdam_rgb', 'data_for_augmentation', 'val_images')
    val_aug_label_dir = os.path.join('..', 'data', 'potsdam_rgb', 'data_for_augmentation', 'val_labels')

    train_dir = os.path.dirname(train_img_dir)
    train_img_gen = generate_train_data(train_aug_img_dir, train_aug_label_dir,
                                        batch_size, seed, n_classes, backbone)
    val_img_gen = generate_train_data(val_aug_img_dir, val_aug_label_dir,
                                      batch_size, seed, n_classes, backbone)

    # X_raw, y_raw = train_img_gen.__next__()
    # for i in range(3):
    #     image = X_raw[i]
    #     label = y_raw[i]
    #     plt.subplot(1,2,1)
    #     plt.imshow(image)
    #     plt.subplot(1,2,2)
    #     plt.imshow(np.argmax(label, axis=2))
    #     plt.show()
    #
    # X_val, y_val = val_img_gen.__next__()
    # for i in range(3):
    #     image = X_raw[i]
    #     label = y_raw[i]
    #     plt.subplot(1,2,1)
    #     plt.imshow(image)
    #     plt.subplot(1,2,2)
    #     plt.imshow(np.argmax(label, axis=2))
    #     plt.show()

    """
    Model definition and training
    """
    IMG_HEIGHT = 256
    IMG_WIDTH = 256
    IMG_CHANNELS = 3

    steps_per_epoch = int(len(os.listdir(os.path.join(train_aug_img_dir, 'train'))) // batch_size)
    val_steps_per_epoch = int(len(os.listdir(os.path.join(val_aug_img_dir, 'val'))) // batch_size)
    model = sm.Unet(backbone, encoder_weights='imagenet', input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS),
                    classes=n_classes, activation='softmax')

    model.compile('Adam', loss=sm.losses.categorical_focal_jaccard_loss,
                  metrics=[sm.metrics.iou_score, sm.metrics.f1_score])

    print(model.summary())
    print(model.input_shape)

    model_dir = os.path.join('..', 'models', 'isprs_postdamRGB_' + str(epochs) + 'epochs_' + str(backbone) +'_backbone_' + str(batch_size) + 'batch.hdf5')
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(model_dir, verbose=1, save_best_only=True),
        tf.keras.callbacks.TensorBoard(log_dir='logs')
    ]
    history = model.fit(train_img_gen, steps_per_epoch=steps_per_epoch, validation_steps=val_steps_per_epoch,
                        epochs=epochs, validation_data=val_img_gen, verbose=1, callbacks=callbacks)

    """
    Training statistics
    """
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'y', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    iou = history.history['iou_score']
    val_iou = history.history['val_iou_score']
    plt.plot(epochs, iou, 'y', label='Training IoU')
    plt.plot(epochs, val_iou, 'r', label='Validation IoU')
    plt.title('Training and validation IoU')
    plt.xlabel('Epochs')
    plt.ylabel('IoU')
    plt.legend()
    plt.show()

    fscore = history.history['f_score']
    val_fscore = history.history['val_f_score']
    plt.plot(epochs, fscore, 'y', label='Training F-score')
    plt.plot(epochs, val_fscore, 'r', label='Validation F-score')
    plt.title('Training and validation F-score')
    plt.xlabel('Epochs')
    plt.ylabel('F-score')
    plt.legend()
    plt.show()