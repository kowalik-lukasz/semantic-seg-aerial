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
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from matplotlib import pyplot as plt
from preprocessing.utils import clear_and_ttv_split, preprocess_data, generate_data
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import MeanIoU

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

    train_aug_img_dir = os.path.join('..', 'data', 'potsdam_rgb', 'data_for_augmentation', 'train_images')
    train_aug_label_dir = os.path.join('..', 'data', 'potsdam_rgb', 'data_for_augmentation', 'train_labels')
    val_aug_img_dir = os.path.join('..', 'data', 'potsdam_rgb', 'data_for_augmentation', 'val_images')
    val_aug_label_dir = os.path.join('..', 'data', 'potsdam_rgb', 'data_for_augmentation', 'val_labels')

    train_dir = os.path.dirname(train_img_dir)
    train_img_gen = generate_data(train_aug_img_dir, train_aug_label_dir,
                                        batch_size, seed, n_classes, backbone)
    val_img_gen = generate_data(val_aug_img_dir, val_aug_label_dir,
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

    losses_dict = {
        'focal_jaccard': sm.losses.categorical_focal_jaccard_loss,
        'focal_dice': sm.losses.categorical_focal_dice_loss,
        'focal': sm.losses.categorical_focal_loss,
        'dice': sm.losses.dice_loss,
        'jaccard': sm.losses.jaccard_loss
    }
    loss_key = 'dice'
    
    model.compile('Adam', loss=losses_dict[loss_key],
                  metrics=[sm.metrics.iou_score, sm.metrics.f1_score])

    print(model.summary())
    print(model.input_shape)

    model_dir = os.path.join('..', 'models', 
                              'isprs_postdamRGB_' + str(backbone) + '_' + str(epochs) + 'epochs_' + loss_key + '.hdf5')
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

    fscore = history.history['f1-score']
    val_fscore = history.history['val_f1-score']
    plt.plot(epochs, fscore, 'y', label='Training F-score')
    plt.plot(epochs, val_fscore, 'r', label='Validation F-score')
    plt.title('Training and validation F-score')
    plt.xlabel('Epochs')
    plt.ylabel('F-score')
    plt.legend()
    plt.show()
    
    """
    Holistic predictions with per class results
    """
    model_path = os.path.join('..', 'models', 'isprs_postdamRGB_resnet34_25epochs_focal_jaccard.hdf5')
    model = load_model(model_path, compile=False)
    
    total_cm = np.zeros((1, n_classes, n_classes))
    
    for step in range(val_steps_per_epoch):
        X_val, y_val = val_img_gen.__next__()
        predictions = model.predict_on_batch(X_val)
        val_preds = np.argmax(predictions, axis=-1)
        val_trues = np.argmax(y_val, axis=-1)
        val_preds = val_preds.flatten()
        val_trues = val_trues.flatten()
        cm = confusion_matrix(val_trues, val_preds, labels=[i for i in range(n_classes)])
        total_cm = np.add(total_cm, cm)
    
    
    """
    Sample predictions
    """
    model_path = os.path.join('..', 'models', 'isprs_postdamRGB_resnet34_25epochs_focal_jaccard.hdf5')
    model = load_model(model_path, compile=False)
    
    val_img_batch, val_label_batch = val_img_gen.__next__()
    
    val_pred_batch = model.predict(val_img_batch)
    val_label_batch_argmax = np.argmax(val_label_batch, axis=3)
    val_pred_batch_argmax = np.argmax(val_pred_batch, axis=3)
    
    IOU_keras = MeanIoU(num_classes=n_classes)
    IOU_keras.update_state(val_pred_batch_argmax[10], val_label_batch_argmax[10])
    print('Mean IoU = ', IOU_keras.result().numpy())
    
    color_palette = np.array([[255, 255, 255],
                              [0, 0, 255],
                              [0, 255, 255],
                              [0, 255, 0],
                              [255, 255, 0],
                              [255, 0, 0]])
    
    img_id = random.randint(0, val_img_batch.shape[0]-1)
    print(img_id)
    plt.figure(figsize=(12, 8))
    plt.subplot(131)
    plt.title('Image')
    plt.imshow(val_img_batch[img_id])
    plt.subplot(132)
    plt.title('Ground Truth')
    plt.imshow(color_palette[val_label_batch_argmax[img_id]])
    plt.subplot(133)
    plt.title('Prediction')
    plt.imshow(color_palette[val_pred_batch_argmax[img_id]])
    plt.show()