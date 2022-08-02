# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 20:11:37 2022

@author: Łukasz Kowalik
"""
import os
import time
import numpy as np
import random
import json  
import segmentation_models as sm
from segmentation_models.utils import set_trainable
import tensorflow as tf
from skimage import io
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, f1_score
from matplotlib import pyplot as plt
from preprocessing.utils import clear_and_ttv_split, preprocess_data, generate_data
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import MeanIoU
from tensorflow.keras.optimizers import Adam, SGD

if __name__ == '__main__':
    """
    Clear previous content of the train/test/val dirs,
    then train/test/val split of the patched data
    """
    # clear_and_ttv_split('potsdam_irrg', 240)

    """
    Initial sanity check of the data
    """
    dataset_name = 'potsdam_rgb'
    patch_size = 256
    train_img_dir = os.path.join('..', 'data', dataset_name, 'data_for_augmentation_' + str(patch_size), 'train_images', 'train')
    train_label_dir = os.path.join('..', 'data', dataset_name, 'data_for_augmentation_' + str(patch_size), 'train_labels', 'train')

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
    plt.title('Oryginalny obraz')
    plt.subplot(122)
    plt.imshow(rand_label)
    plt.title('Maska')
    plt.show()


    """
    Image generator for reading data directly from the drive
    with data augmentation (horizontal + vertical flip methods)
    """
    seed = 1998
    batch_size = 14
    n_classes = 6
    epochs = 25

    backbone = 'efficientnetb1'
    model_key = 'unet'
    loss_key = 'dice'
    encoder_freeze = False
    freeze_epochs = 2

    train_aug_img_dir = os.path.join('..', 'data', dataset_name, 'data_for_augmentation_' + str(patch_size), 'train_images')
    train_aug_label_dir = os.path.join('..', 'data', dataset_name, 'data_for_augmentation_' + str(patch_size), 'train_labels')
    val_aug_img_dir = os.path.join('..', 'data', dataset_name, 'data_for_augmentation_' + str(patch_size), 'val_images')
    val_aug_label_dir = os.path.join('..', 'data', dataset_name, 'data_for_augmentation_' + str(patch_size), 'val_labels')
    test_aug_img_dir = os.path.join('..', 'data', dataset_name, 'data_for_augmentation_' + str(patch_size), 'test_images')
    test_aug_label_dir = os.path.join('..', 'data', dataset_name, 'data_for_augmentation_' + str(patch_size), 'test_labels')

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
    steps_per_epoch = int(len(os.listdir(os.path.join(train_aug_img_dir, 'train'))) // batch_size)
    val_steps_per_epoch = int(len(os.listdir(os.path.join(val_aug_img_dir, 'val'))) // batch_size)
    test_steps_per_epoch = int(len(os.listdir(os.path.join(test_aug_img_dir, 'test'))) // batch_size)

    models_dict = {
        'unet': sm.Unet(backbone, encoder_weights=None, 
                        input_shape=(256, 256, 3),
                        encoder_freeze=encoder_freeze,
                        classes=n_classes, activation='softmax'),
        'linknet': sm.Linknet(backbone, encoder_weights='imagenet', 
                        input_shape=(256, 256, 3),
                        encoder_freeze=encoder_freeze,
                        classes=n_classes, activation='softmax'),
        'fpn': sm.FPN(backbone, encoder_weights='imagenet', 
                        input_shape=(256, 256, 3),
                        encoder_freeze=encoder_freeze,
                        classes=n_classes, activation='softmax'),
        'pspnet': sm.PSPNet(backbone, encoder_weights='imagenet', 
                        input_shape=(240, 240, 3),
                        encoder_freeze=encoder_freeze,
                        classes=n_classes, activation='softmax',
                        downsample_factor=4),
    }
    model = models_dict[model_key]

    losses_dict = {
        'focal_jaccard': sm.losses.categorical_focal_jaccard_loss,
        'focal_dice': sm.losses.categorical_focal_dice_loss,
        'focal': sm.losses.categorical_focal_loss,
        'dice': sm.losses.dice_loss,
        'jaccard': sm.losses.jaccard_loss
    }
    
    # lr = 0.01
    # optimizer = Adam(lr)
    model.compile('Adam', loss=losses_dict[loss_key],
                  metrics=[sm.metrics.iou_score, sm.metrics.f1_score])

    print(model.summary())
    print(model.input_shape)

    model_dir = os.path.join('..', 'models', 
                              'isprs_' + str(dataset_name) + '_' + str(model_key) + '_' + str(backbone) + '_' + str(epochs) + 'epochs_' + loss_key + '_no_pretrain.hdf5')
    hist_dir = os.path.join('history', 
                              'isprs_' + str(dataset_name) + '_' + str(model_key) + '_' + str(backbone) + '_' + str(epochs) + 'epochs_' + loss_key + '_no_pretrain.json')
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(model_dir, verbose=1, save_best_only=True),
        tf.keras.callbacks.TensorBoard(log_dir='logs', update_freq='epoch', profile_batch=0)
    ]
    
    if encoder_freeze:
        start = time.time()
        history1 = model.fit(train_img_gen, steps_per_epoch=steps_per_epoch, validation_steps=val_steps_per_epoch,
                        epochs=freeze_epochs, validation_data=val_img_gen, verbose=1, callbacks=callbacks)
        set_trainable(model, recompile=False)
        model.compile('Adam', loss=losses_dict[loss_key],
                      metrics=[sm.metrics.iou_score, sm.metrics.f1_score])
        history2 = model.fit(train_img_gen, steps_per_epoch=steps_per_epoch, validation_steps=val_steps_per_epoch,
                        epochs=epochs-freeze_epochs, validation_data=val_img_gen, verbose=1, callbacks=callbacks)
        end = time.time()
        
        histories = [history1.history, history2.history]
        history = {}
        for key in history1.history.keys():
            history[key] = list(np.concatenate(list(h[key] for h in histories)))
    else:
        start = time.time()
        history = model.fit(train_img_gen, steps_per_epoch=steps_per_epoch, validation_steps=val_steps_per_epoch,
                        epochs=epochs, validation_data=val_img_gen, verbose=1, callbacks=callbacks)
        end = time.time()
        history = history.history

    print('Training took: ' + str(end-start) + ' seconds')
    history['time'] = end-start
    json.dump(history, open(hist_dir, 'w'))
    
    """
    Training statistics
    """
    loss = history['loss']
    val_loss = history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'y', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    # plt.savefig(os.path.join('graphs', model_key, 'loss.png'))
    plt.show()

    iou = history['iou_score']
    val_iou = history['val_iou_score']
    plt.plot(epochs, iou, 'y', label='Training IoU')
    plt.plot(epochs, val_iou, 'r', label='Validation IoU')
    plt.title('Training and validation IoU')
    plt.xlabel('Epochs')
    plt.ylabel('IoU')
    plt.legend()
    # plt.savefig(os.path.join('graphs', model_key, 'iou.png'))
    plt.show()

    fscore = history['f1-score']
    val_fscore = history['val_f1-score']
    plt.plot(epochs, fscore, 'y', label='Training F-score')
    plt.plot(epochs, val_fscore, 'r', label='Validation F-score')
    plt.title('Training and validation F-score')
    plt.xlabel('Epochs')
    plt.ylabel('F-score')
    plt.legend()
    # plt.savefig(os.path.join('graphs', model_key, 'fscore.png'))
    plt.show()
    
    """
    Sample predictions
    """
    model_path = os.path.join('..', 'models', 'isprs_potsdam_rgb_unet_efficientnetb1_25epochs_dice.hdf5')
    model = load_model(model_path, compile=False)
    test_img_gen = generate_data(test_aug_img_dir, test_aug_label_dir,
                                      batch_size, seed, n_classes, backbone, 
                                      no_augment=True)
    test_img_gen2 = generate_data(test_aug_img_dir, test_aug_label_dir,
                                      batch_size, seed, n_classes, backbone, 
                                      no_augment=True, raw_imgs=True)
    
    val_img_batch, val_label_batch = test_img_gen.__next__()
    raw_img_batch, _ = test_img_gen2.__next__()
    
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
    
    random.seed(1998)
    img_id = random.randint(0, val_img_batch.shape[0]-1)
    print(img_id)
    plt.figure(figsize=(12, 8))
    plt.subplot(131)
    plt.title('Obraz wejściowy')
    scaler = MinMaxScaler()
    raw_img_batch = scaler.fit_transform(raw_img_batch.reshape(-1, raw_img_batch.shape[-1])).reshape(raw_img_batch.shape)
    plt.imshow(raw_img_batch[img_id])
    plt.subplot(132)
    plt.title('Maska')
    plt.imshow(color_palette[val_label_batch_argmax[img_id]])
    plt.subplot(133)
    plt.title('Predykcja')
    plt.imshow(color_palette[val_pred_batch_argmax[img_id]])
    plt.show()
    
    """
    Holistic val dataset predictions with per class results
    """
    backbone = 'efficientnetb1'
    test_img_gen = generate_data(test_aug_img_dir, test_aug_label_dir,
                                      batch_size, seed, n_classes, backbone, 
                                      no_augment=True)
    model_name = 'isprs_potsdam_irrg_unet_'+backbone+'_25epochs_dice'
    model_path = os.path.join('..', 'models', model_name + '.hdf5')
    model = load_model(model_path, compile=False)
    total_cm = np.zeros((1, n_classes, n_classes))
    counter = 0
    iou_list = []
    fscore_list = []
    sm_fscore_list = []
    
    for step in range(test_steps_per_epoch+1):
        X_val, y_val = test_img_gen.__next__()
        counter += len(X_val)
        predictions = model.predict_on_batch(X_val)
        val_trues = np.argmax(y_val, axis=-1)
        val_preds = np.argmax(predictions, axis=-1)
        
        iou_keras = MeanIoU(num_classes=n_classes)
        iou_keras.update_state(val_preds, val_trues)
        iou_list.append(iou_keras.result().numpy())
        
        sm_fscore = sm.metrics.FScore(threshold=0.5)
        sm_fscore_list.append(sm_fscore(y_val, predictions))
        
        val_preds = val_preds.flatten()
        val_trues = val_trues.flatten()
        fscore_list.append(f1_score(val_trues, val_preds, labels=[i for i in range(n_classes)], average=None, zero_division=1))
        
        current_cm = confusion_matrix(val_trues, val_preds, labels=[i for i in range(n_classes)])
        total_cm += current_cm
    
    eval_dir = os.path.join('evaluations', model_name+ '.json')
    fscores = np.mean(np.array(fscore_list), axis=0)
    avg_fscore = np.mean(sm_fscore_list)
    avg_iou = np.mean(iou_list)
    evaluations = {'fscores': list(fscores), 'avg_fscore': float(avg_fscore), 'avg_iou': float(avg_iou)}
    print(fscores)
    print(avg_iou, np.mean(fscore_list), avg_fscore)
    np.set_printoptions(precision=3, suppress=True)
    sums = np.sum(total_cm, axis=2, keepdims=True)
    print(total_cm / sums)
    json.dump(evaluations, open(eval_dir, 'w'))
