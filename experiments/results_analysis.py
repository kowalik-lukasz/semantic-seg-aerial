# -*- coding: utf-8 -*-
"""
Created on Thu May 12 21:28:37 2022

@author: Łukasz Kowalik
"""
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import os
import json
import datetime

datasets = ['RGB', 'IRRG']
archs = ['Unet', 'Linknet', 'FPN']
backbones = ['VGG16', 'VGG19', 'ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'SEResNet18', 'SEResNet34', 'SEResNet50', 'SEResNet101', 'EfficientNetB0', 'EfficientNetB1', 'EfficientNetB2']
histories_arch = {}
histories_back = {}
histories_rgb_irrg = {}

for arch in archs:
    file = open(os.path.join('history', 'isprs_potsdam_rgb_' + arch.lower() + '_resnet34_25epochs_dice.json'))
    histories_arch[arch] = json.load(file)
    
for backbone in backbones:
    file = open(os.path.join('history', 'isprs_potsdam_rgb_unet_' + backbone.lower() + '_25epochs_dice.json'))
    histories_back[backbone] = json.load(file)

for dataset in datasets:
    file1 = open(os.path.join('history', 'isprs_potsdam_' + dataset.lower() + '_unet_efficientnetb1_25epochs_dice.json'))
    file2 = open(os.path.join('history', 'isprs_potsdam_' + dataset.lower() + '_unet_efficientnetb1_25epochs_dice_no_pretrain.json'))
    histories_rgb_irrg[dataset + ' pretrenowany'] = json.load(file1)
    histories_rgb_irrg[dataset] = json.load(file2)

"""
Comparative analysis
"""
for key, data in histories_arch.items():
    val_loss = data['val_loss']
    epochs = range(1, len(val_loss) + 1)
    plt.plot(epochs, val_loss, label=key)
    
plt.ylim([0.15, 0.55])    
# plt.title('Zestawienie krzywych straty na zbiorze walidacyjnym')
plt.xlabel('Nr epoki')
plt.ylabel('Wartość funkcji straty')
plt.legend()
plt.grid(alpha=.7)
plt.tight_layout()
plt.savefig(os.path.join('graphs', 'comparative', 'sem_arch_val_loss.png'), dpi=200)
plt.show()


for key, data in histories_arch.items():
    loss = data['loss']
    epochs = range(1, len(val_loss) + 1)
    plt.plot(epochs, loss, label=key)
    
plt.ylim([0.15, 0.55])    
# plt.title('Zestawienie krzywych straty na zbiorze uczącym')
plt.xlabel('Nr epoki')
plt.ylabel('Wartość funkcji straty')
plt.legend()
plt.grid(alpha=.7)
plt.tight_layout()
plt.savefig(os.path.join('graphs', 'comparative', 'sem_arch_train_loss.png'), dpi=200)
plt.show()


times, params = [], []
losses = [0.7464, 0.7305, 0.7759, 0.7712, 0.7623, 0.7015, 0.7728, 0.7796, 0.7873, 0.7444, 0.8032, 0.8064, 0.8009]
for key, data in histories_back.items():
    times.append(data['time'])
    params.append(data['params'])

params = [p/100000 for p in params]
colors = ['tab:orange', 'tab:orange', 'tab:blue', 'tab:blue', 'tab:blue', 'tab:blue', 'tab:green', 'tab:green', 'tab:green', 'tab:green', 'tab:red', 'tab:red', 'tab:red']
sc = plt.scatter(times, losses, params, colors, alpha=0.4, edgecolors='black')
plt.ylim([0.66, 0.82]) 
plt.xlabel('Czas uczenia (sekundy)')
plt.ylabel('Wartość metryki F1')
plt.grid(alpha=.7)
plt.tight_layout()
kw = dict(prop="sizes", num=6, fmt="{x:.0f}M",
          func=lambda s: s/10)
for i, backbone in enumerate(backbones):
    if backbone == 'EfficientNetB0':
        plt.annotate(backbone, (times[i], losses[i]), xytext=(times[i], losses[i]-0.007))
    elif backbone == 'EfficientNetB2':
        plt.annotate(backbone, (times[i], losses[i]), xytext=(times[i], losses[i]-0.005))
    elif backbone == 'ResNet18':
        plt.annotate(backbone, (times[i], losses[i]), xytext=(times[i], losses[i]+0.005))
    elif backbone == 'ResNet34':
        plt.annotate(backbone, (times[i], losses[i]), xytext=(times[i], losses[i]-0.01))
    elif backbone == 'SEResNet18':
        plt.annotate(backbone, (times[i], losses[i]), xytext=(times[i]+90, losses[i]-0.001))
    elif backbone == 'SEResNet34':
        plt.annotate(backbone, (times[i], losses[i]), xytext=(times[i], losses[i]-0.005))
    elif backbone == 'SEResNet101':
        plt.annotate(backbone, (times[i], losses[i]), xytext=(times[i]-1000, losses[i]))
    else:
        plt.annotate(backbone, (times[i], losses[i]))
plt.legend(*sc.legend_elements(**kw), labelspacing=1.3, loc='lower left', title='Liczba parametrów', ncol=6, handletextpad=0.2, columnspacing=1)
plt.savefig(os.path.join('graphs', 'comparative', 'sem_back_comp.png'), dpi=200)
plt.show()


colors = [(255, 0, 0), (255, 122, 0), (0, 0, 255), (0, 122, 255)]
for i, (key, data) in enumerate(histories_rgb_irrg.items()):
    print(i, key, data)
    loss = data['loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, label=key)
    
plt.ylim([0.15, 0.55])    
# plt.title('Zestawienie krzywych straty na zbiorze uczącym')
plt.xlabel('Nr epoki')
plt.ylabel('Wartość funkcji straty')
plt.legend()
plt.grid(alpha=.7)
plt.tight_layout()
plt.savefig(os.path.join('graphs', 'comparative', 'rgb_irrg_train_loss.png'), dpi=200)
plt.show()

for i, (key, data) in enumerate(histories_rgb_irrg.items()):
    print(i, key, data)
    loss = data['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, label=key)
    
plt.ylim([0.15, 0.55])    
# plt.title('Zestawienie krzywych straty na zbiorze uczącym')
plt.xlabel('Nr epoki')
plt.ylabel('Wartość funkcji straty')
plt.legend()
plt.grid(alpha=.7)
plt.tight_layout()
plt.savefig(os.path.join('graphs', 'comparative', 'rgb_irrg_val_loss.png'), dpi=200)
plt.show()


"""
Times
"""
times = []
for key, data in histories_arch.items():
    times.append(data['time'])
plt.barh(np.arange(len(times)), times, color=['tab:blue', 'tab:orange', 'tab:green'])
plt.yticks(np.arange(len(times)), labels=histories_arch.keys())
plt.xlabel('Czas uczenia (w sekundach)')
plt.grid(alpha=.7)
plt.tight_layout()
plt.savefig(os.path.join('graphs', 'comparative', 'arch_bar_chart_time.png'), dpi=200)
plt.show()

times = []
for key, data in histories_back.items():
    times.append(data['time'])
plt.barh(np.arange(len(times)), times, color=['tab:orange', 'tab:orange', 'tab:blue', 'tab:blue', 'tab:blue', 'tab:blue', 'tab:green', 'tab:green', 'tab:green', 'tab:green', 'tab:red', 'tab:red', 'tab:red'])
plt.yticks(np.arange(len(times)), labels=histories_back.keys())
plt.xlabel('Czas uczenia (sekundy)')
plt.grid(alpha=.7)
plt.tight_layout()
plt.savefig(os.path.join('graphs', 'comparative', 'back_bar_chart_time.png'), dpi=200)
plt.show()

"""
Individual analysis
"""
for key, data in histories_arch.items():
    loss = histories_arch[key]['loss']
    val_loss = histories_arch[key]['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'y', label='Zbiór uczący')
    plt.plot(epochs, val_loss, 'r', label='Zbiór walidacyjny')
    # plt.title('Krzywe uczenia dla modelu ' + key)
    plt.xlabel('Nr epoki')
    plt.ylabel('Wartość funkcji straty')
    plt.ylim([0.15, 0.55])  
    plt.legend()
    plt.grid(alpha=.7)
    plt.tight_layout()
    plt.savefig(os.path.join('graphs', key.lower(), key.lower() + '_loss.png'), dpi=200)
    plt.show()
    
for key, data in histories_back.items():
    loss = histories_back[key]['loss']
    val_loss = histories_back[key]['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'y', label='Zbiór uczący')
    plt.plot(epochs, val_loss, 'r', label='Zbiór walidacyjny')
    # plt.title('Krzywe uczenia dla modelu ' + key)
    plt.xlabel('Nr epoki')
    plt.ylabel('Wartość funkcji straty')
    plt.ylim([0.15, 0.65])  
    plt.legend()
    plt.grid(alpha=.7)
    plt.tight_layout()
    plt.savefig(os.path.join('graphs', 'unet', key.lower() + '_loss.png'), dpi=200)
    plt.show()
