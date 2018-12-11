import glob
import pyedflib
import numpy as np
from tqdm import tqdm
from tensorflow.keras.models import Model
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.layers import SpatialDropout2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.layers import Conv2D, AveragePooling2D
from tensorflow.keras.layers import SeparableConv2D, DepthwiseConv2D

REF_CHANNELS = {'EEG FP1-REF', 'EEG FP2-REF'}
dataset = []
for f_name in tqdm(glob.glob('data/normal/v2.0.0/**/*.edf', recursive=True)):
    f = pyedflib.EdfReader(f_name)
    labels = {label: i for i, label in enumerate(f.getSignalLabels()) if label.lower().startswith('eeg')}
    if len(REF_CHANNELS - set(labels.keys())) == 0:
        is_ok = True
        for channel in REF_CHANNELS:
            if f.samplefrequency(labels[channel]) != 250:
                is_ok = False
                break
        if is_ok:
            dataset.append(np.stack([f.readSignal(labels[channel]) for channel in REF_CHANNELS], axis=0))


def eeg_net_keras(nb_classes, chans=64, samples=128, dropout_rate=0.25,
                  kern_length=64, f1=4, d=2, f2=8, norm_rate=0.25, use_spatial_dropout=False):
    dropout_type = SpatialDropout2D if use_spatial_dropout else Dropout
    input1 = Input(shape=(1, chans, samples))
    block1 = Conv2D(f1, (1, kern_length), padding='same', use_bias=False)(input1)
    block1 = BatchNormalization(axis=1)(block1)
    block1 = DepthwiseConv2D((chans, 1), use_bias=False, depth_multiplier=d, depthwise_constraint=max_norm(1.))(block1)
    block1 = BatchNormalization(axis=1)(block1)
    block1 = Activation('elu')(block1)
    block1 = AveragePooling2D((1, 4))(block1)
    block1 = dropout_type(dropout_rate)(block1)
    block2 = SeparableConv2D(f2, (1, 16), use_bias=False, padding='same')(block1)
    block2 = BatchNormalization(axis=1)(block2)
    block2 = Activation('elu')(block2)
    block2 = AveragePooling2D((1, 8))(block2)
    block2 = dropout_type(dropout_rate)(block2)
    flatten = Flatten(name='flatten')(block2)
    dense = Dense(nb_classes, name='dense', kernel_constraint=max_norm(norm_rate))(flatten)
    softmax = Activation('softmax', name='softmax')(dense)
    return Model(inputs=input1, outputs=softmax)
