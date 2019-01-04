from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.layers import Conv2D, AveragePooling2D
from tensorflow.keras.layers import SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import SpatialDropout2D
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.constraints import max_norm

import numpy as np
from dataset import EEGDataset
from utils import parse_config
from torch.utils.data import DataLoader


def EEGNet(nb_classes=2, Chans=5, Samples=1024 * 4, dropoutRate=0.25, kernLength=32,
           F1=4, D=2, F2=8, norm_rate=0.25, dropoutType='Dropout') -> Model:
    """ Keras Implementation of EEGNet
    http://iopscience.iop.org/article/10.1088/1741-2552/aace8c/meta
    Note that this implements the newest version of EEGNet and NOT the earlier
    version (version v1 and v2 on arxiv). We strongly recommend using this
    architecture as it performs much better and has nicer properties than
    our earlier version. For example:

        1. Depthwise Convolutions to learn spatial filters within a
        temporal convolution. The use of the depth_multiplier option maps
        exactly to the number of spatial filters learned within a temporal
        filter. This matches the setup of algorithms like FBCSP which learn
        spatial filters within each filter in a filter-bank. This also limits
        the number of free parameters to fit when compared to a fully-connected
        convolution.

        2. Separable Convolutions to learn how to optimally combine spatial
        filters across temporal bands. Separable Convolutions are Depthwise
        Convolutions followed by (1x1) Pointwise Convolutions.


    While the original paper used Dropout, we found that SpatialDropout2D
    sometimes produced slightly better results for classification of ERP
    signals. However, SpatialDropout2D significantly reduced performance
    on the Oscillatory dataset (SMR, BCI-IV Dataset 2A). We recommend using
    the default Dropout in most cases.

    Assumes the input signal is sampled at 128Hz. If you want to use this model
    for any other sampling rate you will need to modify the lengths of temporal
    kernels and average pooling size in blocks 1 and 2 as needed (double the
    kernel lengths for double the sampling rate, etc). Note that we haven't
    tested the model performance with this rule so this may not work well.

    The model with default parameters gives the EEGNet-4,2 model as discussed
    in the paper. This model should do pretty well in general, although as the
    paper discussed the EEGNet-8,2 (with 8 temporal kernels and 2 spatial
    filters per temporal kernel) can do slightly better on the SMR dataset.
    Other variations that we found to work well are EEGNet-4,1 and EEGNet-8,1.
    We set F2 = F1 * D (number of input filters = number of output filters) for
    the SeparableConv2D layer. We haven't extensively tested other values of this
    parameter (say, F2 < F1 * D for compressed learning, and F2 > F1 * D for
    overcomplete). We believe the main parameters to focus on are F1 and D.
    Inputs:

      nb_classes      : int, number of classes to classify
      Chans, Samples  : number of channels and time points in the EEG data
      dropoutRate     : dropout fraction
      kernLength      : length of temporal convolution in first layer. We found
                        that setting this to be half the sampling rate worked
                        well in practice. For the SMR dataset in particular
                        since the data was high-passed at 4Hz we used a kernel
                        length of 32.
      F1, F2          : number of temporal filters (F1) and number of pointwise
                        filters (F2) to learn. Default: F1 = 4, F2 = F1 * D.
      D               : number of spatial filters to learn within each temporal
                        convolution. Default: D = 2
      dropoutType     : Either SpatialDropout2D or Dropout, passed as a string.
    """

    if dropoutType == 'SpatialDropout2D':
        dropoutType = SpatialDropout2D
    elif dropoutType == 'Dropout':
        dropoutType = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')

    input1 = Input(shape=(1, Chans, Samples))

    ##################################################################
    block1 = Conv2D(F1, (1, kernLength), padding='same', use_bias=False)(input1)
    block1 = BatchNormalization(axis=1)(block1)

    block1 = DepthwiseConv2D((Chans, 1), use_bias=False, depth_multiplier=D, depthwise_constraint=max_norm(1.))(block1)
    block1 = BatchNormalization(axis=1)(block1)
    block1 = Activation('elu')(block1)
    block1 = AveragePooling2D((1, 4))(block1)
    block1 = dropoutType(dropoutRate)(block1)

    block2 = SeparableConv2D(F2, (1, 16), use_bias=False, padding='same')(block1)
    block2 = BatchNormalization(axis=1)(block2)
    block2 = Activation('elu')(block2)
    block2 = AveragePooling2D((1, 8))(block2)
    block2 = dropoutType(dropoutRate)(block2)

    flatten = Flatten(name='flatten')(block2)

    dense = Dense(1, name='dense', kernel_constraint=max_norm(norm_rate))(flatten)
    softmax = Activation('sigmoid', name='sigmoid')(dense)

    return Model(inputs=input1, outputs=softmax)


default_params = {
    'config_file': 'eeg_net.yml',
    'num_data_workers': 1,
    'random_seed': 1373,
    'cuda_device': 0,
    'batch_size': 256,
    'num_epochs': 20
}

if __name__ == '__main__':
    params = parse_config(default_params, [EEGDataset], False)
    train_dataset, val_dataset = EEGDataset.from_config(**params['EEGDataset'])
    depth = train_dataset.max_dataset_depth - train_dataset.model_dataset_depth_offset
    train_dataset.model_depth = val_dataset.model_depth = depth
    train_dataset.alpha = val_dataset.alpha = 1.0
    model = EEGNet()
    model.compile('adam', loss='binary_crossentropy')
    train_dataloader = DataLoader(train_dataset, params['batch_size'], shuffle=True, drop_last=True)
    val_dataloader = DataLoader(val_dataset, params['batch_size'], shuffle=False, drop_last=False)
    train_set_x = np.stack([train_dataset[i]['x'].numpy()[None, ...] for i in range(len(train_dataset))], axis=0)
    train_set_y = np.stack([train_dataset[i]['y'].numpy()[0:1] for i in range(len(train_dataset))], axis=0)
    val_set_x = np.stack([val_dataset[i]['x'].numpy()[None, ...] for i in range(len(val_dataset))], axis=0)
    val_set_y = np.stack([val_dataset[i]['y'].numpy()[0:1] for i in range(len(val_dataset))], axis=0)
    model.fit(x=train_set_x, y=train_set_y, batch_size=params['batch_size'],
              epochs=params['num_epochs'], validation_data=(val_set_x, val_set_y))
