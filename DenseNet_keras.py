import keras 
from keras.layers import (Conv2D,
                          Dense,
                          Concatenate,
                          GlobalAveragePooling2D,
                          AveragePooling2D,
                          Activation,
                          Input,
                          BatchNormalization)
from keras.callbacks import (ModelCheckpoint, 
                             LearningRateScheduler, 
                             ReduceLROnPlateau)
from keras.regularizers import l2
from keras.models import Model
from keras.optimizers import Adam
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
import os 
import numpy as np
def preprocessing(data):
    inputs, labels = data
    inputs = inputs.astype('float32') / 255
    labels = keras.utils.to_categorical(labels)
    return inputs, labels
def dense_block(inputs,
                num_conv=6,
                growth_rate=12,
                kernel_size=3,
                batch_normalization=True,
                activation='relu',
                weight_decay=1e-4,
                bottleneck=False):
    concat_layer = inputs
    x = inputs
    for layer in range(num_conv):
        if batch_normalization:
            x = BatchNormalization()(concat_layer)
        x = Activation(activation)(x)
        if bottleneck:
            x = Conv2D(filters=growth_rate,
                        kernel_size=1,
                        strides=(1,1),
                        activation=None,
                        kernel_initializer='he_normal',
                        kernel_regularizer=l2(weight_decay))(x)
        x = Conv2D(filters=growth_rate,
                    kernel_size=kernel_size,
                    strides=(1,1),
                    activation=None,
                    padding='same',
                    kernel_initializer='he_normal',
                    kernel_regularizer=l2(weight_decay))(x)
        concat_layer = Concatenate()([concat_layer,x])   
    if batch_normalization:
        concat_layer = BatchNormalization()(concat_layer)
    concat_layer = Activation(activation)(concat_layer)
    return concat_layer
def transition_layer(inputs,
                         compression=1.0,
                         weight_decay=1e-4):
    num_channels_inputs = inputs.get_shape().as_list()[-1]
    num_filters = int(num_channels_inputs*compression)
      
    x = Conv2D(filters=num_filters,
               kernel_size=1,
               strides=(1,1),
               activation=None,
               kernel_initializer='he_normal',
               kernel_regularizer=l2(weight_decay))(inputs)
    x = AveragePooling2D()(x)
    return x
    
def DenseNet(input_shape=(32, 32, 3),
             growth_rate=12,
             num_dense_blocks=3,
             num_convs=[6,6,6],
             bottleneck=False,
             compression=1.0,
             batch_normalization=True,
             activation='relu',
             kernel_size=3,
             weight_decay=1e-4,
             num_classes=10):
  
    inputs = Input(input_shape)
#the first convolutional layers
    x = Conv2D(filters=growth_rate*2,
               kernel_size=1,
               strides=(1,1),
               activation=None,
               kernel_initializer='he_normal',
               kernel_regularizer=l2(weight_decay))(inputs)
 #dense block and transitional stage
    for num_conv in range(len(num_convs)):
        x = dense_block(inputs=x,
                        growth_rate=growth_rate,
                        kernel_size=kernel_size,
                        batch_normalization=batch_normalization,
                        weight_decay=weight_decay)
        if num_conv < len(num_convs) - 1:
            x = transition_layer(inputs=x,
                                 compression=compression,
                                 weight_decay=weight_decay)
    x = GlobalAveragePooling2D()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model
#--------------------------------------------------------------------------------
#                                     IMPLEMENTATION 
#--------------------------------------------------------------------------------
#load data
train_set, test_set = cifar10.load_data()
inputs_train, labels_train = preprocessing(train_set)
inputs_test, labels_test = preprocessing(test_set)

#model parameters
input_shape=(32, 32, 3)
growth_rate=12
num_dense_blocks=3
num_convs=[6,6,6]
bottleneck=False
compression=1.0
batch_normalization=True,
activation='relu'
kernel_size=3
weight_decay=1e-4
num_classes=10
depth = num_dense_blocks + (1+int(bottleneck))*np.sum(num_convs) + 1

#training parameters
batch_size = 128 
epochs = 200
data_augmentation = True
model_name = 'DenseNet%d' % (depth)

#implementing model
densenet = DenseNet()

#compiling model
densenet.compile(loss = 'categorical_crossentropy',
                 optimizer = Adam(),
                 metrics = ['accuracy'])
#learning rate schedule
def lr_schedule(epoch):
    lr = 1e-3
    if epoch > 150:
        lr *= lr * 1e-3
    if epoch > 120:
        lr *= lr * 1e-2
    if epoch > 80:
        lr *= lr * 1e-1
    
    return lr
#callbacks for later reprocibility
save_dir = os.path.join(os.getcwd(), 'saved_models')
saved_model = 'cifar10_%s_model.{epoch:03d}.h5' % model_name
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, saved_model)
checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_acc',
                             verbose=1,
                             save_best_only=True)
lr_scheduler = LearningRateScheduler(lr_schedule)
lr_reducer = ReduceLROnPlateau(factor=0.5,
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)
callbacks = [checkpoint, lr_scheduler, lr_reducer]
#fitting 
if not data_augmentation:
    densenet.fit(inputs_train, labels_train,
                 batch_size=batch_size,
                 validation_data=(inputs_test, labels_test),
                 shuffle=True,
                 callbacks=callbacks,
                 verbose=1)
else: 
    datagen = ImageDataGenerator(width_shift_range=0.09,
                                 height_shift_range=0.09,
                                 rotation_range=9,
                                 horizontal_flip=True,
                                 fill_mode='nearest')
    datagen.fit(inputs_train)
    densenet.fit_generator(datagen.flow(inputs_train, 
                                        labels_train, 
                                        batch_size=batch_size),
                        validation_data=(inputs_test, labels_test),
                        epochs=epochs,
                        callbacks=callbacks,
                        verbose=1,
                        steps_per_epoch=int(len(inputs_train)/batch_size))
