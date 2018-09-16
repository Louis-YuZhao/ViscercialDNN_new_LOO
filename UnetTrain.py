from __future__ import print_function

import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"
from skimage.transform import resize
import numpy as np
import subprocess
import string
import SimpleITK as sitk

from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import backend as K

from UnetDataProcessing import load_train_data, load_test_data, ReadFoldandSort
from UnetPostProcessing import VolumeDataTofiles, WriteListtoFile
# from densenet_fast import dense_block

K.set_image_data_format('channels_last')  # TF dimension ordering in this code
# channel means color channelfigure(figsize = (20,2))

#%%
IFLeaveOne = True
leave_one_out_file = 'TrainingDataFull'
#leave_one_out_file = 'TrainingDataWbCT'
#leave_one_out_file = 'TrainingDataCeCT'

#organ = '29193_first_lumbar_vertebra'
#sliceNum = 56
#image_rows = 88
#image_cols = 64

# organ = '170_pancreas'
# sliceNum = 80
# image_rows = 72
# image_cols = 120

organ = '187_gallbladder'
sliceNum = 80
image_rows = 80
image_cols = 80

#organ = '30325_left_adrenal_gland'
#sliceNum = 40
#image_rows = 56
#image_cols = 40

#organ = '30324_right_adrenal_gland'
#sliceNum = 104
#image_rows = 64
#image_cols = 48

smooth = 1.
IfglobalNorm = False

learningRate = 1e-5
batch_size = 50
epochs = 100

#%%
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def get_unet_short():
    inputs = Input((image_rows, image_cols, 1))
    masks = Input((image_rows, image_cols, 1))
    conv1 = Conv2D(32, (5, 5), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (5, 5), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)
    conv9 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    conv10Block1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)
    conv10Block2 = Conv2D(32, (3, 3), activation='relu', padding='same')(masks)
    conv11 = concatenate([conv10Block1, conv10Block2], axis=3)

    conv12 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv11)
    conv12 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv12)
    conv12 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv12)
    conv12 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv12)
    conv12 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv12)

    conv13 = Conv2D(1, (1, 1), activation='sigmoid')(conv12)

    model = Model(inputs=[inputs, masks], outputs=[conv13])

    model.compile(optimizer=Adam(lr=learningRate), loss=dice_coef_loss, metrics=[dice_coef])

    return model


def preprocess(imgs):
    imgs_p = np.ndarray((imgs.shape[0], image_rows, image_cols), dtype=np.float32)
    for i in range(imgs.shape[0]):
        imgs_p[i] = resize(imgs[i], (image_rows, image_cols), preserve_range=True)

    imgs_p = imgs_p[..., np.newaxis] # 'channels_last'
    return imgs_p 

#def train_and_predict(tempStore, modelPath):
#    print('-'*30)
#    print('Loading and preprocessing train data...')
#    print('-'*30)
#    imgs_train, imgs_label_train, addInformation_train, imgs_id_train = load_train_data(tempStore)
#
#    imgs_train = preprocess(imgs_train)
#    imgs_label_train = preprocess(imgs_label_train)
#    addInformation_train = preprocess(addInformation_train)
#
#    imgs_train = imgs_train.astype('float32')
#
#    if IfglobalNorm == True:
#        mean = np.mean(imgs_train)  # mean for data centering
#        std = np.std(imgs_train)  # std for data normalization
#        imgs_train -= mean
#        imgs_train /= std
#
##   save mean and std of training data    
#    imgs_label_train = imgs_label_train.astype(np.uint32)
#    addInformation_train = addInformation_train.astype(np.float32)
#
#    print('-'*30)
#    print('Creating and compiling model...')
#    print('-'*30)
#    ############################
#    model = get_unet_short()
#    ############################
#    model_checkpoint = ModelCheckpoint(os.path.join(modelPath,'weights.h5'), monitor='val_loss', save_best_only=True)
#    
#    early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=40, verbose=0, mode='auto')
## 
#    print('-'*30)
#    print('Fitting model...')
#    print('-'*30)
#    train_history = model.fit([imgs_train, addInformation_train], imgs_label_train, batch_size,\
#    epochs, verbose=1, shuffle=True, validation_split=0.2,\
#    callbacks=[model_checkpoint, early_stop])
#    
#    loss = train_history.history['loss']
#    val_loss = train_history.history['val_loss']
#    np.save(tempStore + '/loss.npy',loss)
#    np.save(tempStore + '/val_loss',val_loss)
#    
#    print('-'*30)
#    print('Loading and preprocessing test data...')
#    print('-'*30)
#    imgs_test, addInformation_test, imgs_id_test = load_test_data(tempStore)
#    imgs_test = preprocess(imgs_test)
#    addInformation_test = preprocess(addInformation_test)
#
#    imgs_test = imgs_test.astype('float32')
#    addInformation_test = addInformation_test.astype(np.float32)
#    
#    if IfglobalNorm == True:
#        imgs_test -= mean
#        imgs_test /= std
#
#    print('-'*30)
#    print('Loading saved weights...')
#    print('-'*30)
#    model.load_weights(os.path.join(modelPath,'weights.h5'))
#
#    print('-'*30)
#    print('Predicting masks on test data...')
#    print('-'*30)
#    imgs_label_test = model.predict([imgs_test, addInformation_test], verbose=1)
#    np.save(os.path.join(tempStore,'imgs_label_test.npy'), imgs_label_test)
#    
#    if IfglobalNorm == True:
#        np.save(os.path.join(tempStore, 'imgs_mean.npy'), mean)
#        np.save(os.path.join(tempStore, 'imgs_std.npy'), std)

def train_leave_one_out(tempStore, modelPath, testOutputDir, Reference):
    
    print('-'*30)
    print('Loading all the data...')
    print('-'*30)
    imgs_train, imgs_label_train, addInformation_train, imgs_id_train = load_train_data(tempStore)

    imgs_train = preprocess(imgs_train)
    imgs_label_train = preprocess(imgs_label_train)
    addInformation_train = preprocess(addInformation_train)

    imgs_train = imgs_train.astype('float32')

    if IfglobalNorm == True:
        mean = np.mean(imgs_train)  # mean for data centering
        std = np.std(imgs_train)  # std for data normalization
        imgs_train -= mean
        imgs_train /= std

#   save mean and std of training data    
    imgs_label_train = imgs_label_train.astype(np.uint32)
    addInformation_train = addInformation_train.astype(np.float32)

    TotalNum = len(imgs_id_train)
    preImageList = []
    for i in xrange(TotalNum):
        inBaseName = os.path.basename(imgs_id_train[i])
        outBaseName = string.join(inBaseName.split("_")[-4:-1], "_")
        currentTrainImgs = np.delete(imgs_train,range(i*sliceNum,(i+1)*sliceNum), axis=0)
        currentTrainLab = np.delete(imgs_label_train,range(i*sliceNum,(i+1)*sliceNum), axis=0)
        currentTrainAdd = np.delete(addInformation_train,range(i*sliceNum,(i+1)*sliceNum), axis=0)
    
        # begin the model
        #############################
        model = get_unet_short()
        #############################
        weightName = modelPath + '/' + outBaseName + '_weights.h5'
        model_checkpoint = ModelCheckpoint(weightName, monitor='val_loss', save_best_only=True)
        early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=40, verbose=0, mode='auto')        

        train_history = model.fit([currentTrainImgs, currentTrainAdd], currentTrainLab, batch_size,\
        epochs, verbose=1, shuffle=True, validation_split=0.2,\
        callbacks=[model_checkpoint, early_stop])               

        loss = train_history.history['loss']
        val_loss = train_history.history['val_loss']
        np.save(tempStore + '/' + outBaseName + '_loss.npy',loss)
        np.save(tempStore + '/' + outBaseName + '_val_loss',val_loss)
        
        # prediction
        currentTestImgs = imgs_train[i*sliceNum:(i+1)*sliceNum,:,:,:]
        currentTestAdd = addInformation_train[i*sliceNum:(i+1)*sliceNum,:,:,:]               
        
        model.load_weights(weightName)
        imgs_label_test = model.predict([currentTestImgs,currentTestAdd], verbose=1)
        ThreeDImagePath = VolumeDataTofiles(imgs_label_test, outBaseName, testOutputDir, Reference)
        preImageList.append(ThreeDImagePath)    
        # np.save(tempStore + '/' + outBaseName + '_imgs_label_test.npy', imgs_label_test)

        print('-'*30)
        print(str(i) + 'th is finished...')
        print('-'*30)

    WriteListtoFile(preImageList, testOutputDir + '/FileList.txt')

if __name__ == '__main__':
    tempStore = './tempData_' + organ
    modelPath = './model_' + organ 
    if not os.path.exists(tempStore):
        subprocess.call('mkdir ' + '-p ' + tempStore, shell=True)
    if not os.path.exists(modelPath):
        subprocess.call('mkdir ' + '-p ' + modelPath, shell=True)
    
    if IFLeaveOne != True:
        train_and_predict(tempStore, modelPath)
    else:
        input_data_path = '/home/louis/project/ViscercialDNN_new_LOO/'
        output_data_path = '/home/louis/project/ViscercialDNN_new_LOO/'
        reflist = ReadFoldandSort(os.path.join(input_data_path, leave_one_out_file, organ + '_Linear_Imagepatch'))
        refImage = reflist[0]
        Reference={}
        refImage = sitk.ReadImage(refImage)
        Reference['origin'] = refImage.GetOrigin()
        Reference['spacing'] = refImage.GetSpacing()
        Reference['direction'] = refImage.GetDirection()
    
        ThreeDImageDir = os.path.join (output_data_path, 'Pred3D', organ + leave_one_out_file)
        if not os.path.exists(ThreeDImageDir):
            subprocess.call('mkdir ' + '-p ' + ThreeDImageDir, shell=True)
        train_leave_one_out(tempStore, modelPath, ThreeDImageDir, Reference)
