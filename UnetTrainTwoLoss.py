from __future__ import print_function

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from skimage.transform import resize
import numpy as np
import subprocess
import string
import SimpleITK as sitk

from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import backend as K

from UnetDataProcessing import load_train_data, load_test_data, ReadFoldandSort
from UnetPostProcessing import VolumeDataTofiles, WriteListtoFile
#from densenet_fast import dense_block

K.set_image_data_format('channels_last')  # TF dimension ordering in this code

#%%
IFLeaveOne = True
smooth = 1.
IfglobalNorm = False

learningRate = 1e-5
batch_size = 50
patience = 20
loss_weights=[1, 0.1]

leave_one_out_file = 'TrainingDataFull'
#leave_one_out_file = 'TrainingDataWbCT'
#leave_one_out_file = 'TrainingDataCeCT'

organList = []
organdict1 = dict()
organdict2 = dict()
organdict3 = dict()
organdict4 = dict()

organdict1['organ'] = '170_pancreas'
organdict1['sliceNum'] = 80
organdict1['image_rows'] = 72
organdict1['image_cols'] = 120
organdict1['learningRate'] = learningRate
organdict1['batch_size'] = batch_size
organdict1['epochs'] = 100
organList.append(organdict1)

organdict2['organ'] = '187_gallbladder'
organdict2['sliceNum'] = 80
organdict2['image_rows'] = 80
organdict2['image_cols'] = 80
organdict2['learningRate'] = learningRate
organdict2['batch_size'] = batch_size
organdict2['epochs'] = 100
organList.append(organdict2)

organdict3['organ'] = '30325_left_adrenal_gland'
organdict3['sliceNum'] = 40
organdict3['image_rows'] = 56
organdict3['image_cols'] = 40
organdict3['learningRate'] = learningRate
organdict3['batch_size'] = batch_size
organdict3['epochs'] = 100
organList.append(organdict3)

organdict4['organ'] = '30324_right_adrenal_gland'
organdict4['sliceNum'] = 104
organdict4['image_rows'] = 64
organdict4['image_cols'] = 48
organdict4['learningRate'] = learningRate
organdict4['batch_size'] = batch_size
organdict4['epochs'] = 100
organList.append(organdict4)

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def get_unet_short_twoloss(image_rows, image_cols, learningRate):
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
    conv9out = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    conv10Block1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)
    conv10Block2 = Conv2D(32, (3, 3), activation='relu', padding='same')(masks)
    conv11 = concatenate([conv10Block1, conv10Block2], axis=3)

    conv12 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv11)
    conv12 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv12)
    conv12 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv12)
    conv12 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv12)
    conv12 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv12)

    conv13 = Conv2D(1, (1, 1), activation='sigmoid')(conv12)

    model = Model(inputs=[inputs, masks], outputs=[conv13, conv9out])

    model.compile(optimizer=Adam(lr=learningRate), loss=[dice_coef_loss, dice_coef_loss], loss_weights=loss_weights)

    return model

def preprocess(imgs, image_rows, image_cols):
    imgs_p = np.ndarray((imgs.shape[0], image_rows, image_cols), dtype=np.float32)
    for i in range(imgs.shape[0]):
        imgs_p[i] = resize(imgs[i], (image_rows, image_cols), preserve_range=True)

    imgs_p = imgs_p[..., np.newaxis] # 'channels_last'
    return imgs_p 

def train_leave_one_out(tempStore, modelPath, testOutputDir, Reference, config):

    sliceNum = config['sliceNum']
    image_rows = config['image_rows']
    image_cols = config['image_cols']
    learningRate = config['learningRate']
    batch_size= config['batch_size']
    epochs = config['epochs']
    
    print('-'*30)
    print('Loading all the data...')
    print('-'*30)

    imgs_train, imgs_label_train, addInformation_train, imgs_id_train = load_train_data(tempStore)

    imgs_train = preprocess(imgs_train, image_rows, image_cols)
    imgs_label_train = preprocess(imgs_label_train, image_rows, image_cols)
    addInformation_train = preprocess(addInformation_train, image_rows, image_cols)

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
        ###########################################################################
        model = get_unet_short_twoloss(image_rows, image_cols, learningRate)
        ###########################################################################
        weightName = modelPath + '/' + outBaseName + '_weights.h5'
        model_checkpoint = ModelCheckpoint(weightName, monitor='val_loss', save_best_only=True)
        early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=patience, verbose=0, mode='auto')

        train_history = model.fit([currentTrainImgs, currentTrainAdd], [currentTrainLab,currentTrainLab], batch_size,\
        epochs, verbose=1, shuffle=True, validation_split=0.2,\
        callbacks=[model_checkpoint,early_stop])
        
        # model.save_weights(weightName)

        loss = train_history.history['loss']
        val_loss = train_history.history['val_loss']
        np.save(tempStore + '/' + outBaseName + '_loss.npy',loss)
        np.save(tempStore + '/' + outBaseName + '_val_loss',val_loss)
        
        # prediction
        currentTestImgs = imgs_train[i*sliceNum:(i+1)*sliceNum,:,:,:]
        currentTestAdd = addInformation_train[i*sliceNum:(i+1)*sliceNum,:,:,:]               
        
        model.load_weights(weightName)
        imgs_label_test, _ = model.predict([currentTestImgs,currentTestAdd], verbose=1)
        ThreeDImagePath = VolumeDataTofiles(imgs_label_test, outBaseName, testOutputDir, Reference)
        preImageList.append(ThreeDImagePath)    
        # np.save(tempStore + '/' + outBaseName + '_imgs_label_test.npy', imgs_label_test)

        print('-'*30)
        print(str(i) + 'th is finished...')
        print('-'*30)

    WriteListtoFile(preImageList, testOutputDir + '/FileList.txt')

def main(input_data_path, output_data_path, config):

    organ = config['organ']
    print (organ)
    tempStore = './tempData_' + organ
    modelPath = './model_' + organ
    if not os.path.exists(tempStore):
        subprocess.call('mkdir ' + '-p ' + tempStore, shell=True)
    if not os.path.exists(modelPath):
        subprocess.call('mkdir ' + '-p ' + modelPath, shell=True)

    reflist = ReadFoldandSort(os.path.join(input_data_path, leave_one_out_file, organ + '_Linear_Imagepatch'))
    refImage = reflist[0]
    Reference={}
    refImage = sitk.ReadImage(refImage)
    Reference['origin'] = refImage.GetOrigin()
    Reference['spacing'] = refImage.GetSpacing()
    Reference['direction'] = refImage.GetDirection()

    ThreeDImageDir = os.path.join(output_data_path, 'Pred3D', organ + leave_one_out_file)
    if not os.path.exists(ThreeDImageDir):
        subprocess.call('mkdir ' + '-p ' + ThreeDImageDir, shell=True)
    train_leave_one_out(tempStore, modelPath, ThreeDImageDir, Reference, config)

if __name__ == '__main__':
    input_data_path = os.path.abspath(os.path.dirname(os.getcwd()))
    output_data_path = os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),'TwoLoss')
    if not os.path.exists(output_data_path):
        subprocess.call('mkdir ' + '-p ' + output_data_path, shell=True)
    
    #main(input_data_path, output_data_path, organList[0])
    for config in organList:
        main(input_data_path, output_data_path, config)

