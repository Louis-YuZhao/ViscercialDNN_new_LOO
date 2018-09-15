#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from __future__ import print_function

import os
import subprocess
import numpy as np
import SimpleITK as sitk

#organ = '29193_first_lumbar_vertebra'
#organ = '170_pancreas'
organ = '187_gallbladder'
#organ = '30325_left_adrenal_gland'
#organ = '30324_right_adrenal_gland'

fileType = '_MAS'
IFLeaveOne = True
leave_one_out_file = 'TrainingDataFull'
#leave_one_out_file = 'TrainingDataWbCT'
#leave_one_out_file = 'TrainingDataCeCT'

#%%
def ReadFoldandSort(data_path):
    imageList = []
    for fileItem in os.listdir(data_path):
        if fileItem.endswith(".nrrd") or fileItem.endswith(".nii.gz"):
            imageList.append(os.path.join(data_path, fileItem))
    imageList.sort()
    return imageList

def ReadVolumeData(data_path, outputID, outputTitle, dataType): 
  
    imageList = []
    for fileItem in os.listdir(data_path):
        if fileItem.endswith(".nrrd") or fileItem.endswith(".nii.gz"):
            imageList.append(fileItem)
    imageList.sort()
    total = len(imageList)

    img = sitk.ReadImage(os.path.join(data_path, imageList[0]))
    imgsArray = sitk.GetArrayFromImage(img)
    mean = np.mean(imgsArray)  # mean for data centering
    std = np.std(imgsArray)  # std for data normalization
    imgsArray -= mean
    imgsArray /= std

    imgs_id = [imageList[0].split('.')[0]]
    z_dim, x_dim, y_dim = imgsArray.shape
    print('-'*30)
    print('(z, x, y) = (%s, %s, %s)' %(z_dim, x_dim, y_dim))
    print('-'*30)

    for i in xrange (1,total):
        img_id = imageList[i].split('.')[0]        
        img = sitk.ReadImage(os.path.join(data_path, imageList[i]))
        tempArray = sitk.GetArrayFromImage(img)      
        mean = np.mean(tempArray)  # mean for data centering
        std = np.std(tempArray)  # std for data normalization
        tempArray -= mean
        tempArray /= std
        imgsArray = np.concatenate((imgsArray, tempArray), axis=0)
        imgs_id.append(img_id)
                        
        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('Loading done.')
    imgsArray.astype(dataType)
    np.save(outputTitle, imgsArray)
    np.save(outputID, imgs_id)
    print('Saving to .npy files done.')
    
def ReadLabelData(data_path, outputTitle, dataType):   
    imageList = []
    for fileItem in os.listdir(data_path):
        if fileItem.endswith(".nrrd") or fileItem.endswith(".nii.gz"):
            imageList.append(fileItem)
    imageList.sort()
    total = len(imageList)

    img = sitk.ReadImage(os.path.join(data_path, imageList[0]))
    imgsArray = sitk.GetArrayFromImage(img)
    
    for i in xrange (1,total):        
        img = sitk.ReadImage(os.path.join(data_path, imageList[i]))
        tempArray = sitk.GetArrayFromImage(img)
        imgsArray = np.concatenate((imgsArray, tempArray), axis=0)                
        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('Loading done.')
    imgsArray.astype(dataType)
    np.save(outputTitle, imgsArray)
    print('Saving to .npy files done.')
    
def ReadAddData(data_path, outputTitle, dataType): 
  
    imageList = []
    for fileItem in os.listdir(data_path):
        if fileItem.endswith(".nrrd") or fileItem.endswith(".nii.gz"):
            imageList.append(fileItem)
    imageList.sort()
    total = len(imageList)

    img = sitk.ReadImage(os.path.join(data_path, imageList[0]))
    imgsArray = sitk.GetArrayFromImage(img)
    
    for i in xrange (1,total):        
        img = sitk.ReadImage(os.path.join(data_path, imageList[i]))
        tempArray = sitk.GetArrayFromImage(img)
        imgsArray = np.concatenate((imgsArray, tempArray), axis=0)                
        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('Loading done.')
    imgsArray.astype(dataType)
    np.save(outputTitle, imgsArray)
    print('Saving to .npy files done.')

def create_train_data(train_volume_path, train_label_path, train_add_path, tempStore):
    
    print('-'*30)
    print('Creating training images...')
    print('-'*30)

    # trainging volume
    outputID = os.path.join(tempStore, 'imgs_id_train.npy')
    outputTitle = os.path.join(tempStore,'imgs_train.npy')
    ReadVolumeData(train_volume_path, outputID, outputTitle, np.float32)

    # trainging label
    outputTitle = os.path.join(tempStore,'labs_train.npy')
    ReadLabelData(train_label_path, outputTitle, np.uint8)
    
    # training additional information
    outputTitle = os.path.join(tempStore,'addInformation_train.npy')
    ReadAddData(train_add_path, outputTitle, np.float32)
    
def create_test_data(test_volume_path, test_add_path, tempStore):
    
    # test volume
    outputID = os.path.join(tempStore, 'imgs_id_test.npy')
    outputTitle = os.path.join(tempStore,'imgs_test.npy')
    ReadVolumeData(test_volume_path, outputID, outputTitle, np.float32)
    
    # test additional information
    outputTitle = os.path.join(tempStore,'addInformation_test.npy')
    ReadAddData(test_add_path, outputTitle, np.float32)

def load_train_data(tempStore):
    imgs_train = np.load(os.path.join(tempStore,'imgs_train.npy'))
    imgs_label_train = np.load(os.path.join(tempStore,'labs_train.npy'))
    addInformation_train = np.load(os.path.join(tempStore,'addInformation_train.npy'))
    imgs_id_train = np.load(os.path.join(tempStore, 'imgs_id_train.npy'))
    return imgs_train, imgs_label_train, addInformation_train, imgs_id_train

def load_test_data(tempStore):
    imgs_test = np.load(os.path.join(tempStore, 'imgs_test.npy'))
    addInformation_test = np.load(os.path.join(tempStore,'addInformation_test.npy'))
    imgs_id_test = np.load(os.path.join(tempStore, 'imgs_id_test.npy'))
    return imgs_test, addInformation_test, imgs_id_test

def main(data_path):
   
    tempStore = './tempData_' + organ 
    if not os.path.exists(tempStore):
        subprocess.call('mkdir ' + '-p ' + tempStore, shell=True)

    # train part
    train_volume_path = os.path.join(data_path, 'TrainingData', organ+'_Linear_Imagepatch')
    train_label_path = os.path.join(data_path, 'TrainingData', organ+'_Linear_Labelpatch')
    train_add_path = os.path.join(data_path, 'TrainingData', organ+fileType)
    create_train_data(train_volume_path, train_label_path, train_add_path, tempStore)
    
    # test part
    test_data_path = os.path.join(data_path, 'TestData', organ+'_Linear_Imagepatch')
    test_add_path = os.path.join(data_path, 'TestData', organ+fileType)   
    create_test_data(test_data_path, test_add_path, tempStore)
    
def main_leave_one_out(data_path):
     
    tempStore = './tempData_' + organ 
    if not os.path.exists(tempStore):
        subprocess.call('mkdir ' + '-p ' + tempStore, shell=True)

    # train part
    train_volume_path = os.path.join(data_path, leave_one_out_file, organ+'_Linear_Imagepatch')
    train_label_path = os.path.join(data_path, leave_one_out_file, organ+'_Linear_Labelpatch')
    train_add_path = os.path.join(data_path, leave_one_out_file, organ+fileType)
    create_train_data(train_volume_path, train_label_path, train_add_path, tempStore)
    
if __name__ == '__main__':
    data_path = '/home/louis/project/ViscercialDNN_new_LOO'
    if IFLeaveOne != True:
        main(data_path)
    else:
        main_leave_one_out(data_path)