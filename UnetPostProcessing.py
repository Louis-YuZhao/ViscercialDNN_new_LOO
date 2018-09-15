# -*- coding: utf-8 -*-
"""
Post Processing after Unet.
"""
import os
import subprocess
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt

#%%
preThreshold = 0.5
groundThreshold = 0.02

#organ = '29193_first_lumbar_vertebra'
#sliceNum = 56
#image_rows = 88
#image_cols = 64

#organ = '170_pancreas'
#sliceNum = 80
#image_rows = 72
#image_cols = 120

organ = '187_gallbladder'
sliceNum = 80
image_rows = 80
image_cols = 80

#organ = '30325_left_adrenal_gland'
#sliceNum = 40
#image_rows = 56
#image_cols = 40

# organ = '30324_right_adrenal_gland'
# sliceNum = 104
# image_rows = 64
# image_cols = 48
#%%
def WriteListtoFile(filelist, filename):
    with open(filename, 'w') as f:
        for i in filelist:
            f.write(i+'\n')
    return 1

def VolumeDataTofiles(inputArray, inputImId, ThreeDImageDir, Reference):
    """
    save np.array to .nrrd image.
    """   
    
    threeDImageArray = inputArray[:,:,:,0]
    
    threeDimage = sitk.GetImageFromArray(threeDImageArray)
    threeDimage.SetOrigin(Reference['origin'])                               
    threeDimage.SetSpacing(Reference['spacing'])                                
    threeDimage.SetDirection(Reference['direction'])
    
    ThreeDImagePath = os.path.join (ThreeDImageDir, inputImId + '_pred' +'.nrrd')
    sitk.WriteImage(threeDimage, ThreeDImagePath)    
    return ThreeDImagePath
    
def VolumeDataToVolumes(tempStore, ThreeDImageDir, Reference, sliceNum, image_rows, image_cols):

    imgs_label_test = np.load(os.path.join(tempStore,'imgs_label_test.npy'))
    dimZ, _, _, _ = np.shape(imgs_label_test)
    NumOfImage = int(dimZ/sliceNum)
    
    imgs_id_test = np.load(os.path.join(tempStore,'imgs_id_test.npy'))

    ImageList = []
    for i in xrange(NumOfImage):
        threeDImageArray = np.zeros((sliceNum, image_rows, image_cols))
        for j in xrange(sliceNum):
            threeDImageArray[j,:,:] = imgs_label_test[i*sliceNum+j,:,:,0]
        
        threeDimage = sitk.GetImageFromArray(threeDImageArray)
        threeDimage.SetOrigin(Reference['origin'])
        threeDimage.SetSpacing(Reference['spacing'])                                
        threeDimage.SetDirection(Reference['direction'])
        
        ThreeDImagePath = os.path.join(ThreeDImageDir, imgs_id_test[i] + '_pred' +'.nrrd')
        sitk.WriteImage(threeDimage, ThreeDImagePath)
        ImageList.append(ThreeDImagePath)

    WriteListtoFile(ImageList, ThreeDImageDir + '/FileList.txt')
    return ImageList

def showlosscurve():
    loss = np.load(os.path.join(tempStore,'loss.npy'))
    val_loss = np.load(os.path.join(tempStore,'val_loss.npy'))
    plt.plot(loss)
    plt.plot(val_loss)
    plt.legend(['loss', 'val_loss'])
    plt.show()
    
def Test3DDataCollecting(tempStore):
    data_path = '/media/data/louis/ProgramWorkResult/ViscercialDNN_new_LOO/'
    ThreeDImageDir = data_path + 'Pred3D/'
    if not os.path.exists(ThreeDImageDir):
        subprocess.call('mkdir ' + '-p ' + ThreeDImageDir, shell=True)
    refImage = data_path + 'TrainingData/' + organ + '_Linear_Imagepatch/patch_ImRegResult_Imdownsample_CutResult_sizeAdjust_10000020_1_CT_wb.nrrd'
    Reference={}
    refImage = sitk.ReadImage(refImage)
    Reference['origin'] = refImage.GetOrigin()
    Reference['spacing'] = refImage.GetSpacing()
    Reference['direction'] = refImage.GetDirection()
    VolumeDataToVolumes(tempStore,ThreeDImageDir, Reference, sliceNum, image_rows, image_cols)
    
def diceComputing():        
    import CompareThePreandtruth as CTP
    data_path = '/media/data/louis/ProgramWorkResult/ViscercialDNN_new_LOO/'
    ThreeDImageDir = data_path + 'Pred3D/'
    groundTruthDir = data_path + 'TestData/'+ organ +'_Linear_Labelpatch/'
    predictInput = ThreeDImageDir + 'FileList.txt'
    groundTruthInput = groundTruthDir + 'FileList.txt'
    predictOutput = data_path + 'Pred3DMod/'
    if not os.path.exists(predictOutput):
        subprocess.call('mkdir ' + '-p ' + predictOutput, shell=True)
#    groundTruthOutput = data_path + 'GT3DMod'
    
    dicorestat = CTP.CompareThePreandTruth(predictInput, groundTruthInput)
    dicorestat.readPredictImagetoList()
    dicorestat.readgroundTruthtoList()
    dicorestat.predictModification(predictOutput, preThreshold)
#    dicorestat.groundTruthModification(groundTruthOutput, groundThreshold)
    dicorestat.diceScoreStatistics()     

if __name__ == '__main__':
    tempStore = './tempData_' + organ
    print organ
    Test3DDataCollecting(tempStore)
    diceComputing()
    showlosscurve()