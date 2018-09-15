# -*- coding: utf-8 -*-
"""
Post Processing after Unet.
"""
#%%
import numpy as np
import os
import subprocess
import matplotlib.pyplot as plt
#%%
preThreshold = 0.5
groundThreshold = 0.02

FileType = 'TrainingDataFull'
#FileType = 'TrainingDataWbCT'
#FileType = 'TrainingDataCeCT'

#organ = '29193_first_lumbar_vertebra'
#CTwb = 19
# organ = '170_pancreas'
# CTwb = 19
# organ = '187_gallbladder'
# CTwb = 18
# organ = '30325_left_adrenal_gland'
# CTwb = 15
organ = '30324_right_adrenal_gland'
CTwb = 14

#%%
def WriteListtoFile(filelist, filename):
    with open(filename, 'w') as f:
        for i in filelist:
            f.write(i+'\n')
    return 1

def diceComputing(data_path):        
    import CompareThePreandtruth as CTP    
    ThreeDImageDir = os.path.join (data_path, 'Pred3D', organ + FileType)
    groundTruthDir = os.path.join (data_path, FileType, organ +'_Linear_Labelpatch')
    predictInput = ThreeDImageDir + '/FileList.txt'
    groundTruthInput = groundTruthDir + '/FileList.txt'
    predictOutput = os.path.join(data_path, 'Pred3DMod', organ + FileType )
    if not os.path.exists(predictOutput):
        subprocess.call('mkdir ' + '-p ' + predictOutput, shell=True)
    
    dicorestat = CTP.CompareThePreandTruth(predictInput, groundTruthInput)
    dicorestat.readPredictImagetoList()
    dicorestat.readgroundTruthtoList()
    Filelist = dicorestat.predictModification(predictOutput, preThreshold)
    Filelist.sort()
    WriteListtoFile(Filelist, os.path.join(predictOutput,"FileList.txt"))
        
    print('-'*30)
    print('printing the DSC...')
    print('-'*30)
    diceScore = dicorestat.diceScoreStatistics()
    wbCTDiceScore = diceScore[:CTwb]
    dice_Statistics = {}
    dice_Statistics['mean'] = np.mean(wbCTDiceScore)
    dice_Statistics['std'] = np.std(wbCTDiceScore)
    dice_Statistics['max'] = np.amax(wbCTDiceScore)
    dice_Statistics['min'] = np.amin(wbCTDiceScore)
    print dice_Statistics    
    
    CTceDiceScore = diceScore[CTwb:]
    dice_Statistics = {}
    dice_Statistics['mean'] = np.mean(CTceDiceScore)
    dice_Statistics['std'] = np.std(CTceDiceScore)
    dice_Statistics['max'] = np.amax(CTceDiceScore)
    dice_Statistics['min'] = np.amin(CTceDiceScore)
    print dice_Statistics 
    
    
    print('-'*30)
    print('printing the TPR...')
    print('-'*30)
    TPR = dicorestat.TPRStatistics()
    wbCTTPR = TPR[:CTwb]
    TPR_Statistics = {}
    TPR_Statistics['mean'] = np.mean(wbCTTPR)
    TPR_Statistics['std'] = np.std(wbCTTPR)
    TPR_Statistics['max'] = np.amax(wbCTTPR)
    TPR_Statistics['min'] = np.amin(wbCTTPR)
    print TPR_Statistics    
    
    CTceTPR = TPR[CTwb:]
    TPR_Statistics = {}
    TPR_Statistics['mean'] = np.mean(CTceTPR)
    TPR_Statistics['std'] = np.std(CTceTPR)
    TPR_Statistics['max'] = np.amax(CTceTPR)
    TPR_Statistics['min'] = np.amin(CTceTPR)
    print TPR_Statistics 


def diceComputing_full():        
    import CompareThePreandtruth as CTP
    data_path = '/media/data/louis/ProgramWorkResult/ViscercialDNN_new_LOO/'
    ThreeDImageDir = os.path.join (data_path, 'Pred3D', organ + FileType)
    groundTruthDir = os.path.join (data_path, FileType, organ +'_Linear_Labelpatch')
    predictInput = ThreeDImageDir + '/FileList.txt'
    groundTruthInput = groundTruthDir + '/FileList.txt'
    predictOutput = os.path.join(data_path, 'Pred3DMod', organ + FileType)
    if not os.path.exists(predictOutput):
        subprocess.call('mkdir ' + '-p ' + predictOutput, shell=True)
    
    dicorestat = CTP.CompareThePreandTruth(predictInput, groundTruthInput)
    dicorestat.readPredictImagetoList()
    dicorestat.readgroundTruthtoList()
    Filelist = dicorestat.predictModification(predictOutput, preThreshold)
    Filelist.sort()
    WriteListtoFile(Filelist, os.path.join(predictOutput,"FileList.txt"))
    print('-'*30)
    print('printing the DSC...')
    print('-'*30)
    dicorestat.diceScoreStatistics()
    print('-'*30)
    print('printing the TPR...')
    print('-'*30)
    dicorestat.TPRStatistics()

def showlosscurve():
    tempStore = './tempData_' + organ
#    loss = np.load(os.path.join(tempStore,'loss.npy'))
#    val_loss = np.load(os.path.join(tempStore,'val_loss.npy'))
    loss = np.load(os.path.join(tempStore,'10000020_1_CT_loss.npy'))
    val_loss = np.load(os.path.join(tempStore,'10000020_1_CT_val_loss.npy')) 
    plt.plot(loss)
    plt.plot(val_loss)
    plt.legend(['loss', 'val_loss'])
    plt.show()

if __name__ == '__main__':
    print organ
    data_path = '/media/data/louis/ProgramWorkResult/ViscercialDNN_new_LOO/'
    diceComputing(data_path)
#    diceComputing_full()
    showlosscurve()