# -*- coding: utf-8 -*-
# Louis
# 07/06/2017
# version 1
# 
# postprocessing for showing the final result

#%%
import numpy as np # Numpy for general purpose processing
import SimpleITK as sitk # SimpleITK to load images
import subprocess
import os
#%%

#organ = '187_gallbladder'
#organ = '170_pancreas' 
#organ = '30325_left_adrenal_gland' 
organ = '30324_right_adrenal_gland'
#organ = '29193_first_lumbar_vertebra' 

FileType = 'TrainingDataFull'

#%%
def readTxtIntoList(filename):
   flist = []
   with open(filename) as f:
         flist = f.read().splitlines()
   return flist

def WriteListtoFile(filelist, filename):
    with open(filename, 'w') as f:
        for i in filelist:
            f.write(i+'\n')
    return 1

def imagezoom(InputImageList, output, samplefactor, InterP):
    num_data = len(InputImageList)
    outputList = []
    for i in range(num_data):        
        image = sitk.ReadImage(InputImageList[i])
        image_array = sitk.GetArrayFromImage(image) # get numpy array
        z_dim, x_dim, y_dim = image_array.shape 
        
        # image upsampering
#       im = scipy.ndimage.interpolation.zoom(image_array, samplefactor, order = InterP)    
        img = sitk.GetImageFromArray(image_array)   
        img.SetOrigin(image.GetOrigin())    
        img.SetSpacing(image.GetSpacing())        
        img.SetDirection(image.GetDirection())        
        
        name, ext = os.path.splitext(InputImageList[i])
        baseName = os.path.basename(name)
        fn= os.path.join(output,'Imzoomed_' + baseName + '.nrrd')        
        sitk.WriteImage(img,fn)
        outputList.append(fn) 
        print 'Iteration Num %d is finished'% i        
    del image,image_array    
    return outputList

def predictRecover(predictlist, central_point, ImageInfo, outputdir):
    
    ImageDim = ImageInfo['dimAdj']
    z_im=ImageDim[0]; x_im=ImageDim[1]; y_im=ImageDim[2]
    num_data = len(predictlist)
    labelRecover = np.zeros(tuple(ImageDim))
    outputList = []
    for i in range(num_data):        
        image = sitk.ReadImage(predictlist[i])
        image_array = sitk.GetArrayFromImage(image) # get numpy array
        z_dim, x_dim, y_dim = image_array.shape 
        
        # recover
        z_begin=int(central_point[0,i]-(z_dim/2))
        x_begin=int(central_point[1,i]-(x_dim/2))
        y_begin=int(central_point[2,i]-(y_dim/2))
        
        z_end=z_begin+z_dim
        x_end=x_begin+x_dim
        y_end=y_begin+y_dim
        
        if z_begin < 0:
            pad_z = np.abs(0-z_begin)
            z_begin=z_begin+pad_z
            z_end=z_end+pad_z
        elif z_end > z_im:
            pad_z = np.abs(z_end-z_im)
            z_begin=z_begin-pad_z
            z_end=z_end-pad_z
        else:
            pass
        
        if x_begin < 0:
            pad_x = np.abs(0-x_begin)
            x_begin=x_begin+pad_x
            x_end=x_end+pad_x
        elif x_end > x_im:
            pad_x = np.abs(x_end-x_im)
            x_begin=x_begin-pad_x
            x_end=x_end-pad_x
        else:
            pass            
             
        if y_begin < 0:
            pad_y = np.abs(0-y_begin)
            y_begin=y_begin+pad_y
            y_end=y_end+pad_y
        elif y_end > y_im:
            pad_y = np.abs(y_end-y_im)
            y_begin=y_begin-pad_y
            y_end=y_end-pad_y
        else:
            pass
        
        labelRecover[z_begin:(z_begin+z_dim),x_begin:(x_begin+x_dim),y_begin:(y_begin+y_dim)]=image_array
        # write image       
        img = sitk.GetImageFromArray(labelRecover)   
        img.SetOrigin(ImageInfo['origin'])    
        img.SetSpacing(ImageInfo['spacing'])        
        img.SetDirection(ImageInfo['direction'])
        
        name, ext = os.path.splitext(predictlist[i])
        baseName = os.path.basename(name)
        fn = outputdir + '/predictRecover_' + baseName + '.nrrd'
        sitk.WriteImage(img,fn)
        outputList.append(fn) 
        print 'Iteration Num %d is finished'% i        
    del image,image_array
    return outputList        

#%%
def main():
    regIfLinear = 'Linear'
    
    # information of central point and ImageInfo
    CWD = '/home/louis/Documents/program_work/Python/MAS_visceral_ANT_2018_1_21'
    central_point = np.load(os.path.join(CWD,'PatchedImage_v2', organ + '_'+ regIfLinear+'_central_point.npy'))
    ImageInfo = np.load(os.path.join(CWD, 'PreProcessing_v5','ImageInfo.npy'))
    ImageInfo = ImageInfo.item()
    
    root = '/media/data/louis/ProgramWorkResult/ViscercialDNN_new_LOO'
    rawPredictDir = os.path.join(root, 'Pred3DMod', organ + FileType)
    rawPredictlist = readTxtIntoList(rawPredictDir + "/FileList.txt")
    
    PredictDir = os.path.join(root, "InverseTrans", "PredictRecovery", organ + FileType)
    if not os.path.exists(PredictDir):
        subprocess.call('mkdir ' + '-p ' + PredictDir, shell=True)
       
    predict_List = predictRecover(rawPredictlist, central_point, ImageInfo, PredictDir)
    WriteListtoFile(predict_List, PredictDir + "/FileList.txt")
    
    #upsample
    resultDir = os.path.join(root, "InverseTrans", "UpsampleResult", organ + FileType)      
    predictList = readTxtIntoList(PredictDir + "/FileList.txt")
    predictOutput = os.path.join(resultDir, "Predict")    
    if not os.path.exists(predictOutput):
        subprocess.call('mkdir ' + '-p ' + predictOutput, shell=True) 
    
    samplefactor=[2,2,2]
    InterP = 0 
    zoomPredictList = imagezoom(predictList, predictOutput, samplefactor, InterP)
    WriteListtoFile(zoomPredictList, predictOutput + "/FileList.txt")

if __name__ == '__main__':
    main()