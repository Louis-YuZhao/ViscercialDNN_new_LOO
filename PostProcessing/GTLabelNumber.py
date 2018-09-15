# -*- coding: utf-8 -*-

# 09/06/2017

#%%

import SimpleITK as sitk
import os
import string
import subprocess

#%%
def readTxtIntoList(filename):
    flist = []
    with open(filename) as f:
        flist = f.read().splitlines()
    return flist

def labelNumbering(labeldir, outputdir, labelNo, threshold = 10**(-2)):

    image = sitk.ReadImage(labeldir)
    im_arr_mod = sitk.GetArrayFromImage(image) # get numpy array    
    
    im_arr_mod[im_arr_mod > threshold] = labelNo 
    im_arr_mod[im_arr_mod < threshold] = 0        
    img = sitk.GetImageFromArray(im_arr_mod)   
    img.SetOrigin(image.GetOrigin())    
    img.SetSpacing(image.GetSpacing())        
    img.SetDirection(image.GetDirection())

    fn = outputdir + '_labelNo_' + str(labelNo) + '.nrrd'
    sitk.WriteImage(img,fn)
    return fn 
    
def labelfusion(inputlist, outputdir):
    num=len(inputlist)    
    image = sitk.ReadImage(inputlist[0])
    finallabel = sitk.GetArrayFromImage(image) # get numpy array 
        
    for i in range(1,num): 
        im = sitk.ReadImage(inputlist[i])
        finallabel += sitk.GetArrayFromImage(im) # get numpy array
        print 'Num %d is finished'% i
    # write image       
    img = sitk.GetImageFromArray(finallabel)   
    img.SetOrigin(image.GetOrigin())    
    img.SetSpacing(image.GetSpacing())        
    img.SetDirection(image.GetDirection())

    fn = outputdir + '_finallabel'+'.nrrd'
    sitk.WriteImage(img,fn)        
    del image, finallabel
           

def ImageID(InputList):
    ImageID = []
    for item in InputList:
        name, ext = os.path.splitext(item)
        BaseName = os.path.basename(name)
        testIndicator = string.join(BaseName.split("_")[-6:-2], "_")
        ImageID.append(testIndicator)
    return ImageID
#%%
regIfLinear = 'Modification_Regtrans_Linear_ANT'
#regIfLinear = 'Regtrans_nonlinear_ANT'
organList = ["170_pancreas","187_gallbladder","30325_left_adrenal_gland","30324_right_adrenal_gland"]
organNumList = [1,2,4,5]
rootDir = '/media/data/louis/ProgramWorkResult/VisercialMAS_ANT/Full_Image/GC_label_adjustment/'

output_root_dir = '/media/data/louis/ProgramWorkResult/VisercialMAS_ANT'+'/MultiOrgan' + '_GT' + '/'
if not os.path.exists(output_root_dir):
    subprocess.call('mkdir ' + '-p ' + output_root_dir, shell=True)   

InputFolderList = []
ImageIDDict = {}
ImagePathsDict = {}
for item in organList:
    currentFile = rootDir + item + '/' + regIfLinear
    InputFolderList.append(currentFile)
    currentlist = readTxtIntoList(currentFile + "/FileList.txt")
    ImagePathsDict[item]=currentlist
    ImageIDDict[item]=ImageID(currentlist)

intersection = ImageIDDict[organList[0]]
for i in xrange(1,len(organList)):
    intersection = list(set(intersection).intersection(set(ImageIDDict[organList[i]])))

for item in intersection:
    
    predictDirlist = []
    for i in xrange(len(organList)):    
        organ = organList[i]
        labelNo = organNumList[i]
        filelist = ImagePathsDict[organ]
        for f in filelist:
            if item in f:
                labelin = f        
                labelout = output_root_dir + organ + "_" + item        
                currentFile = labelNumbering(labelin, labelout, labelNo, threshold = 10**(-2))
                predictDirlist.append(currentFile)
    resultDir = output_root_dir+'/GT_'+ item
    labelfusion(predictDirlist, resultDir)