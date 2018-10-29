import imutils
from radiomics import featureextractor
import cv2
import csv
import numpy as np
import SimpleITK as sitk
import six
import matplotlib.pyplot as plt
import pandas as pd
path = "GTSRB/Final_Training/Images/"
params = 'exampleCT.yaml'

extractor = featureextractor.RadiomicsFeaturesExtractor(params)

def doCellFeatExtract(cell_image, mask_image, extractor,stat):
    gray_image_patch = cv2.cvtColor(cell_image, cv2.COLOR_BGR2GRAY)
    im_arr = np.array(gray_image_patch);
    im_arr = np.expand_dims(im_arr, axis=0)
    image3d = sitk.GetImageFromArray(im_arr)
    im_arr = np.array(mask_image);
    im_arr = np.expand_dims(im_arr, axis=0)
    im_arr=np.array(im_arr,np.uint8)
    mask3d = sitk.GetImageFromArray(im_arr)
    result = extractor.execute(image3d, mask3d)
    cellFeat = []
    featName=[]
    count = 0

    for key, val in six.iteritems(result):
        #print("\t%s: %s" % (key, val))
        if count > 11:
            cellFeat.append(val)
            if (stat):
                featName.append(key)

        count = count+1

    #print(featName)
    return (cellFeat,featName)




def feature_extractor(shape):
    labels =[['labels']]
    stat= True
    for i in shape:
        j = 1
        if i<=9:
            extrapath = "0000"+str(i)
            #ep=path+extrapath
            finalpath = path+extrapath+"/GT-0000"+str(i)+".csv"
            file = pd.read_csv(finalpath)
        else:
            extrapath = "000"+str(i)
            finalpath = path+extrapath+"/GT-000"+str(i)+".csv"
            file = pd.read_csv(finalpath)

        temp=file
        temp=np.asarray(temp)

        for (name,h,w,x1,y1,x2,y2,l) in temp:
            #,file["Roi.X1"],file["Roi.X2"],file["Roi.Y1"],file["Roi.Y2"]
            labels.append(l)
            #print(name)
            tt = path+extrapath+"/"+name
            #print(tt)
            image = cv2.imread(tt)
            col_image_patch2 = image.copy()
            bin_image_patch2 =np.zeros(image.shape[:2])
            bin_image_patch2[x1:x2,y1:y2]=1
            #plt.imshow(bin_image_patch2)
            #plt.show()
            #print(bin_image_patch2)


            FEATUREDATAVECTOR,FEATURENAMES = doCellFeatExtract(col_image_patch2, bin_image_patch2, extractor,stat)
            #print(FEATUREDATAVECTOR)
            #FEATUREDATAVECTOR=np.append([label],FEATUREDATAVECTOR,axis=0)
            FEATUREDATAVECTOR=np.asarray(FEATUREDATAVECTOR[2:])
            #print((FEATUREDATAVECTOR))
            #print(FEATUREDATAVECTOR.shape)
            if (stat):
                #FEATURENAMES = np.append(['Feature Name'], FEATURENAMES, axis=0)
                a=np.array([FEATURENAMES[2:]])
                #a=a[9:]
                #print(a)
                #print(a.shape)
                a = np.append(a,[FEATUREDATAVECTOR],axis=0)
                stat = False
            else:
                a = np.append(a, [FEATUREDATAVECTOR], axis=0)
            j+=1


    return a, labels

def mean_calc(shape, temp):
    mean = []
    for i in shape:
        count = 0
        a = np.zeros((798))
        for j in range(0, len(temp)):
            if temp[j,798] == i:
                a = a+ temp[j,:798]
                count += 1
        a = a/count
        mean.append(a)
    return mean


def detectShape(c):
    shape = 'unknown'
    # calculate perimeter using
    peri = cv2.arcLength(c, True)
    # apply contour approximation and store the result in vertices
    approx = cv2.approxPolyDP(c, 0.04 * peri, True)

    # If the shape it triangle, it will have 3 vertices
    if len(approx) in range(3,5):
        shape = 'triangle'

    # the shape is a circle
    else :
        shape = "circle"

    # return the name of the shape
    return shape,approx

def suitable_index(lth, th, arr):
    count = 0
    index = []
    for i in range(0, len(arr)):
        if lth<= arr[i] <=th:
            count += 1
            index.append(i)
    return index, count

def stdev_calc(shape, temp):
    arr = np.arange(798)
    arr = np.reshape(arr, (1,798))
    for f in shape:
        final_dev = []
        stdev = []
        for i in range(0, len(temp)):
            if temp[i,798] == f:
                stdev.append(temp[i,:798])
        dev = np.asarray(stdev)
        for j in range(0,798):
            s = statistics.stdev(dev[: , j])
            final_dev.append(s)
        d = np.asarray(final_dev)
        d = np.reshape(d, (1,798))
        arr = np.concatenate((arr, d), axis = 0)
    stdv = arr[1:,:]
    return stdv

def simple_calc(arr,mode = None):
    if mode == "mean":
        mean_stdev = np.zeros((1,arr.shape[1]))
        for i in range(0,arr.shape[0]):
            mean_stdev = mean_stdev + arr[i]
        return mean_stdev/len(arr)
    elif mode == "std":
        stdev_mean = []
        for i in range(0,arr.shape[1]):
            stdev_mean.append(statistics.stdev(arr[:, i]))
        stdev_mean = np.reshape(stdev_mean, (1,arr.shape[1]))
        return stdev_mean
    else:
        return "Nothing to calculate please add mode = 'mean' for calculating mean or mode = 'std' for calculating std in the function"
    
    
   
