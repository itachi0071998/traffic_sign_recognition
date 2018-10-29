circles = [0,1,2,3,4,5,6,7,8,9,10,15,16,17,32,33,34,35,36,37,38,39,40,41,42]
triangles = [11,13,18,19,20,21,22,23,24,25,26,27,28,29,30,31]
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

file1 = pd.read_csv("featuredataset_triangle.csv")
file = pd.read_csv("featuredataset_circle.csv")

arrst=stdev_calc(circles,np.asarray(file))
arrstMean=simple_calc(arrst,mode='mean')
a,c=suitable_index(0,5,arrstMean.T)
a1,c1=suitable_index(5,9999999999999999999999999999999,arrmnst.T)

aint=list()
for i in a:
    if i in a1:
        aint.append(i)
        
print(len(aint))


arrst=stdev_calc(triangles,np.asarray(file1))
arrstMean=simple_calc(arrst,mode='mean')
a,c=suitable_index(0,5,arrstMean.T)
a1,c1=suitable_index(5,9999999999999999999999999999999,arrmnst.T)

aint=list()
for i in a:
    if i in a1:
        aint.append(i)
        
print(len(aint))