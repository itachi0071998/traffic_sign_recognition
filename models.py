import imutils
from radiomics import featureextractor
import cv2
import csv
import numpy as np
import SimpleITK as sitk
import six
import matplotlib.pyplot as plt
import pandas as pd

def pattern_matching_model(img, arr, arr2):
    test = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    test = cv2.resize(test, (30,30), interpolation = cv2.INTER_CUBIC)
    #print(test.shape)
    thres = 0
    index = 0
    for i in range(0,len(arr2)):

        a = np.asarray(arr[i])
        res = cv2.matchTemplate(test,a.astype(np.uint8),cv2.TM_CCOEFF_NORMED)
        #print(res)
        if res>= thres:
            thres = res
            index = i
    return arr2[index]


def rk_calc_model(A,arr,arr2):

    A = cv2.resize(A, (30,30), interpolation = cv2.INTER_CUBIC)
    A_gray = cv2.cvtColor(A, cv2.COLOR_BGR2GRAY)
    #print(A_gray.shape)
    #print(B_gray.shape)
    #A_gray=lantplace_of_gaussian(A_gray)
    #B_gray=laplace_of_gaussian(B_gray)
    temp = 0
    index = 0
    for x in range(0,len(arr2)):
        m=30
        n=30
        a_mean=0.0
        b_mean=0.0
        num_sum=0.0
        dsum1=0.0
        dsum2=0.0
        B_gray = np.asarray(arr[x])
        B_gray.astype(np.uint8)
        for i in range(0,m):
            for j in range(0,n):
                a_mean+=A_gray[i][j]
                b_mean+=B_gray[i][j]
        a_mean/=(m*n)
        b_mean/=(m*n)
        #print(a_mean)
        #print(b_mean)

        for i in range(0,m):
            for j in range(0,n):
                num_sum+=(A_gray[i][j]-a_mean)*(B_gray[i][j]-b_mean)
                dsum1+=(A_gray[i][j]-a_mean)*(A_gray[i][j]-a_mean)
                dsum2+=(B_gray[i][j]-b_mean)*(B_gray[i][j]-b_mean)
        #print(dsum1)
        #print(dsum2)

        d=math.sqrt(dsum1*dsum2)
        rk=num_sum/d
        if temp<=rk:
            temp = rk
            index = x
    return arr2[x]