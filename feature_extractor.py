from utils import *
import numpy as np
import csv
circles = [0,1,2,3,4,5,6,7,8,9,10,15,16,17,32,33,34,35,36,37,38,39,40,41,42]


triangles = [11,13,18,19,20,21,22,23,24,25,26,27,28,29,30,31]


a, labels = feature_extractor(circles)
labels = np.asarray(labels)
labels = np.reshape(labels, (len(labels),1))

a = np.concatenate((a, labels),axis = 1)


with open("featuredataset_circle.csv", "w+") as output:
    writer = csv.writer(output, lineterminator='\n')
    writer.writerows(a)

    
    
    

triangles = [11,13,18,19,20,21,22,23,24,25,26,27,28,29,30,31]
a, labels = feature_extractor(triangles)
labels = np.asarray(labels)
labels = np.reshape(labels, (len(labels),1))

a = np.concatenate((a, labels),axis = 1)
with open("featuredataset_triangle.csv", "w+") as output:
    writer = csv.writer(output, lineterminator='\n')
    writer.writerows(a)

