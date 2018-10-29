
from sklearn.externals import joblib
from utils import *
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
params = 'exampleCT.yaml'
#print(params)
extractor = featureextractor.RadiomicsFeaturesExtractor(params)
stat = True

test_image = cv2.imread("example.ppm")
shape = detectShape(test)

# way 1 using svm model finidng traffic sign
col_image_patch2 = image.copy()
bin_image_patch2 =np.ones(image.shape[:2])
FEATUREDATAVECTOR,FEATURENAMES = doCellFeatExtract(col_image_patch2, bin_image_patch2, extractor,stat)
test=np.asarray(FEATUREDATAVECTOR[2:])
min_max = StandardScaler()
x = min_max.fit_transform(test)
if shape == "circle":
    model = joblib.load('circle_model.joblib')
    prediction = model.predict(x)
if shape == "triangle":
    model = joblib.load('triangle_model.joblib')
    prediction = model.predict(test)
print("The Traffic signal is: ", prediction)

# way 2 using patteren matching finding traffic sign
circles = [0,1,2,3,4,5,6,7,8,9,10,15,16,17,32,33,34,35,36,37,38,39,40,41,42]
triangles = [11,13,18,19,20,21,22,23,24,25,26,27,28,29,30,31]
from models import *
templates_circles = []
for i in circles:
    j = 1
    if i<=9:
        extrapath = "0000"+str(i)
        #ep=path+extrapath
        finalpath = path+extrapath+"/GT-0000"+str(i)+".csv"
        file = pd.read_csv(finalpath, sep = ";")
    else:
        extrapath = "000"+str(i)
        finalpath = path+extrapath+"/GT-000"+str(i)+".csv"
        file = pd.read_csv(finalpath, sep = ";")
    temp=file
    temp=np.asarray(temp)
    temp22 = np.zeros((30,30))
    for (name,h,w,x1,y1,x2,y2,l) in temp:
        tt = path+extrapath+"/"+name
        image = cv2.imread(tt, 0)
        image = image[x1:x2,y1:y2]
        image = cv2.resize(image, (30,30), interpolation = cv2.INTER_CUBIC)
        temp22 = temp22 + image
        
    templates_circles.append(temp22/len(temp))
    
templates_triangles = []
for i in triangles:
    j = 1
    if i<=9:
        extrapath = "0000"+str(i)
        #ep=path+extrapath
        finalpath = path+extrapath+"/GT-0000"+str(i)+".csv"
        file = pd.read_csv(finalpath, sep=";")
    else:
        extrapath = "000"+str(i)
        finalpath = path+extrapath+"/GT-000"+str(i)+".csv"
        file = pd.read_csv(finalpath, sep = ";")
    temp=file
    temp=np.asarray(temp)
    temp22 = np.zeros((30,30))
    for (name,h,w,x1,y1,x2,y2,l) in temp:
        tt = path+extrapath+"/"+name
        #print(tt)
        image = cv2.imread(tt, 0)
        image = image[x1:x2,y1:y2]
        image = cv2.resize(image, (30,30), interpolation = cv2.INTER_CUBIC)
        temp22 = temp22 + image
        
    templates_triangles.append(temp22/len(temp))
    
if shape == "circle":
    prediction = model(test_image, templates_circles, circles)
if shape == "triangle":
    prediction = model(test_image, templates_triangles, triangles)
print("The Traffic signal is: ", prediction)



#way 3 predicting using corelation coefficient
corelation_matrix = []
for i in range(0,43):
    j = 1
    if i<=9:
        extrapath = "0000"+str(i)
        #ep=path+extrapath
        finalpath = path+extrapath+"/GT-0000"+str(i)+".csv"
        file = pd.read_csv(finalpath, sep = ";")
    else:
        extrapath = "000"+str(i)
        finalpath = path+extrapath+"/GT-000"+str(i)+".csv"
        file = pd.read_csv(finalpath, sep = ";")
    temp=file
    temp=np.asarray(temp)
    temp22 = np.zeros((30,30))
    for (name,h,w,x1,y1,x2,y2,l) in temp:
        tt = path+extrapath+"/"+name
        image = cv2.imread(tt, 0)
        image = image[x1:x2,y1:y2]
        image = cv2.resize(image, (30,30), interpolation = cv2.INTER_CUBIC)
        temp22 = temp22 + image
        
    corelation_matrix.append(temp22/len(temp))
arr = np.arange(43)
prediction = rk_calc_model(test_image, corelation_matrix, arr)
print("The Traffic signal is: ", prediction)

