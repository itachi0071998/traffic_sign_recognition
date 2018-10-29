
import pandas as pd
import numpy as np
from utils import *
from radiomics import featureextractor



circles = [0,1,2,3,4,5,6,7,8,9,10,15,16,17,32,33,34,35,36,37,38,39,40,41,42]
triangles = [11,13,18,19,20,21,22,23,24,25,26,27,28,29,30,31]



circle = pd.read_csv("featuredataset_circle.csv")
triangle = pd.read_csv("featuredataset_triangle.csv")



circle = np.asarray(circle)
np.argwhere(np.isnan(circle))
print(circle[1619,141])
print(circle[10527,141])
print(circle[16356, 227])
circle = np.delete(circle, [1619,10527,16356], 0)
np.argwhere(np.isnan(circle))

x = circle[:, :798]
y = circle[:, 798:]
from sklearn.preprocessing import StandardScaler
min_max = StandardScaler()
x = min_max.fit_transform(x)


from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(np.asarray(x), np.asarray(y), stratify = y, test_size=0.2, random_state=42)



from sklearn import svm



model = svm.SVC()
model.fit(x_train, y_train) 




image = cv2.imread("12088.ppm")


path = "GTSRB/Final_Training/Images/"
params = 'exampleCT.yaml'
stat = True
extractor = featureextractor.RadiomicsFeaturesExtractor(params)
col_image_patch2 = image.copy()
bin_image_patch2 =np.ones(image.shape[:2])



FEATUREDATAVECTOR,FEATURENAMES = doCellFeatExtract(col_image_patch2, bin_image_patch2, extractor,stat)
FEATUREDATAVECTOR=np.asarray(FEATUREDATAVECTOR[2:])



test = FEATUREDATAVECTOR


delcol = [0,1,2,5,6,7,8,12,13,14,17,18]



test = np.delete(test, delcol, 0)



test = np.asarray(test)
test = np.reshape(test, (1,798))




a = model.predict(test)




from sklearn.metrics import accuracy_score




predict = model.predict(x_val)
print(accuracy_score(y_val , predict))


from sklearn.externals import joblib
joblib.dump(model, 'circle_model.joblib') 



triangle = np.asarray(triangle)
np.argwhere(np.isnan(triangle))




x = triangle[:, :798]
y = triangle[:, 798:]
from sklearn.preprocessing import StandardScaler
min_max = StandardScaler()
x = min_max.fit_transform(x)


from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(np.asarray(x), np.asarray(y), stratify = y, test_size=0.2, random_state=42)



model2 = svm.SVC()
model2.fit(x_train, y_train) 




predict = model.predict(x_val)
print(accuracy_score(y_val , predict))



from sklearn.externals import joblib
joblib.dump(model, 'triangle_model.joblib') 

