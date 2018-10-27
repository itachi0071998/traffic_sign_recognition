
# coding: utf-8

# In[1]:


from sklearn.externals import joblib
from utils import *
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# In[2]:


params = 'exampleCT.yaml'
#print(params)
extractor = featureextractor.RadiomicsFeaturesExtractor(params)
stat = True


# In[ ]:


test_image = cv2.imread("example.ppm")
shape = detectShape(test)
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

