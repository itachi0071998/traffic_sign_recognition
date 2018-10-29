
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[28]:


from utils import *
from radiomics import featureextractor


# In[3]:


circles = [0,1,2,3,4,5,6,7,8,9,10,15,16,17,32,33,34,35,36,37,38,39,40,41,42]
triangles = [11,13,18,19,20,21,22,23,24,25,26,27,28,29,30,31]


# In[4]:


circle = pd.read_csv("featuredataset_circle.csv")
triangle = pd.read_csv("featuredataset_triangle.csv")


# In[5]:


circle = np.asarray(circle)
np.argwhere(np.isnan(circle))
print(circle[1619,141])
print(circle[10527,141])
print(circle[16356, 227])
circle = np.delete(circle, [1619,10527,16356], 0)
np.argwhere(np.isnan(circle))
#print(x[1619,141])
##print(x[10527,141])
#print(x[16356, 227])


# In[6]:


x = circle[:, :798]
y = circle[:, 798:]
from sklearn.preprocessing import StandardScaler
min_max = StandardScaler()
x = min_max.fit_transform(x)
x.shape


# In[7]:


#from keras import utils
#y = utils.to_categorical(y)
y.shape


# In[8]:


from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(np.asarray(x), np.asarray(y), stratify = y, test_size=0.2, random_state=42)


# In[9]:


from sklearn import svm


# In[10]:


model = svm.SVC()
model.fit(x_train, y_train) 


# In[34]:


image = cv2.imread("12088.ppm")


# In[35]:


path = "GTSRB/Final_Training/Images/"
params = 'exampleCT.yaml'
#print(params)
stat = True
extractor = featureextractor.RadiomicsFeaturesExtractor(params)
col_image_patch2 = image.copy()
bin_image_patch2 =np.ones(image.shape[:2])


# In[36]:


FEATUREDATAVECTOR,FEATURENAMES = doCellFeatExtract(col_image_patch2, bin_image_patch2, extractor,stat)
#print(FEATUREDATAVECTOR)
#FEATUREDATAVECTOR=np.append([label],FEATUREDATAVECTOR,axis=0)
FEATUREDATAVECTOR=np.asarray(FEATUREDATAVECTOR[2:])


# In[37]:


test = FEATUREDATAVECTOR
len(test)


# In[38]:


delcol = [0,1,2,5,6,7,8,12,13,14,17,18]


# In[39]:


test = np.delete(test, delcol, 0)


# In[46]:


test = np.asarray(test)
test = np.reshape(test, (1,798))


# In[47]:


a = model.predict(test)


# In[48]:


a


# In[50]:


from sklearn.metrics import accuracy_score


# In[51]:


predict = model.predict(x_val)
print(accuracy_score(y_val , predict))


# In[52]:


from sklearn.externals import joblib
joblib.dump(model, 'circle_model.joblib') 


# In[53]:


triangle = np.asarray(triangle)
np.argwhere(np.isnan(triangle))


# In[54]:


x = triangle[:, :798]
y = triangle[:, 798:]
from sklearn.preprocessing import StandardScaler
min_max = StandardScaler()
x = min_max.fit_transform(x)
x.shape


# In[55]:


from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(np.asarray(x), np.asarray(y), stratify = y, test_size=0.2, random_state=42)


# In[59]:


model2 = svm.SVC()
model2.fit(x_train, y_train) 


# In[60]:


predict = model.predict(x_val)
print(accuracy_score(y_val , predict))


# In[61]:


from sklearn.externals import joblib
joblib.dump(model, 'triangle_model.joblib') 

