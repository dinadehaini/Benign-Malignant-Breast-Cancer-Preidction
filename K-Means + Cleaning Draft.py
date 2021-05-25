#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import numpy.matlib
import pandas as pd
import numpy.linalg as la
get_ipython().run_line_magic('matplotlib', 'inline')


# In[8]:


df = pd.read_csv('breast-cancer-wisconsin.data.csv',header=None)
df.rename(columns={0:'id',1:'clump thickness',2:'uniformity of cellsize',
           3:'uniformity of cell shape',4:'marginal adhesion', 
           5:'single epithelial cell size',6:'bare nuclei', 
           7:'bland chromatin',8:'normal nucleoli',9:'mitosis',10:'class'},inplace=True)

'''Sample code number: id number
Clump Thickness: 1 - 10
Uniformity of Cell Size: 1 - 10
Uniformity of Cell Shape: 1 - 10
Marginal Adhesion: 1 - 10
Single Epithelial Cell Size: 1 - 10
Bare Nuclei: 1 - 10
Bland Chromatin: 1 - 10
Normal Nucleoli: 1 - 10
Mitoses: 1 - 10
Class: (2 for benign, 4 for malignant)'''


# In[9]:


df.isnull().any().any()


# no null values

# In[10]:


df['class'] = df['class'].replace(to_replace = 2, value = 0) #benign
df['class'] = df['class'].replace(to_replace = 4, value = 1) #malignant


# In[11]:


df['bare nuclei'] = df['bare nuclei'].replace(to_replace = '?', value = None)


# In[12]:


df['bare nuclei'] = df['bare nuclei'].astype('int64')


# In[13]:


df = df.drop(columns=['id'])
df


# # K-Means

# In[14]:


def plotCurrent(X, Rnk, Kmus):
    N, D = np.shape(X)
    K = np.shape(Kmus)[0]
    InitColorMat = np.matrix([[1, 0, 0], 
                              [0, 1, 0],   
                              [0, 0, 1],
                              [0, 0, 0],
                              [1, 1, 0], 
                              [1, 0, 1], 
                              [0, 1, 1]])
    KColorMat = InitColorMat[0:K]
    colorVec = Rnk.dot(KColorMat)
    muColorVec = np.eye(K).dot(KColorMat)
    plt.scatter(X[:,3], X[:,9], edgecolors=colorVec, marker='o', facecolors='none', alpha=0.3)
    plt.scatter(Kmus[:,0], Kmus[:,1], c=muColorVec, marker='D', s=50);


# In[15]:


def calcSqDistances(X, Kmus): #NEED TO CALCULATE MULTIDIMENSIONAL DISTANCE
    N = np.shape(X)[0]
    K = Kmus.shape[0]
    D = [] #initialize to be NxK shape np.zeros()
    for point in X:
        for kpoint in Kmus:
            #D.append([(la.norm(point - kpoint))**2])
            #D.append([(point - kpoint)**2])
            #print(type(point),type(kpoint))
            #print(point, kpoint)
            D.append(np.sqrt(np.sum((point - kpoint)**2)))
    D = np.array(D)
    D = D.reshape((N, K))
                     
    return D


# In[16]:


def determineRnk(sqDmat):
    m,n = sqDmat.shape
    for arr in sqDmat:
        index = np.argmin(arr, axis = None, out = None)
        for i in range(n):
            if not i==index:
                arr[i] = 0
        arr[index] = 1
    return sqDmat


# In[17]:


def recalcMus(X, Rnk):
    return (np.divide(X.T.dot(Rnk), np.sum(Rnk,axis=0))).T


# In[18]:


def runKMeans(K, data):
    X = data.to_numpy()
    N = np.shape(X)[0]
    D = np.shape(X)[1]
    Kmus = np.zeros((K, D))
    rndinds = np.random.permutation(N)
    Kmus = X[rndinds[:K]];
    maxiters = 1000;
    for iter in range(maxiters):
        sqDmat = calcSqDistances(X, Kmus);
        Rnk = determineRnk(sqDmat)
        KmusOld = Kmus
        plotCurrent(X, Rnk, Kmus)
        plt.show()
        Kmus = recalcMus(X, Rnk)
        if sum(abs(KmusOld.flatten() - Kmus.flatten())) < 1e-6:
            break
    plotCurrent(X,Rnk,Kmus)


# In[19]:


runKMeans(2, df)


# In[20]:


from sklearn.cluster import KMeans
clusters = 2
  
kmeans = KMeans(n_clusters = clusters)
kmeans.fit(df)
  
print(kmeans.labels_)


# In[21]:


from sklearn.decomposition import PCA
  
pca = PCA(3)
pca.fit(df)
  
pca_data = pd.DataFrame(pca.transform(df))
  
print(pca_data.head())


# In[22]:


from matplotlib import colors as mcolors
import math
clusters = 2   
''' Generating different colors in ascending order 
                                of their hsv values '''
colors = list(zip(*sorted((
                    tuple(mcolors.rgb_to_hsv(
                          mcolors.to_rgba(color)[:3])), name)
                     for name, color in dict(
                            mcolors.BASE_COLORS, **mcolors.CSS4_COLORS
                                                      ).items())))[1]
   
   
# number of steps to taken generate n(clusters) colors 
skips = math.floor(len(colors[5 : -5])/clusters)
cluster_colors = colors[5 : -5 : skips]


# In[23]:


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
   
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.scatter(pca_data[0], pca_data[1], pca_data[2], 
           c = list(map(lambda label : cluster_colors[label],
                                            kmeans.labels_)))
   
str_labels = list(map(lambda label:'% s' % label, kmeans.labels_))
   
list(map(lambda data1, data2, data3, str_label:
        ax.text(data1, data2, data3, s = str_label, size = 0.6,
        zorder = 20, color = 'k'), pca_data[0], pca_data[1],
        pca_data[2], str_labels))
   
plt.show()


# In[ ]:




