import numpy as np
import pandas as pd
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import neighbors
import joblib
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.cluster import kmeans_plusplus

#Treinamento full
limite,url_predictor = 3823,'dataset/trickbag'
#teste
limite,url_predictor = 500,'dataset/trickbag_useless'

#||||||||| SETUP ||||||||||# Espera-se score de 0.016
localdata = 'data/optdigits.tra'
rdm_state = 170
clusters = 10
cmapa = plt.colormaps['BrBG']
#-MODELO-#
trickbag = Pipeline([
    ('kmeans', KMeans(n_clusters=clusters,init='k-means++', random_state=rdm_state)),
    ('pca', PCA(n_components=2,random_state=rdm_state)),
    ('cmapa', cmapa)
])
kmeans = trickbag.named_steps['kmeans']
pca = trickbag.named_steps['pca']
cmapa = trickbag.named_steps['cmapa']
#-FimModelo-#
#########FIM SETUP##########

#Load Dataset
dataset = pd.read_csv(localdata, sep = ',', header=None)
features = dataset.iloc[:limite,0:64]
labels = dataset.iloc[:limite,64:65]

#Treinamento e criação de Clusters
y_pred = kmeans.fit_predict(features)

#Treinamento e transformação de x em PCA
x_pca = pca.fit_transform(features)



