import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.cluster import kmeans_plusplus

#Treinamento full
limite,url_predictor = 5620,'dataset/trickbag'
#teste
#limite,url_predictor = 500,'dataset/trickbag_useless'

#||||||||| SETUP ||||||||||# Espera-se score de 0.016
data1 = 'data/optdigits.tra'
data2 = 'data/optdigits.tes'
proporcao_treino_teste = 0.64
rdm_state = 170
clusters = 10
max_iter = 1000 #Default : 300
#########FIM SETUP##########

#Load Dataset, Merge, Shuffle and Split (Treino / Teste)
dataset = pd.concat([pd.read_csv(data1, sep = ',', header=None),
                     pd.read_csv(data2, sep=',', header=None)],
                    axis= 0, ignore_index=True)
dataset = dataset.reindex(np.random.permutation(dataset.index))
treino = dataset.iloc[:int(len(dataset)*proporcao_treino_teste), : ]
teste = dataset.iloc[((int(len(dataset)*proporcao_treino_teste))-len(dataset)): , :]
features = treino.iloc[:limite,0:64]
labels = treino.iloc[:limite,64:65]
features_teste = teste.iloc[:limite,0:64]
labels_teste = teste.iloc[:limite,64:65]

#-MODELO-#
trickbag = Pipeline([
    ('kmeans', KMeans(n_clusters= clusters,
                      init='k-means++',
                      random_state=rdm_state,
                      n_init= 'auto',
                      max_iter=max_iter)),
    ('pca', PCA(n_components=2,
                random_state=rdm_state)),
    ('features_teste', features_teste),
    ('labels_teste', labels_teste)
])
kmeans = trickbag.named_steps['kmeans']
pca = trickbag.named_steps['pca']
#-FimModelo-#

#Treinamento e criação de Clusters
y_pred = kmeans.fit_predict(features)
print(np.unique(y_pred))
#Treinamento e transformação de x em PCA
x_pca = pca.fit_transform(features)

#Primeiro plot:
plt.figure(figsize=(12,12))
plt.subplot(221)
plt.scatter(x=x_pca[:,0],
            y=x_pca[:,1],
            c=y_pred,
            marker='o',
            linewidths=1,
            edgecolors='black',)
centroids = kmeans.cluster_centers_
print(centroids)
plt.scatter(centroids[:,0], centroids[:,1],
            marker='x',
            s=90,
            linewidths=1,
            color='pink',
            edgecolors='black',)
plt.title('Numero incorreto de clusters.')
plt.show()
