import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import joblib
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
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
features, features_teste, labels, labels_teste = train_test_split(dataset.iloc[:,:64],dataset.iloc[:,64:65], test_size=0.33, random_state=42)


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

#Treinamento e transformação de x em PCA
x_pca = pca.fit_transform(features)

#Score

#Simulando Nº errados de clusters com K calculado por KMeans :
plt.figure(figsize=(10,10))
kmw = KMeans(n_clusters= 2, random_state=rdm_state, n_init= 'auto',max_iter=max_iter)
y_pred_wrong_k = kmw.fit_predict(x_pca)
plt.subplot(221)
plt.scatter(x=x_pca[:,0],
            y=x_pca[:,1],
            c=y_pred_wrong_k)
centroids = kmw.cluster_centers_
plt.scatter(centroids[:,0], centroids[:,1],
            marker='*',
            s=360,
            linewidths=1,
            color='green',
            edgecolors='black',)
plt.title('Numero incorreto de clusters.(2)')

#Plotando distribuição Anisotrópica
#Ajustando o modelo
transformation = [[0.60834549,-0.63667341],[-0.40887718,0.85253229]]
x_pca_aniso = np.dot(x_pca, transformation)
y_pred_aniso = kmeans.fit_predict(x_pca_aniso)
#plotando
plt.subplot(222)
plt.scatter(x=x_pca_aniso[: , 0],
            y=x_pca_aniso[:, 1],
            c=y_pred_aniso,)
centroids = kmeans.cluster_centers_
plt.scatter(x=centroids[:, 0],
            y=centroids[:, 1],
            marker='*',
            s=360,
            color='green',)
plt.title('Distribuição anisotrópica da Inferência.')

plt.subplot(223)
y_pred = kmeans.fit_predict(x_pca)
plt.scatter(x=x_pca[:,0],
            y=x_pca[:,1],
            c=y_pred)
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:,0], centroids[:,1],
            marker='*',
            s=360,
            linewidths=1,
            color='green',
            edgecolors='black',)
plt.title('Disperssão de bolhas por KMeans')
#
plt.show()
