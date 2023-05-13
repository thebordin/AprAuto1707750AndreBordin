import matplotlib.pyplot as plt
import joblib

#Parametros
url_predictor='data/Preditor'

#Setup
#Carregando o predictor e separando suas funções:
trickbag = joblib.load(open(url_predictor, 'rb'))
trick = trickbag['trick']
features_teste = trickbag['features_teste']
labels_teste = trickbag['labels_teste']
plot_title = trickbag['plot_title']
#Fim Setup

#Transformação de x em PCA
feat_pca = trick['pca'].transform(features_teste)

#Predição de Clusters
y_pred = trick.predict(features_teste)

#Apresentando o Score:
print('Score do modelo: ',trick.score(features_teste))

#Montando o plot e definindo os Centroids:
plt.scatter(x=feat_pca[:,0],
            y=feat_pca[:,1],
            c=y_pred)
centroids = trick['kmeans'].cluster_centers_
plt.scatter(centroids[:,0], centroids[:,1],
            marker='*',
            s=360,
            linewidths=1,
            color='green',
            edgecolors='black',)
plt.title(plot_title)
plt.show()