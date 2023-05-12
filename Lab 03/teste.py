import matplotlib.pyplot as plt
import joblib

#Parametros
url_predictor='data/Preditor'

#Setup
#Carregando o predictor e separando suas funções:
trickbag = joblib.load(open(url_predictor, 'rb'))
pca = trickbag.named_steps['pca']
kmeans = trickbag.named_steps['kmeans']
features_teste = trickbag.named_steps['features_teste']
labels_teste = trickbag.named_steps['labels_teste']
plot_title = trickbag.named_steps['plot_title']
#Fim Setup

#Treinamento e transformação de x em PCA
feat_pca = pca.transform(features_teste)

#Predição de Clusters
y_pred = kmeans.predict(feat_pca)

#Apresentando o Score:
print('Score do modelo: ',kmeans.score(feat_pca))

#Montando o plot e definindo os Centroids:
plt.scatter(x=feat_pca[:,0],
            y=feat_pca[:,1],
            c=y_pred)
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:,0], centroids[:,1],
            marker='*',
            s=360,
            linewidths=1,
            color='green',
            edgecolors='black',)
plt.title(plot_title)
plt.show()