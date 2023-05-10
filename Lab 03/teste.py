import matplotlib.pyplot as plt
import joblib

#Parametros
url_predictor='data/trickbag'

#Setup
#Carregando o predictor e separando suas funções:
trickbag = joblib.load(open(url_predictor, 'rb'))
pca = trickbag.named_steps['pca']
kmeans = trickbag.named_steps['kmeans']
features_teste = trickbag.named_steps['features_teste']
labels_teste = trickbag.named_steps['labels_teste']
#Fim Setup

#Treinamento e criação de Clusters
y_pred = kmeans.predict(features_teste)

#Treinamento e transformação de x em PCA
x_pca = pca.transform(features_teste)

#Montando o plot e definindo os Centroids:
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
plt.show()