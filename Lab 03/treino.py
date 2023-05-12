import pandas as pd
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import joblib
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

#||||||||| SETUP ||||||||||#
url_predictor = 'data/Preditor'
data1 = 'data/optdigits.tra'
data2 = 'data/optdigits.tes'
proporcao_treino_teste = 0.33
rdm_state = 42 #Max 42
clusters = 10
max_iter = 300 #Default : 300
#########FIM SETUP##########

#Load Dataset, Merge, Shuffle and Split (Treino / Teste)
dataset = pd.concat([pd.read_csv(data1, sep = ',', header=None),
                     pd.read_csv(data2, sep=',', header=None)],
                    axis= 0, ignore_index=True)
features, features_teste, labels, labels_teste = train_test_split(dataset.iloc[:,:64],
                                                                  dataset.iloc[:,64:65],
                                                                  test_size=proporcao_treino_teste,
                                                                  random_state=rdm_state)

#-MODELO-#
trickbag = Pipeline([
    ('preprocessor', Pipeline(steps=[
        ('pca', PCA(n_components=2,
                random_state=rdm_state)),
        ('kmeans', KMeans(n_clusters= clusters,
                      init='k-means++',
                      random_state=rdm_state,
                      n_init= 'auto',
                      max_iter=max_iter)),])),
    ('features_teste', features_teste),
    ('labels_teste', labels_teste),
    ('plot_title', ('Disperssão de bolhas por KMeans\nTreino/Teste=%.0f%%/%.0f%%, Clusters=%d, Random State=%d, Max iter=%d'% (((1-proporcao_treino_teste)*100),((proporcao_treino_teste)*100),clusters,rdm_state,max_iter)))
])
preprocessor = trickbag.named_steps['preprocessor']
plot_title = trickbag.named_steps['plot_title']
#-FimModelo-#

#Treinamento e transformação de x em PCA
feat_pca = preprocessor.fit_transform(features)
#feat_pca = preprocessor.named_steps['pca'].fit_transform(features)

#Treinamento e criação de Clusters
#label_pred = preprocessor.named_steps['kmeans'].fit_predict(feat_pca)
label_pred = preprocessor.fit_predict(feat_pca)

#Apresentando o Score:
print('Score do modelo: ',preprocessor.score(feat_pca))

#Montando o plot e definindo os Centroids:
plt.scatter(x=feat_pca[:,0],
            y=feat_pca[:,1],
            c=label_pred)
centroids = preprocessor.named_steps['kmeans'].cluster_centers_
plt.scatter(centroids[:,0], centroids[:,1],
            marker='*',
            s=360,
            linewidths=1,
            color='green',
            edgecolors='black',)
plt.title(plot_title)
plt.show()
#Gravar o predictor.
predictor_file= open(url_predictor, 'wb')
joblib.dump(trickbag, predictor_file)
print('===== CHARACTER PREDICTOR DONE =====')