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
preprocessor = PCA(n_components=2,random_state=rdm_state)
sheepherd = KMeans(n_clusters= clusters,
                      init='k-means++',
                      random_state=rdm_state,
                      n_init= 'auto',
                      max_iter=max_iter)

trickbag = Pipeline([
    ('trick', Pipeline(steps=[('pca', preprocessor),('kmeans', sheepherd)])),
    ('features_teste', features_teste),
    ('plot_title', ('Disperssão de bolhas por KMeans\nTreino/Teste=%.0f%%/%.0f%%\nClusters=%d, Random State=%d, Max iter=%d'% (((1-proporcao_treino_teste)*100),((proporcao_treino_teste)*100),clusters,rdm_state,max_iter)))
])
trick = trickbag['trick']
plot_title = trickbag['plot_title']
#-FimModelo-#

#Treinamento e transformação de x com PCA (OU PCA/KMEANS)
trick.fit(features)
feat_pca_pipe = trick.transform(features)
feat_pca = trick['pca'].transform(features)
print('PIPE: \n',feat_pca_pipe[0:1],'\nSTEPS:\n', feat_pca[0:1]) #>>>>>>>>>> PORQUE O PIPELINE INTEIRO GERA RESULTADO DIFERENTE ?

feat_pca_kmeans = trick['kmeans'].transform(feat_pca)

print('PIPE: \n',feat_pca_pipe[0:1],'\nSTEPS:\n', feat_pca_kmeans[0:1]) #>>>>>>>>>> PORQUE ELE PASSA POR 2 TRANSFORMACOES : PCA E KMEANS !!!!!

#Treinamento e criação de Clusters
label_pred_pipe = trick.predict(features)
label_pred = trick['kmeans'].predict(feat_pca)
#Apresentando o Score:
print('Score do modelo PIPE: ',trick.score(features))
print('Score do modelo KMEANS: ',trick['kmeans'].score(feat_pca))

#Montando o plot e definindo os Centroids:
plt.figure(figsize=(8,8))
plt.subplot(221)
plt.scatter(x=feat_pca_pipe[:,0], #<<<<<<<<<<<<<< DADOS ERRADOS.... ELE SÓ USA 2/10 COLUNAS
            y=feat_pca_pipe[:,1],
            c=label_pred)
plt.title(plot_title +'\n!!!!!!WRONG FULL PIPE!!!!!!')
plt.subplot(224)
plt.scatter(x=feat_pca[:,0],
            y=feat_pca[:,1],
            c=label_pred)
centroids = trick['kmeans'].cluster_centers_
plt.scatter(centroids[:,0], centroids[:,1],
            marker='*',
            s=360,
            linewidths=1,
            color='green',
            edgecolors='black',)
plt.title(plot_title+'\nSTEPS')
plt.show()

#Gravar o predictor.
predictor_file= open(url_predictor, 'wb')
joblib.dump(trickbag, predictor_file)
print('===== CHARACTER PREDICTOR DONE =====')