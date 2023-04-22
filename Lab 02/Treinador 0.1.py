import numpy as np
import pandas as pd
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import neighbors
import pickle as p1
from sklearn import linear_model

#Parametros
url_train='dataset/optdigits.tra'
url_test='dataset/optidigits.tes'
url_predictor='dataset/character_snca_predictor'
limite= 500 # 1797
n_neighbors = 12
grid = 2
pesos = ["uniform"] #"uniform" "distance"
cmapa=sns.color_palette('BrBG',n_colors=10, as_cmap=True)

#Setup
nca = make_pipeline(StandardScaler(), NeighborhoodComponentsAnalysis(n_components=2, random_state=None))

# Importar o conjunto de dados para Treino
dataset = pd.read_csv(url_train, sep=',', header=None)
xy_raw = dataset[:]

# Separar as variáveis dependentes e independentes
x_raw = xy_raw.iloc[:limite, 0:64]
y_raw = xy_raw.iloc[:limite, 64:65]
y = y_raw[64].values.tolist()

# Reduzir as dimensões das entradas
x_nca = nca.fit_transform(x_raw,y_raw)

# Define os agrupamentos e coloração do grafico após o treinamento
knc = neighbors.KNeighborsClassifier(n_neighbors, weights='uniform')
knc.fit(x_nca, y)

# Define pontos na malha para ... [x_min, x_max]x[y_min, y_max].
for weights in pesos:
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights=pesos)
    x_min, x_max = x_nca[:, 0].min() - 1, x_nca[:, 0].max() + 1
    y_min, y_max = x_nca[:, 1].min() - 1, x_nca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, grid), np.arange(y_min, y_max, grid))
    # ... Abrir , pintar e fechar ela.
    Z = knc.predict(np.c_[xx.ravel(), yy.ravel()]).astype(int)
    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, cmap=cmapa)
print(type(x_nca))
print('######################')
print(y)

# Plotagem
sns.scatterplot(
    x=x_nca[:, 0],
    y=x_nca[:, 1],
    marker='o',
    hue=y,
    palette=cmapa,
    alpha=1.0,
    edgecolor="black")
#Desenha
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("Classificação e previsão de caractere numérico \n "
          "(k = %i, weights = '%s')" % (n_neighbors, pesos))
plt.show()

#Regressão linear para treinar o predictor.
regr = linear_model.LinearRegression()
preditor_linear_model: object=regr.fit(x_nca, y)
preditor_Pickle= open(url_predictor, 'wb')
print('===== CHARACTER PREDICTOR DONE =====')
p1.dump(preditor_linear_model, preditor_Pickle)
