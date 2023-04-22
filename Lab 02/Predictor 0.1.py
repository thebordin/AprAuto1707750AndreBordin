import numpy as np
import pandas as pd
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import pickle as p1
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
from sklearn import neighbors, datasets

# Parametros
url_data = 'dataset/optdigits.tes'
url_predictor = 'dataset/character_snca_predictor'
limite = 500  # 1797
n_neighbors = 12
grid = 2
pesos = ["uniform"]  # "uniform" "distance"
cmapa = sns.color_palette('BrBG', n_colors=10, as_cmap=True)

# Setup
nca = make_pipeline(StandardScaler(), NeighborhoodComponentsAnalysis(n_components=2, random_state=None))
knc = neighbors.KNeighborsClassifier(n_neighbors)
ss = StandardScaler()
rng = np.random.randint(0, limite-10)

# Importar o conjunto de dados para Previsão
dataset = pd.read_csv(url_data, sep=',', header=None)
xy_pred_raw = dataset[:]

# Selecionar os intervalos de Previsão
x_pred = xy_pred_raw.iloc[rng:rng+10, 0:64]
y_pred = xy_pred_raw.iloc[rng:rng+10, 64:65]
print (x_pred)
x_pred = nca.fit_transform(x_pred)
print (x_pred)
#Demonstra o número previsto
print('Numeros esperados:', y_pred)

# Fazer a previsão para a observação de teste
predictor = p1.load(open(url_predictor, 'rb'))
y_pred = predictor.predict(x_pred)[:]


# Define pontos na malha em cada 'peso' para ... [x_min, x_max]x[y_min, y_max].
for weights in pesos:
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights='uniform')
    x_min, x_max = x_pred[:, 0].min() - 1, x_pred[:, 0].max() + 1
    y_min, y_max = x_pred[:, 1].min() - 1, x_pred[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, grid), np.arange(y_min, y_max, grid))
    # ... Abrir , pintar e fechar ela.
    knc.fit(x_pred, y_pred)
    Z = knc.predict(np.c_[xx.ravel(), yy.ravel()]).astype(int)
    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, cmap=cmapa)

# Plotagem
sns.scatterplot(
    x=x_pred[:, 0],
    y=x_pred[:, 1],
    marker='o',
    hue=y_pred.values.ravel(),
    palette=cmapa,
    alpha=1.0,
    edgecolor="black")

# Desenha
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("Classificacao e previsao de caractere numerico\n"
          "(K = %i, weights = '%s')" % (n_neighbors, pesos))
plt.show()