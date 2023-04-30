import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.colors as mcolors
import seaborn as sns
import joblib
from sklearn import neighbors

#Parametros
url_test='dataset/optdigits.tes'
url_predictor='dataset/character_macro_predictor'
limite= 1797
n_neighbors = 12
grid = 2

#Setup
#Carregando o predictor e separando suas funções:
trickbag = joblib.load(open(url_predictor, 'rb'))
preprocessor = trickbag.named_steps['preprocessor']
knc = trickbag.named_steps['knc']
cmapa = trickbag.named_steps['cmapa']
#Fim Setup

# Importar o conjunto de dados para Teste
dataset = pd.read_csv(url_test, sep=',', header=None)
xy_test_raw = dataset[:]

# Selecionar os intervalos de teste
x_test = xy_test_raw.iloc[:limite, 0:64]
y_test = xy_test_raw.iloc[:limite, 64:65]

# Reduzir as dimensões das entradas fazer a previsao e ver a eficacia
x_nca=preprocessor.transform(x_test)
y_predicted = knc.predict(x_nca)
score = knc.score(x_nca,y_test)
print('Score do K Neighbous Classifier(Knc.score): %.2f'% (score*100))

# Define pontos na malha em cada para ... [x_min, x_max]x[y_min, y_max].
knc.fit(x_nca,y_predicted)
x_min, x_max = x_nca[:, 0].min() - 1, x_nca[:, 0].max() + 1
y_min, y_max = x_nca[:, 1].min() - 1, x_nca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, grid), np.arange(y_min, y_max, grid))
# ... Abrir , pintar e fechar ela.
Z = knc.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.figure(figsize=(8, 6))
plt.pcolormesh(xx, yy, Z, cmap=cmapa)

# Plotagem
sns.scatterplot(
    x=x_nca[:, 0],
    y=x_nca[:, 1],
    marker='o',
    hue=y_predicted.ravel(),
    palette=cmapa,
    alpha=1.0,
    edgecolor="red",
    legend = 'full')####################>>>> MOSTRA TODAS AS ENTRADAS NO MAPA

#Desenha
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("Classificacao e previsao de caractere numerico\n"
          "(K = %i, weights = '%s', Eficiencia: %.2f%%)" % (n_neighbors, 'uniform', score*100))
plt.show()