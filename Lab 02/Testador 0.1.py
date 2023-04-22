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

#Parametros
url_test='dataset/optdigits.tes'
url_predictor='dataset/character_snca_predictor'
limite= 500
n_neighbors = 12
grid = 2
pesos = ["uniform"] #"uniform" "distance"
cmapa=sns.color_palette('BrBG',n_colors=10, as_cmap=True)

#Setup
nca = make_pipeline(StandardScaler(), NeighborhoodComponentsAnalysis(n_components=2, random_state=None))
knc = neighbors.KNeighborsClassifier(n_neighbors)

# Importar o conjunto de dados para Teste
dataset = pd.read_csv(url_test, sep=',', header=None)
xy_test_raw = dataset[:]

# Selecionar os intervalos de teste
x_test = xy_test_raw.iloc[:limite, 0:64]
y_test = xy_test_raw.iloc[:limite, 64:65]
#x_test = x_test.to_frame().T

# Reduzir as dimensões das entradas
x_test_nca = nca.fit_transform(x_test,y_test)

#Preditor
predictor = p1.load(open(url_predictor, 'rb'))
y_predicted = predictor.predict(x_test_nca)
match=y_predicted-np.array(y_test)
right = 0
wrong = 0
total = 0
for i in match[64]:
    z=int(i)
    total=total+1
    if z==0:
        right += 1
    else: wrong +=1
print('Frequencia de Certos:', right/total, 'Frequencia de errados:', wrong/total)

# Define pontos na malha em cada 'peso' para ... [x_min, x_max]x[y_min, y_max].
for weights in pesos:
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights='uniform')
    x_min, x_max = x_test_nca[:, 0].min() - 1, x_test_nca[:, 0].max() + 1
    y_min, y_max = x_test_nca[:, 1].min() - 1, x_test_nca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, grid), np.arange(y_min, y_max, grid))
    # ... Abrir , pintar e fechar ela.
    knc.fit(x_test_nca,y_test)
    Z = knc.predict(np.c_[xx.ravel(), yy.ravel()]).astype(int)
    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, cmap=cmapa)
x_test_nca = np.squeeze(x_test_nca)
# Plotagem
sns.scatterplot(
    x=x_test_nca[:, 0],
    y=x_test_nca[:, 1],
    marker='o',
    hue=y_test,
    palette=cmapa,
    alpha=1.0,
    edgecolor="black")

#Desenha
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("Classificação e previsão de caractere numérico \n "
          "(K = %i, weights = '%s', ''%'' de acertos: %f)" % (n_neighbors, pesos, right/total))
plt.show()