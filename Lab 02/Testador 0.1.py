import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn import neighbors

#Parametros
url_test='dataset/optdigits.tes'
url_predictor='dataset/character_macro_predictor'
limite= 1797
n_neighbors = 12
grid = 2
cmapa=sns.color_palette('BrBG',n_colors=10, as_cmap=True)

#Setup
#Carregando o predictor e separando suas funções:
trickbag = joblib.load(open(url_predictor, 'rb'))
predictor = trickbag.named_steps['regressor']
preprocessor = trickbag.named_steps['preprocessor']
knc = trickbag.named_steps['knc']
knc2 = neighbors.KNeighborsClassifier(n_neighbors, weights='uniform')

# Importar o conjunto de dados para Teste
dataset = pd.read_csv(url_test, sep=',', header=None)
xy_test_raw = dataset[:]

# Selecionar os intervalos de teste
x_test = xy_test_raw.iloc[:limite, 0:64]
y_test = xy_test_raw.iloc[:limite, 64:65]

# Reduzir as dimensões das entradas fazer a previsao e ver a eficacia
x_nca=preprocessor.transform(x_test)
y_predicted = predictor.predict(x_nca).astype(int)
print(y_predicted,'\n',  y_test)
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

# Define pontos na malha em cada para ... [x_min, x_max]x[y_min, y_max].
knc2.fit(x_nca,y_predicted)
x_min, x_max = x_nca[:, 0].min() - 1, x_nca[:, 0].max() + 1
y_min, y_max = x_nca[:, 1].min() - 1, x_nca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, grid), np.arange(y_min, y_max, grid))
# ... Abrir , pintar e fechar ela.
Z = knc2.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, cmap=cmapa)

# Plotagem
sns.scatterplot(
    x=x_nca[:, 0],
    y=x_nca[:, 1],
    marker='o',
    hue=y_predicted.astype(int).ravel(),
    palette=cmapa,
    alpha=1.0,
    edgecolor="black")

#Desenha
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("Classificacao e previsao de caractere numerico\n"
          "(K = %i, weights = '%s', Eficiencia: %.2f%%)" % (n_neighbors, 'uniform', right/total*100))
plt.show()