import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Parametros
url_data = 'dataset/optdigits.tes'
url_predictor = 'dataset/character_macro_predictor'
limite = 1797
n_neighbors = 12
grid = 2
cmapa = sns.color_palette('BrBG', n_colors=10, as_cmap=True)

# Setup
rng = np.random.randint(0, limite-200)
#Carregando o predictor e separando suas funções:
trickbag = joblib.load(open(url_predictor, 'rb'))
predictor = trickbag.named_steps['regressor']
preprocessor = trickbag.named_steps['preprocessor']
knc = trickbag.named_steps['knc']

# Importar o conjunto de dados para Previsão
dataset = pd.read_csv(url_data, sep=',', header=None)
xy_pred_raw = dataset[:]

# Selecionar os intervalos de Previsão
x_pred_raw = xy_pred_raw.iloc[rng:rng+200, 0:64]
x_pred = preprocessor.transform(x_pred_raw)

# Fazer a previsão para a observação de teste
y_pred = predictor.predict(x_pred).astype(int)
print(y_pred)


# Define pontos na malha em cada 'peso' para ... [x_min, x_max]x[y_min, y_max].
x_min, x_max = x_pred[:, 0].min() - 1, x_pred[:, 0].max() + 1
y_min, y_max = x_pred[:, 1].min() - 1, x_pred[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, grid), np.arange(y_min, y_max, grid))
# ... Abrir , pintar e fechar ela.
Z = knc.predict(np.c_[xx.ravel(), yy.ravel()]).astype(int)
Z = Z.reshape(xx.shape)
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, cmap=cmapa)

# Plotagem
sns.scatterplot(
    x=x_pred[:, 0],
    y=x_pred[:, 1],
    marker='X',
    hue=y_pred.ravel(),
    palette=cmapa,
    alpha=1.0,
    edgecolor="black")

# Desenha
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("Classificacao e previsao de caractere numerico\n"
          "(K = %i, weights = '%s')" % (n_neighbors, 'uniform'))
plt.show()