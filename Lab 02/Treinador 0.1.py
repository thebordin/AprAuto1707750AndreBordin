import numpy as np
import pandas as pd
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import neighbors
from sklearn import linear_model
import joblib

#Treinamento full
limite,url_predictor = 3823,'dataset/character_macro_predictor'
#teste
limite,url_predictor = 500,'dataset/character_macro_predictor_useless'

#Parametros
url_train='dataset/optdigits.tra'
url_test='dataset/optidigits.tes'
n_neighbors = 12
grid = 2
pesos = ["uniform"]
cmapa=sns.color_palette('BrBG',n_colors=10, as_cmap=True)

#Setup
# Criar o pipeline com as etapas de pré-processamento e modelo
trickbag = Pipeline([
    ('preprocessor', Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('nca', NeighborhoodComponentsAnalysis(n_components=2))
    ])),
    ('knc', neighbors.KNeighborsClassifier(n_neighbors, weights='uniform')),
    ('regressor', linear_model.LinearRegression())
])
preprocessor = trickbag.named_steps['preprocessor']
knc = trickbag.named_steps['knc']
regressor = trickbag.named_steps['regressor']

# Importar o conjunto de dados para Treino
dataset = pd.read_csv(url_train, sep=',', header=None)
xy_raw = dataset[:]

# Separar as variáveis dependentes e independentes
x_raw = xy_raw.iloc[:limite, 0:64]
y_raw = xy_raw.iloc[:limite, 64:65]
y = y_raw[64].values.tolist()

# Treinar o motor e reduzir x_raw
x_nca = preprocessor.fit_transform(x_raw, y) # Treina redução
regressor.fit(x_nca,y) # Treina o predictor
knc.fit(x_nca,y) # Define os agrupamentos e coloração do grafico após o treinamento

# Define pontos na malha para ... [x_min, x_max]x[y_min, y_max].
x_min, x_max = x_nca[:, 0].min() - 1, x_nca[:, 0].max() + 1
y_min, y_max = x_nca[:, 1].min() - 1, x_nca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, grid), np.arange(y_min, y_max, grid))
# ... Abrir , pintar e fechar ela.
Z = knc.predict(np.c_[xx.ravel(), yy.ravel()]).astype(int)
Z = Z.reshape(xx.shape)
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, cmap=cmapa)

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
          "(k = %i, weights = '%s')" % (n_neighbors, 'uniform'))
plt.show()

#Gravar o predictor.
predictor_file= open(url_predictor, 'wb')
joblib.dump(trickbag, predictor_file)
print('===== CHARACTER PREDICTOR DONE =====')
