import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import pickle

#Parametros
url_test='dataset/optdigits.tes'
url_predictor='dataset/trickbag_mesh'
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

# Carrega a malha.
pickle.load(open('dataset/mesh_background.fig.pickle', 'rb'))
# Plotagem
sns.scatterplot(
    x=x_nca[:, 0],
    y=x_nca[:, 1],
    marker='o',
    hue=y_predicted.ravel(),
    palette=cmapa,
    alpha=1.0,
    edgecolor="red",
    legend = 'full',)####################>>>> MOSTRA TODAS AS ENTRADAS NO MAPA

#Desenha
plt.title("Classificacao e previsao de caractere numerico\n"
          "(K = %i, weights = '%s', Eficiencia: %.2f%%)" % (n_neighbors, 'uniform', score*100))
plt.show()