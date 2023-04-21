import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
import pandas as pd
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
################## Não FINALIZADO ################################
limite = int(1750)
# Gerar um número aleatório entre 0 e 3822
rng = np.random.randint(0, limite)
# Definir o número de vizinhos mais próximos
n_neighbors = 12
# Create color maps
cmap_light = ListedColormap(['silver', 'sandybrown', 'darksalmon', 'white','bisque','moccasin','palegreen','paleturquoise','deepskyblue', 'm'])
cmap_bold = ['darkorchid','blue','darkcyan','darkgreen','darkkhaki','gold','sienna','red','black','navy']
cmapa=sns.color_palette('BrBG', as_cmap=True)
# Importar o conjunto de dados
dataset = pd.read_csv('dataset/optdigits.tra', sep=',', header=None)
xyraw = dataset[:]

# Separar as variáveis dependentes e independentes
X = xyraw.iloc[:limite, 0:64]
y = xyraw.iloc[:limite, 64:65]
y = y[64].values.tolist()

# Selecionar a observação de teste
X_test = xyraw.iloc[rng, 0:64]
y_test = xyraw.iloc[rng, 64:65]
X_test = X_test.to_frame().T
y_esperado=np.array(y_test)

#Demonstra o número previsto
print('Linha: ',rng,' Numero esperado:', y_esperado)

# Reduzir as dimensões das entradas
nca = make_pipeline(StandardScaler(), NeighborhoodComponentsAnalysis(n_components=2, random_state=None))
X_nca = nca.fit_transform(X,y)

# Treinar o modelo de KNN
knn = neighbors.KNeighborsClassifier(n_neighbors)
knn.fit(X_nca, y)

# Fazer a previsão para a observação de teste
X_test_nca = nca.transform(X_test)
y_pred = knn.predict(X_test_nca)[:]

# Definição do Grid
h = 1

# Define os agrupamentos e coloração do grafico
#for weights in ["uniform"]:
clf = neighbors.KNeighborsClassifier(n_neighbors, weights='uniform')
clf.fit(X_nca, y)
# point in the mesh [x_min, x_max]x[y_min, y_max].
x_min, x_max = X_nca[:, 0].min() - 1, X_nca[:, 0].max() + 1
y_min, y_max = X_nca[:, 1].min() - 1, X_nca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).astype(int)
Z = Z.reshape(xx.shape)
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, cmap=cmapa)

# Cria um cmap para a previsão.
cmap_pred=[]
cmap_pred.append(cmap_bold[int(y_pred)])

# Plotagem
sns.scatterplot(
    x=X_nca[:, 0],
    y=X_nca[:, 1],
    marker='o',
    hue=y,
    palette=cmapa,
    alpha=1.0,
    edgecolor="black",)
# Plot a predict point
sns.scatterplot(
    legend=False,
    x=X_test_nca[:,0],
    y=X_test_nca[:,1],
    marker="X",
    s=90,
    hue=y_pred,
    palette=cmap_pred,
    alpha=1.0,
    edgecolor="black",)
#Desenha
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("Classificação e previsão de caractere numérico \n "
          "(k = %i, weights = '%s', nº previsto = '%s')" % (n_neighbors, 'uniform', y_pred))
plt.xlabel(y[0])
plt.ylabel(y[1])
plt.show()