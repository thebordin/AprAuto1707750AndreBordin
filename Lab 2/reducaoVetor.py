import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
import pandas as pd
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import random
################## NÃO FINALIZADO ################################
rng=random.randint(0,3822)
n_neighbors = 15
# import some data to play with =]
dataset = pd.read_csv('dataset/optdigits.tra', sep=',', header=None)
xyraw = dataset[:]
X=xyraw.iloc[:,0:64]
y=xyraw.iloc[:,64:65]
#X=X.values.tolist() #Transforma de dataframe para lista
X=np.array(X) #Transforma em array do numpy
y= y[64].values.tolist()
y=np.array(y)
X_test=xyraw.iloc[rng,0:64]
y_test=xyraw.iloc[rng,64:65]
X_test=X_test.values.tolist()
print((X_test))
y_esperado=np.array(y_test)
print('Linha: ',rng,' Numero esperado:', y_esperado)

### Padronização das entradas:
nca = make_pipeline(StandardScaler(), NeighborhoodComponentsAnalysis(
n_components=64, random_state=None), )
nca.fit(X, y)
knn = neighbors.KNeighborsClassifier(n_neighbors)
knn.fit(nca.transform(X), y)
print(nca.predict(X_test))

'''#### Redução do vetor
knn = neighbors.KNeighborsClassifier(n_neighbors)
knn.fit(X, y)
print(knn.score(X_test, y_test))

nca = NeighborhoodComponentsAnalysis(n_components=2, random_state=None)
nca.fit(X, y)
knn.fit(nca.transform(X), y)
#print(knn.score(nca.transform(X_test), y_test))'''

XX=np.c_[nca.transform(X_test),y_test]
XX=XX[np.argsort(XX[:, 2])]
sns.scatterplot(x=XX[:, 0], y=XX[:, 1],
hue=Y,
palette=cmap_bold, alpha=1.0, edgecolor="black",)
plt.show()

'''h = 0.02 # step size in the mesh
print(X)

# Create color maps
cmap_light = ListedColormap(["orange", "cyan", "cornflowerblue"])
cmap_bold = ["darkorange", "c", "darkblue"]

for weights in ["uniform", "distance"]:
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
    clf.fit(X, y)
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, cmap=cmap_light)

# Plot also the training points
sns.scatterplot(
    x=X[:, 0],
    y=X[:, 1],
    hue=Y,
    palette=cmap_bold,
    alpha=1.0,
    edgecolor="black",)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("3-Class classification (k = %i, weights = '%s')" % (n_neighbors, weights))
plt.xlabel(Y[0])
plt.ylabel(Y[1])

# Plot a predict point

sns.scatterplot(
    x=X[rng,0],
    y=X[rng,1],
    marker="X",
    s=90,
    hue=Y,
    palette=cmap_bold,
    alpha=1.0,
    edgecolor="w",)
plt.show()
'''