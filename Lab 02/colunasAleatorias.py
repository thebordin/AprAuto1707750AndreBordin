import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
import pandas as pd
import random

n_neighbors = 15
# import some data to play with =]
#iris = datasets.load_iris()
dataset = pd.read_csv('dataset/optdigits.tra', sep=',', header=None)
xyraw = dataset[:]
X=xyraw.iloc[:,50:52]
y=xyraw.iloc[:,64:65]
X=np.array(X) #Transforma em array do numpy
Y= y[64].values.tolist()#Transforma de dataframe para lista
y=np.array(Y)
h = 0.02 # step size in the mesh

# Create color maps
cmapa = plt.colormaps['BrBG']

#for weights in ["uniform", "distance"]:
clf = neighbors.KNeighborsClassifier(n_neighbors, weights="uniform")
clf.fit(X, y)
# point in the mesh [x_min, x_max]x[y_min, y_max].
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(figsize=(8, 6))
plt.pcolormesh(xx, yy, Z, cmap=cmapa)
# Plot also the training points
sns.scatterplot(
    x=X[:, 0],
    y=X[:, 1],
    hue=Y,
    palette=cmapa,
    alpha=1.0,
    edgecolor="black",
    legend = 'full')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("3-Class classification (k = %i, weights = '%s')" % (n_neighbors, "uniform"))
#plt.xlabel(Y[0])
plt.ylabel(Y[1])

# Plot a predict point
z=random.randint(0,3822)
print('Linha: ',z,' Numero esperado:', y[z])
sns.scatterplot(
    x=X[z,0],
    y=X[z,1],
    legend=False,
    marker="X",
    s=90,
    hue=Y,
    palette=cmapa,
    alpha=1.0,
    edgecolor="w")
plt.show()
